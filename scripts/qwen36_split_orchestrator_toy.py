"""Phase 2.3 end-to-end demo: split-IR orchestration on the toy model.

Converts each decoder layer's backbone (attention + router + shared expert)
to an OV IR and each routed expert to its own OV IR, then runs one decode
step via:

    for each layer L:
        x_post_attn, hidden_for_experts, topk_idx, topk_w, shared, state_out
            = layer_backbone_IR_L(x_in, ..., state_in)
        for k in 0..top_k-1:
            e_out_k = expert_IRs[L][topk_idx[k]](hidden_for_experts)
        x_out = combine(x_post_attn, e_out_*, topk_w, shared)

and compares logits + final state against the monolithic PyTorch forward.

Note: This converts num_layers + num_layers*num_experts IRs. For the toy
that's 2 + 2*4 = 10 IRs; for the real model it would be 40 + 40*256 = 10,280
expert IRs (Phase 2 will introduce weight-as-input to amortize this).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import openvino as ov  # noqa: E402

from src.qwen36.configuration_qwen36 import make_toy_config  # noqa: E402
from src.qwen36.modeling_qwen36 import QwenForCausalLM  # noqa: E402
from src.qwen36.split_inference import (  # noqa: E402
    QwenLayerBackboneFull,
    QwenLayerBackboneLinear,
    build_standalone_expert,
    combine_layer_output,
    extract_expert_state_dict,
)


def convert_one_layer_backbone(model: QwenForCausalLM, layer_idx: int, max_seq: int):
    """Trace and compile one layer-backbone IR. Returns (compiled_model, layer_type)."""
    layer = model.model.layers[layer_idx]
    cfg = model.config

    if layer.layer_type == "full_attention":
        wrapper = QwenLayerBackboneFull(layer)
        x = torch.randn(1, 1, cfg.hidden_size)
        cos = torch.randn(1, 1, cfg.partial_rotary_dim)
        sin = torch.randn(1, 1, cfg.partial_rotary_dim)
        k = torch.zeros(1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
        v = torch.zeros_like(k)
        write_pos = torch.tensor(0, dtype=torch.long)
        example = (x, cos, sin, k, v, write_pos)
    else:
        wrapper = QwenLayerBackboneLinear(layer)
        x = torch.randn(1, 1, cfg.hidden_size)
        conv = torch.zeros(1, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim)
        rec = torch.zeros(
            1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim,
        )
        example = (x, conv, rec)

    ov_model = ov.convert_model(wrapper, example_input=example)
    return ov.Core().compile_model(ov_model, "CPU"), layer.layer_type


def convert_one_expert(model: QwenForCausalLM, layer_idx: int, expert_idx: int):
    cfg = model.config
    sd = extract_expert_state_dict(model, layer_idx, expert_idx)
    expert = build_standalone_expert(cfg, weights=sd)
    example = torch.randn(1, cfg.hidden_size)
    ov_model = ov.convert_model(expert, example_input=example)
    return ov.Core().compile_model(ov_model, "CPU")


@torch.no_grad()
def orchestrate_one_step(
    model: QwenForCausalLM,
    layer_irs: list[tuple[object, str]],
    expert_irs: list[list[object]],  # expert_irs[layer][expert] -> compiled IR
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    state: dict,
) -> tuple[torch.Tensor, dict]:
    mdl = model.model
    x = mdl.embed_tokens(input_ids)
    cos = mdl.rope_cos[position_ids.long()].to(x.dtype)
    sin = mdl.rope_sin[position_ids.long()].to(x.dtype)
    write_pos = position_ids[0, 0]

    k_caches = [t.clone() for t in state["k_caches"]]
    v_caches = [t.clone() for t in state["v_caches"]]
    conv_states = [t.clone() for t in state["conv_states"]]
    rec_states = [t.clone() for t in state["rec_states"]]

    full_i = 0
    lin_i = 0
    for layer_idx, layer in enumerate(mdl.layers):
        ir, layer_type = layer_irs[layer_idx]
        if layer_type == "full_attention":
            outs = ir([x.numpy(), cos.numpy(), sin.numpy(),
                       k_caches[full_i].numpy(), v_caches[full_i].numpy(),
                       write_pos.numpy()])
            x_post_attn = torch.from_numpy(outs[ir.outputs[0]])
            flat = torch.from_numpy(outs[ir.outputs[1]])
            topk_idx = torch.from_numpy(outs[ir.outputs[2]])
            topk_w = torch.from_numpy(outs[ir.outputs[3]])
            shared = torch.from_numpy(outs[ir.outputs[4]])
            k_caches[full_i] = torch.from_numpy(outs[ir.outputs[5]])
            v_caches[full_i] = torch.from_numpy(outs[ir.outputs[6]])
            full_i += 1
        else:
            outs = ir([x.numpy(), conv_states[lin_i].numpy(), rec_states[lin_i].numpy()])
            x_post_attn = torch.from_numpy(outs[ir.outputs[0]])
            flat = torch.from_numpy(outs[ir.outputs[1]])
            topk_idx = torch.from_numpy(outs[ir.outputs[2]])
            topk_w = torch.from_numpy(outs[ir.outputs[3]])
            shared = torch.from_numpy(outs[ir.outputs[4]])
            conv_states[lin_i] = torch.from_numpy(outs[ir.outputs[5]])
            rec_states[lin_i] = torch.from_numpy(outs[ir.outputs[6]])
            lin_i += 1

        # Dispatch the K selected experts for this layer, per token
        BS, K = topk_idx.shape
        hidden = flat.shape[-1]
        expert_outs = torch.zeros(BS, K, hidden)
        for t in range(BS):
            for k in range(K):
                e_idx = int(topk_idx[t, k].item())
                e_ir = expert_irs[layer_idx][e_idx]
                token_in = flat[t : t + 1].numpy()  # (1, hidden)
                e_result = e_ir({e_ir.inputs[0]: token_in})
                expert_outs[t, k] = torch.from_numpy(e_result[e_ir.outputs[0]]).squeeze(0)

        x = combine_layer_output(x_post_attn, expert_outs, topk_w, shared)

    x = mdl.norm(x)
    logits = model.lm_head(x)
    new_state = {
        "k_caches": k_caches, "v_caches": v_caches,
        "conv_states": conv_states, "rec_states": rec_states,
    }
    return logits, new_state


def main() -> int:
    cfg = make_toy_config()
    print(f"toy cfg: {len(cfg.layer_types)} layers, {cfg.num_experts} experts top-{cfg.num_experts_per_tok}")

    torch.manual_seed(0)
    model = QwenForCausalLM(cfg)
    model.eval()

    max_seq = 8

    # Reference: monolithic torch forward
    state = model.empty_state(batch=1, max_seq=max_seq)
    input_ids = torch.tensor([[42]])
    position_ids = torch.tensor([[0]])
    with torch.no_grad():
        mono_logits, mono_k, mono_v, mono_conv, mono_rec = model(
            input_ids, position_ids,
            state["k_caches"], state["v_caches"], state["conv_states"], state["rec_states"],
        )

    # Convert layer backbones + experts
    t0 = time.time()
    layer_irs = [convert_one_layer_backbone(model, i, max_seq) for i in range(len(cfg.layer_types))]
    print(f"  converted {len(layer_irs)} layer backbones in {time.time()-t0:.1f}s")

    t0 = time.time()
    expert_irs: list[list[object]] = []
    for layer_idx in range(len(cfg.layer_types)):
        expert_irs.append([convert_one_expert(model, layer_idx, e) for e in range(cfg.num_experts)])
    print(f"  converted {len(layer_irs) * cfg.num_experts} expert IRs in {time.time()-t0:.1f}s")

    # Orchestrate one decode step
    t0 = time.time()
    split_logits, split_state = orchestrate_one_step(
        model, layer_irs, expert_irs, input_ids, position_ids, state,
    )
    print(f"  orchestrated 1 step in {time.time()-t0:.2f}s")

    # Compare
    diff = (mono_logits - split_logits).abs()
    print(f"\nmonolithic-torch vs split-OV logits max abs-diff: {diff.max().item():.3e}")
    ok_logits = diff.max().item() < 5e-4

    ok_k = all(torch.allclose(a, b.float(), atol=5e-4) for a, b in zip(mono_k, split_state["k_caches"]))
    ok_conv = all(torch.allclose(a, b.float(), atol=5e-4) for a, b in zip(mono_conv, split_state["conv_states"]))
    ok_rec = all(torch.allclose(a, b.float(), atol=5e-4) for a, b in zip(mono_rec, split_state["rec_states"]))
    print(f"state match: k={ok_k} conv={ok_conv} rec={ok_rec}")

    ok = ok_logits and ok_k and ok_conv and ok_rec
    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
