"""Phase 3.2: CPU vs CPU+iGPU device split for the Qwen3.6 split-IR engine.

The split architecture naturally maps to heterogeneous devices: the backbone
(attention + router + shared expert -- dense, runs every token) can live on the
Arc iGPU, while the routed experts (sparse, only top-k fire) stay on the CPU
where lazy load + LRU eviction is cheap. This script measures one decode step
under two placements:

  A) all-CPU              -- backbone IRs + expert IRs on CPU
  B) backbone=GPU, experts=CPU

and verifies the logits match across placements (within fp tolerance).

If the GPU device can't compile a graph (unsupported op), it reports the
failure rather than crashing -- that is itself a useful Phase 3.2 finding.

Run from venv-qwen.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import openvino as ov  # noqa: E402

from src.configuration import make_toy_config  # noqa: E402
from src.modeling import QwenForCausalLM  # noqa: E402
from src.split_inference import (  # noqa: E402
    QwenLayerBackboneFull,
    QwenLayerBackboneLinear,
    build_standalone_expert,
    combine_layer_output,
    extract_expert_state_dict,
)


def _backbone_ov(model, layer_idx, max_seq):
    layer = model.model.layers[layer_idx]
    cfg = model.config
    if layer.layer_type == "full_attention":
        wrapper = QwenLayerBackboneFull(layer)
        ex = (
            torch.randn(1, 1, cfg.hidden_size),
            torch.randn(1, 1, cfg.partial_rotary_dim),
            torch.randn(1, 1, cfg.partial_rotary_dim),
            torch.zeros(1, cfg.num_key_value_heads, max_seq, cfg.head_dim),
            torch.zeros(1, cfg.num_key_value_heads, max_seq, cfg.head_dim),
            torch.tensor(0, dtype=torch.long),
        )
    else:
        wrapper = QwenLayerBackboneLinear(layer)
        ex = (
            torch.randn(1, 1, cfg.hidden_size),
            torch.zeros(1, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim),
            torch.zeros(1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim),
        )
    return ov.convert_model(wrapper, example_input=ex), layer.layer_type


def _expert_ov(model, layer_idx, expert_idx):
    cfg = model.config
    sd = extract_expert_state_dict(model, layer_idx, expert_idx)
    expert = build_standalone_expert(cfg, weights=sd)
    return ov.convert_model(expert, example_input=torch.randn(1, cfg.hidden_size))


@torch.no_grad()
def _run(model, backbone_compiled, expert_compiled, input_ids, position_ids, state):
    mdl = model.model
    x = mdl.embed_tokens(input_ids)
    cos = mdl.rope_cos[position_ids.long()].to(x.dtype)
    sin = mdl.rope_sin[position_ids.long()].to(x.dtype)
    write_pos = position_ids[0, 0]
    k = [t.clone() for t in state["k_caches"]]
    v = [t.clone() for t in state["v_caches"]]
    conv = [t.clone() for t in state["conv_states"]]
    rec = [t.clone() for t in state["rec_states"]]

    full_i = lin_i = 0
    for li, layer in enumerate(mdl.layers):
        ir, lt = backbone_compiled[li]
        if lt == "full_attention":
            outs = ir([x.numpy(), cos.numpy(), sin.numpy(), k[full_i].numpy(), v[full_i].numpy(), write_pos.numpy()])
            x_post = torch.from_numpy(outs[ir.outputs[0]]); flat = torch.from_numpy(outs[ir.outputs[1]])
            topk_idx = torch.from_numpy(outs[ir.outputs[2]]); topk_w = torch.from_numpy(outs[ir.outputs[3]])
            shared = torch.from_numpy(outs[ir.outputs[4]])
            k[full_i] = torch.from_numpy(outs[ir.outputs[5]]); v[full_i] = torch.from_numpy(outs[ir.outputs[6]])
            full_i += 1
        else:
            outs = ir([x.numpy(), conv[lin_i].numpy(), rec[lin_i].numpy()])
            x_post = torch.from_numpy(outs[ir.outputs[0]]); flat = torch.from_numpy(outs[ir.outputs[1]])
            topk_idx = torch.from_numpy(outs[ir.outputs[2]]); topk_w = torch.from_numpy(outs[ir.outputs[3]])
            shared = torch.from_numpy(outs[ir.outputs[4]])
            conv[lin_i] = torch.from_numpy(outs[ir.outputs[5]]); rec[lin_i] = torch.from_numpy(outs[ir.outputs[6]])
            lin_i += 1

        BS, K = topk_idx.shape
        eo = torch.zeros(BS, K, flat.shape[-1])
        for t in range(BS):
            for kk in range(K):
                e = int(topk_idx[t, kk].item())
                eir = expert_compiled[li][e]
                eo[t, kk] = torch.from_numpy(eir({eir.inputs[0]: flat[t:t+1].numpy()})[eir.outputs[0]]).squeeze(0)
        x = combine_layer_output(x_post, eo, topk_w, shared)

    x = mdl.norm(x)
    return model.lm_head(x)


def _compile_all(core, model, backbones, experts, backbone_dev, expert_dev):
    bb = [(core.compile_model(m, backbone_dev), lt) for (m, lt) in backbones]
    ex = [[core.compile_model(m, expert_dev) for m in layer] for layer in experts]
    return bb, ex


def _time_step(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    core = ov.Core()
    devices = core.available_devices
    print("=" * 70)
    print("Qwen3.6 device-split benchmark")
    print(f"available devices: {devices}")
    for d in devices:
        try:
            print(f"  {d} = {core.get_property(d, 'FULL_DEVICE_NAME')}")
        except Exception:
            pass
    print("=" * 70)
    if "GPU" not in devices:
        print("No GPU device -- cannot run the iGPU split. Exiting.")
        return 1

    cfg = make_toy_config(num_layers=4, full_attention_interval=4, hidden_size=256,
                          num_experts=16, top_k=4, moe_intermediate=128,
                          num_attention_heads=8, num_key_value_heads=2, head_dim=32,
                          linear_num_key_heads=4, linear_num_value_heads=8,
                          linear_key_head_dim=32, linear_value_head_dim=32)
    print(f"config: {cfg.num_hidden_layers} layers, {cfg.num_experts} experts top-{cfg.num_experts_per_tok}, hidden={cfg.hidden_size}")

    torch.manual_seed(0)
    model = QwenForCausalLM(cfg)
    model.eval()
    max_seq = 8

    print("\nconverting backbone + expert IRs ...")
    backbones = [_backbone_ov(model, i, max_seq) for i in range(cfg.num_hidden_layers)]
    experts = [[_expert_ov(model, i, e) for e in range(cfg.num_experts)] for i in range(cfg.num_hidden_layers)]

    state = model.empty_state(batch=1, max_seq=max_seq)
    input_ids = torch.tensor([[42]])
    position_ids = torch.tensor([[0]])

    # A) all-CPU
    print("\n[A] all-CPU ...")
    bb_cpu, ex_cpu = _compile_all(core, model, backbones, experts, "CPU", "CPU")
    logits_cpu = _run(model, bb_cpu, ex_cpu, input_ids, position_ids, state)
    t_cpu = _time_step(lambda: _run(model, bb_cpu, ex_cpu, input_ids, position_ids, state))
    print(f"    {t_cpu*1e3:.2f} ms/step")

    # B) backbone on GPU, experts on CPU
    print("\n[B] backbone=GPU, experts=CPU ...")
    try:
        bb_gpu, ex_cpu2 = _compile_all(core, model, backbones, experts, "GPU", "CPU")
        logits_gpu = _run(model, bb_gpu, ex_cpu2, input_ids, position_ids, state)
        t_gpu = _time_step(lambda: _run(model, bb_gpu, ex_cpu2, input_ids, position_ids, state))
        print(f"    {t_gpu*1e3:.2f} ms/step")
        diff = (logits_cpu - logits_gpu).abs().max().item()
        print(f"\nnumerical match CPU vs GPU-backbone: max abs-diff = {diff:.3e} "
              f"({'OK' if diff < 5e-3 else 'CHECK'})")
        print(f"backbone placement speedup (all-CPU / split): {t_cpu/t_gpu:.2f}x")
        print("\nnote: toy backbone is tiny -- GPU dispatch overhead can dominate at this")
        print("      size. The split's value at full scale is freeing CPU/RAM for the")
        print("      sparse experts while the dense 2048-wide backbone runs on the iGPU.")
    except Exception as e:  # noqa: BLE001
        print(f"    GPU compile/run FAILED: {type(e).__name__}: {str(e)[:120]}")
        print("    (finding: some ported op is unsupported on the Arc GPU plugin)")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
