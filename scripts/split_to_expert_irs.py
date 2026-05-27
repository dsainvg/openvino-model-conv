"""2.1 — Split the toy DeepSeek-V4 into backbone + per-expert OpenVINO IRs.

The goal of this step is to validate that the V4 forward can be decomposed into
small, independently-loadable IRs without losing numerical equivalence with the
monolithic toy IR. This is the foundation for 2.2 (expert offloading) and 2.3
(per-expert real-V4 weight conversion).

Segments emitted (prefill-only, no KV cache for this step):
  - ov_ir_toy/expert_split/embed.xml           input_ids → h_expanded
  - ov_ir_toy/expert_split/pre_moe_L{i}.xml    h, ids → (y2_flat, x_residual, post2, comb2,
                                                          gate_weights, gate_indices,
                                                          shared_out_flat)
  - ov_ir_toy/expert_split/expert_L{i}_E{e}.xml  x_flat → expert_out_flat
  - ov_ir_toy/expert_split/post_moe_L{i}.xml   moe_out, x_residual, post2, comb2 → h_out
  - ov_ir_toy/expert_split/final.xml           h → logits

The orchestrator runs them in sequence, dispatching only the experts that the
gate selected on any token (np.unique(indices)). Output is compared against the
monolithic toy IR's prefill logits (must greedy-match; abs/rel diff reported).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch
import torch.nn as nn
import openvino as ov

from deepseek_v4 import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


# ------------------------------------------------------------------------------
# Wrapper modules — thin slices over the existing model's submodules.
# These hold references (no parameter copies) so each emitted IR contains only
# the weights it actually uses.
# ------------------------------------------------------------------------------
class EmbedWrapper(nn.Module):
    """input_ids [B,S] → h_expanded [B,S,H,dim]."""
    def __init__(self, model):
        super().__init__()
        self.embed = model.model.embed
        self.hc_mult = model.config.hc_mult

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        return h


class PreMoeWrapper(nn.Module):
    """Runs the attention sub-block + ffn pre-MoE up to and including the gate.

    Returns:
      y2_flat        [N, dim]  pre-MoE per-token input for the experts (N = B*S)
      x_residual     [B, S, H, dim]  carry for post-MoE HC combine
      post2          [B, S, H]
      comb2          [B, S, H, H]
      gate_weights   [N, topk]
      gate_indices   [N, topk]  int64
      shared_out     [N, dim]   shared-expert output (computed alongside the gate
                                because it shares the per-token weights with the
                                MoE input; safe to keep here while we're still
                                touching y2_flat).
    """
    def __init__(self, model, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.block = model.model.layers[layer_idx]
        # Hold references to the rope buffers so the trace can slice them.
        self.rope_cos = model.model.rope_cos
        self.rope_sin = model.model.rope_sin

    def forward(self, h_in, input_ids):
        block = self.block
        S = input_ids.size(1)
        cos_full = self.rope_cos[:S]
        sin_full = self.rope_sin[:S]

        # Attention sub-block (mirror Block.forward; prefill semantics → past = 0).
        y_full, post_full, comb_full = block._hc_pre(
            h_in, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
        )
        y_full = block.attn_norm(y_full)
        y_new = block.attn(y_full, cos_full, sin_full, seqlen_new=S)
        x_new = block._hc_post(y_new, h_in, post_full, comb_full)

        # FFN pre-MoE (HC pre + norm).
        y2, post2, comb2 = block._hc_pre(
            x_new, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
        )
        y2 = block.ffn_norm(y2)

        # Flatten and run the gate + shared expert.
        b, s, d = y2.shape
        y2_flat = y2.reshape(b * s, d)
        ids_flat = input_ids.reshape(b * s)
        weights, indices = block.ffn.gate(y2_flat, ids_flat)
        shared_out = block.ffn.shared_experts(y2_flat)

        return y2_flat, x_new, post2, comb2, weights, indices.to(torch.int64), shared_out


class ExpertWrapper(nn.Module):
    """Single Expert as a standalone module: x_flat [N,dim] → out_flat [N,dim]."""
    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, x_flat):
        return self.expert(x_flat)


class PostMoeWrapper(nn.Module):
    """Applies _hc_post over (combined MoE output, FFN residual) → block output."""
    def __init__(self, model, layer_idx):
        super().__init__()
        self.block = model.model.layers[layer_idx]

    def forward(self, moe_out_bsd, x_residual, post2, comb2):
        # moe_out_bsd is the post-aggregation, post-shared-add MoE output [B,S,dim].
        return self.block._hc_post(moe_out_bsd, x_residual, post2, comb2)


class FinalHeadWrapper(nn.Module):
    """h [B,S,H,dim] → logits [B,S,V]."""
    def __init__(self, model):
        super().__init__()
        self.inner = model.model      # DeepseekV4Model
        self.lm_head = model.lm_head

    def forward(self, h):
        h = self.inner._hc_head_reduce(h)
        h = self.inner.norm(h)
        return self.lm_head(h)


# ------------------------------------------------------------------------------
# Conversion helpers
# ------------------------------------------------------------------------------
def _convert(wrapped, example_inputs, input_shapes_and_types, output_names, save_path):
    ov_model = ov.convert_model(
        wrapped, example_input=example_inputs, input=input_shapes_and_types
    )
    for i, name in enumerate(output_names):
        ov_model.outputs[i].set_names({name})
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(save_path), compress_to_fp16=False)
    return ov_model


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    model = DeepseekV4ForCausalLM(cfg).eval()

    L = cfg.num_hidden_layers
    H = cfg.hc_mult
    d = cfg.hidden_size
    E = cfg.n_routed_experts
    topk = cfg.num_experts_per_tok
    V = cfg.vocab_size

    B, S = 1, 128
    input_ids = torch.randint(0, V, (B, S))

    print(f"Config: L={L} layers, H={H} hc_mult, dim={d}, E={E} experts, topk={topk}")
    print(f"Input shape: ({B},{S})")

    print("\n=== Reference (monolithic PyTorch) ===")
    with torch.inference_mode():
        ref_logits = model(input_ids=input_ids).logits
    print(f"  ref logits {tuple(ref_logits.shape)}  greedy_top={ref_logits[0,-1].argmax().item()}")

    out_dir = ROOT / "ov_ir_toy" / "expert_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Tracing and saving split IRs ===")

    # 1. Embed.
    embed_w = EmbedWrapper(model).eval()
    _convert(
        embed_w,
        example_inputs=(input_ids,),
        input_shapes_and_types=[([-1, -1], ov.Type.i64)],
        output_names=["h"],
        save_path=out_dir / "embed.xml",
    )
    print(f"  saved embed.xml")

    # 2. Per-layer pre_moe / post_moe / experts.
    # We need an example h_in for pre_moe. Compute one by running embed in PT.
    with torch.no_grad():
        h_in_example = embed_w(input_ids).clone()

    for i in range(L):
        pre_w = PreMoeWrapper(model, i).eval()

        # Trace pre_moe.
        _convert(
            pre_w,
            example_inputs=(h_in_example, input_ids),
            input_shapes_and_types=[
                ([-1, -1, H, d], ov.Type.f32),
                ([-1, -1], ov.Type.i64),
            ],
            output_names=["y2_flat", "x_residual", "post2", "comb2",
                          "gate_weights", "gate_indices", "shared_out"],
            save_path=out_dir / f"pre_moe_L{i}.xml",
        )

        # Run pre_moe in PT to get an example for downstream segments.
        with torch.no_grad():
            y2_flat_ex, x_res_ex, post2_ex, comb2_ex, _, _, _ = pre_w(h_in_example, input_ids)
            y2_flat_ex = y2_flat_ex.clone()
            x_res_ex = x_res_ex.clone()
            post2_ex = post2_ex.clone()
            comb2_ex = comb2_ex.clone()

        # Per-expert IRs.
        block = model.model.layers[i]
        for e in range(E):
            expert_w = ExpertWrapper(block.ffn.experts[e]).eval()
            _convert(
                expert_w,
                example_inputs=(y2_flat_ex,),
                input_shapes_and_types=[([-1, d], ov.Type.f32)],
                output_names=["expert_out"],
                save_path=out_dir / f"expert_L{i}_E{e}.xml",
            )

        # post_moe IR.
        moe_out_ex = torch.zeros(B, S, d)  # placeholder for tracing
        post_w = PostMoeWrapper(model, i).eval()
        _convert(
            post_w,
            example_inputs=(moe_out_ex, x_res_ex, post2_ex, comb2_ex),
            input_shapes_and_types=[
                ([-1, -1, d], ov.Type.f32),
                ([-1, -1, H, d], ov.Type.f32),
                ([-1, -1, H], ov.Type.f32),
                ([-1, -1, H, H], ov.Type.f32),
            ],
            output_names=["block_out"],
            save_path=out_dir / f"post_moe_L{i}.xml",
        )

        # Advance h_in_example through this layer (via PT) for the next layer's trace.
        with torch.no_grad():
            block_out, _ = block(h_in_example, input_ids, model.model.rope_cos[:S], model.model.rope_sin[:S])
        h_in_example = block_out.clone()

        print(f"  saved pre_moe_L{i}.xml + {E} experts + post_moe_L{i}.xml")

    # 3. Final head.
    final_w = FinalHeadWrapper(model).eval()
    _convert(
        final_w,
        example_inputs=(h_in_example,),
        input_shapes_and_types=[([-1, -1, H, d], ov.Type.f32)],
        output_names=["logits"],
        save_path=out_dir / "final.xml",
    )
    print(f"  saved final.xml")

    print(f"\nTotal IRs saved: 1 (embed) + {L} pre + {L * E} experts + {L} post + 1 (final) = {2 + L * (E + 2)}")

    # --------------------------------------------------------------------------
    # Orchestrate: load all IRs, run end-to-end, compare to reference.
    # --------------------------------------------------------------------------
    print("\n=== Orchestrated OV inference (split IRs) ===")
    core = ov.Core()
    embed_c = core.compile_model(str(out_dir / "embed.xml"), "CPU")
    final_c = core.compile_model(str(out_dir / "final.xml"), "CPU")
    pre_c = [core.compile_model(str(out_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(out_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    expert_c = [
        [core.compile_model(str(out_dir / f"expert_L{i}_E{e}.xml"), "CPU") for e in range(E)]
        for i in range(L)
    ]
    print(f"  compiled {1 + L + L * E + L + 1} IRs")

    input_ids_np = input_ids.numpy().astype(np.int64)
    h = embed_c([input_ids_np])[0]  # [B, S, H, d]

    for i in range(L):
        out = pre_c[i]([h, input_ids_np])
        y2_flat = out[0]      # [N, d]
        x_res = out[1]        # [B, S, H, d]
        post2 = out[2]        # [B, S, H]
        comb2 = out[3]        # [B, S, H, H]
        weights = out[4]      # [N, topk]
        indices = out[5]      # [N, topk] int64
        shared_out = out[6]   # [N, d]

        N = y2_flat.shape[0]
        gate_mat = np.zeros((N, E), dtype=np.float32)
        np.put_along_axis(gate_mat, indices.astype(np.int64), weights.astype(np.float32), axis=1)

        moe_out_flat = np.zeros((N, d), dtype=np.float32)
        active = np.unique(indices).tolist()
        for e in active:
            expert_out = expert_c[i][e]([y2_flat])[0]  # [N, d]
            moe_out_flat += gate_mat[:, e:e + 1] * expert_out
        moe_out_flat += shared_out

        moe_out_bsd = moe_out_flat.reshape(B, S, d)
        h = post_c[i]([moe_out_bsd, x_res, post2, comb2])[0]
        print(f"  layer {i}: {len(active)}/{E} experts active  h_out abs_mean={np.abs(h).mean():.4e}")

    split_logits = final_c([h])[0]
    split_logits_t = torch.from_numpy(split_logits)

    print("\n=== Compare split-IR output vs monolithic PyTorch ===")
    diff = (ref_logits.float() - split_logits_t.float()).abs()
    rel = diff / (ref_logits.float().abs() + 1e-6)
    print(f"  abs diff  max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")
    print(f"  rel diff  max={rel.max().item():.4e}  mean={rel.mean().item():.4e}")
    pt_top = ref_logits[0, -1].argmax().item()
    sp_top = split_logits_t[0, -1].argmax().item()
    print(f"  greedy next-token: PT={pt_top}  split-OV={sp_top}  match={pt_top == sp_top}")
    assert pt_top == sp_top, "split-OV greedy mismatch vs monolithic PT"

    print("\nSPLIT IR EQUIVALENCE: PASSED")


if __name__ == "__main__":
    main()
