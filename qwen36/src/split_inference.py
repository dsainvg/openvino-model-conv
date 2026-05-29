"""Phase 2 split inference engine for Qwen3.6.

The monolithic QwenForCausalLM in modeling_qwen36.py traces fine for OV but
the MoE block has 256 experts -- a single IR with all experts unrolled would
be enormous. The split approach keeps the backbone (attention + router +
shared expert + everything else) in one IR and exports each routed expert as
a separate small IR. At inference, Python orchestrates: backbone -> top-k
selection -> dispatch only the selected experts -> combine.

This file provides:
  - A 'split-mode' variant of the MoE forward that returns the inputs needed
    to dispatch experts externally, without running them inside the IR.
  - Helpers to extract per-expert weight dicts from a full QwenForCausalLM.
  - A Python orchestrator that drives backbone + per-expert pieces.

Phase 1 compute-all stays intact as a numerical reference.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_qwen36 import Qwen36Config
from .modeling_qwen36 import (
    QwenDecoderLayer,
    QwenExpertFFN,
    QwenForCausalLM,
    QwenMLP,
    QwenMoEBlock,
)


# ---------------------------------------------------------------------------
# Split-mode MoE forward: router-only + combine helper
# ---------------------------------------------------------------------------


def moe_router_step(
    block: QwenMoEBlock,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run only the router + shared expert -- DO NOT touch the 256 experts.

    Returns:
        flat: (B*S, hidden) -- the input the experts should receive.
        topk_idx: (B*S, K) int64 -- which experts to dispatch.
        topk_weights: (B*S, K) -- normalized weights, casted to flat.dtype.
        shared_plus_gate: (B*S, hidden) -- shared expert contribution
                          (sigmoid-gated), ready to be added to expert sum.
    """
    batch, seq, hidden = x.shape
    flat = x.reshape(batch * seq, hidden)

    router_logits = block.gate(flat)
    probs = F.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_idx = probs.topk(block.top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(flat.dtype)

    shared = block.shared_expert(flat) * torch.sigmoid(block.shared_expert_gate(flat))
    return flat, topk_idx, topk_weights, shared


def moe_combine_step(
    expert_outputs: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_plus_gate: torch.Tensor,
    orig_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Weighted-sum the K dispatched expert outputs, add the shared output,
    reshape back to (B, S, hidden).

    expert_outputs: (B*S, K, hidden) -- one row per dispatched expert in order.
    topk_weights:   (B*S, K)
    shared_plus_gate: (B*S, hidden)
    """
    weighted = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)  # (B*S, hidden)
    out = weighted + shared_plus_gate
    return out.reshape(*orig_shape)


# ---------------------------------------------------------------------------
# Per-layer backbone wrappers (for OV conversion)
# ---------------------------------------------------------------------------


class QwenLayerBackboneFull(nn.Module):
    """Wraps a full-attention QwenDecoderLayer to expose pre-MoE pieces as
    distinct outputs. The expert dispatch is performed externally by the
    orchestrator; this IR does NOT call any expert.
    """

    def __init__(self, layer: QwenDecoderLayer):
        super().__init__()
        if layer.layer_type != "full_attention":
            raise ValueError(f"expected full_attention layer, got {layer.layer_type}")
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: torch.Tensor,
    ):
        h = self.layer.input_layernorm(x)
        attn_out, k_new, v_new = self.layer.attn(h, cos, sin, k_cache, v_cache, write_pos)
        x_post_attn = x + attn_out
        h2 = self.layer.post_attention_layernorm(x_post_attn)
        flat, topk_idx, topk_w, shared = moe_router_step(self.layer.mlp, h2)
        return x_post_attn, flat, topk_idx, topk_w, shared, k_new, v_new


class QwenLayerBackboneLinear(nn.Module):
    """Same for linear-attention layers."""

    def __init__(self, layer: QwenDecoderLayer):
        super().__init__()
        if layer.layer_type != "linear_attention":
            raise ValueError(f"expected linear_attention layer, got {layer.layer_type}")
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ):
        h = self.layer.input_layernorm(x)
        attn_out, conv_new, rec_new = self.layer.attn(h, conv_state, rec_state)
        x_post_attn = x + attn_out
        h2 = self.layer.post_attention_layernorm(x_post_attn)
        flat, topk_idx, topk_w, shared = moe_router_step(self.layer.mlp, h2)
        return x_post_attn, flat, topk_idx, topk_w, shared, conv_new, rec_new


def combine_layer_output(
    x_post_attn: torch.Tensor,
    expert_outputs: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_plus_gate: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the layer's output tensor from the backbone pieces and the
    externally-computed expert outputs. expert_outputs has shape (B*S, K, hidden).
    """
    weighted = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)
    moe_out = (weighted + shared_plus_gate).reshape(x_post_attn.shape)
    return x_post_attn + moe_out


# ---------------------------------------------------------------------------
# Per-expert OV IR helpers
# ---------------------------------------------------------------------------


def extract_expert_state_dict(model: QwenForCausalLM, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
    """Pull one expert's weights out of the full model into a small dict
    suitable for loading into a fresh QwenExpertFFN."""
    layer = model.model.layers[layer_idx]
    expert: QwenExpertFFN = layer.mlp.experts[expert_idx]
    return {k: v.detach().clone() for k, v in expert.state_dict().items()}


def build_standalone_expert(
    config: Qwen36Config, weights: dict[str, torch.Tensor] | None = None
) -> QwenExpertFFN:
    """Create a QwenExpertFFN sized to the routed-expert dimensions, optionally
    loaded with the given weight dict."""
    expert = QwenExpertFFN(config.hidden_size, config.moe_intermediate_size)
    if weights is not None:
        expert.load_state_dict(weights)
    expert.eval()
    return expert


# ---------------------------------------------------------------------------
# Python orchestrator: monolithic-equivalent inference via split components
# ---------------------------------------------------------------------------


def monolithic_step_via_split(
    model: QwenForCausalLM,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    state: dict,
) -> tuple[torch.Tensor, dict]:
    """Drive one decode step using the SPLIT path (router-only + per-expert
    forwards orchestrated in Python), and return the same outputs the
    monolithic forward would produce.

    This is the pure-torch reference that the OV split orchestrator must
    match. It exercises the moe_router_step + per-expert dispatch +
    moe_combine_step seam end-to-end, so any drift from monolithic shows up
    here before we touch OV.
    """
    mdl = model.model
    cfg = model.config
    x = mdl.embed_tokens(input_ids)

    batch, seq = position_ids.shape
    cos = mdl.rope_cos[position_ids.long()].to(x.dtype)
    sin = mdl.rope_sin[position_ids.long()].to(x.dtype)
    write_pos = position_ids[0, 0]

    k_caches = list(state["k_caches"])
    v_caches = list(state["v_caches"])
    conv_states = list(state["conv_states"])
    rec_states = list(state["rec_states"])

    full_i = 0
    lin_i = 0
    for layer in mdl.layers:
        if layer.layer_type == "full_attention":
            h = layer.input_layernorm(x)
            attn_out, k_new, v_new = layer.attn(h, cos, sin, k_caches[full_i], v_caches[full_i], write_pos)
            k_caches[full_i] = k_new
            v_caches[full_i] = v_new
            full_i += 1
        else:
            h = layer.input_layernorm(x)
            attn_out, conv_new, rec_new = layer.attn(h, conv_states[lin_i], rec_states[lin_i])
            conv_states[lin_i] = conv_new
            rec_states[lin_i] = rec_new
            lin_i += 1
        x = x + attn_out

        # MoE split path
        h2 = layer.post_attention_layernorm(x)
        flat, topk_idx, topk_w, shared = moe_router_step(layer.mlp, h2)
        # Dispatch each unique selected expert
        K = topk_idx.shape[-1]
        BS = flat.shape[0]
        expert_outs = torch.zeros(BS, K, flat.shape[-1], dtype=flat.dtype)
        for t in range(BS):
            for k in range(K):
                e_idx = topk_idx[t, k].item()
                expert_outs[t, k] = layer.mlp.experts[e_idx](flat[t : t + 1]).squeeze(0)
        moe_out = moe_combine_step(expert_outs, topk_w, shared, h2.shape)
        x = x + moe_out

    x = mdl.norm(x)
    logits = model.lm_head(x)
    new_state = {
        "k_caches": k_caches,
        "v_caches": v_caches,
        "conv_states": conv_states,
        "rec_states": rec_states,
    }
    return logits, new_state
