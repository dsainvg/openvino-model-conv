"""Inference wrappers for Qwen3.5-4B OpenVINO export.

Two export strategies are supported:

1. Split-IR (convert_to_openvino.py)
   One .xml per component, state managed externally by Python orchestrator.
   Wrappers: QwenEmbedWrapper, QwenLayerFullWrapper,
             QwenLayerLinearWrapper, QwenLMHeadWrapper

2. GenAI / NPU stateful (convert_to_openvino_genai.py)
   Single .xml, state wired as ReadValue/Assign nodes via MakeStateful.
   Wrapper: QwenGenAIWrapper
"""
from __future__ import annotations

import torch
from torch import nn

from .modeling import QwenDecoderLayer, QwenForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
# Split-IR wrappers  (used by convert_to_openvino.py)
# ─────────────────────────────────────────────────────────────────────────────

class QwenEmbedWrapper(nn.Module):
    """Wraps embedding layer: input_ids -> hidden_states."""

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.embed = model.model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class QwenLayerFullWrapper(nn.Module):
    """Wraps a full_attention layer with explicit arguments."""

    def __init__(self, layer: QwenDecoderLayer):
        super().__init__()
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layer.forward_full(x, cos, sin, k_cache, v_cache, write_pos)


class QwenLayerLinearWrapper(nn.Module):
    """Wraps a linear_attention layer with explicit arguments."""

    def __init__(self, layer: QwenDecoderLayer):
        super().__init__()
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layer.forward_linear(x, conv_state, recurrent_state)


class QwenLMHeadWrapper(nn.Module):
    """Wraps final norm and lm_head: hidden_states -> logits."""

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.norm    = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# GenAI / NPU stateful wrapper  (used by convert_to_openvino_genai.py)
# ─────────────────────────────────────────────────────────────────────────────

class QwenGenAIWrapper(nn.Module):
    """Single-module wrapper compatible with openvino_genai.LLMPipeline.

    Flat explicit state I/O lets OpenVINO's MakeStateful pass bind each
    (state_in, state_out) pair into ReadValue / Assign nodes so the state
    is managed inside the graph — no host orchestration needed at runtime.

    State layout (flat, in order):
        k_cache_0..N-1     — key caches for full-attention layers
        v_cache_0..N-1     — value caches for full-attention layers
        conv_state_0..M-1  — conv states for linear-attention layers
        rec_state_0..M-1   — recurrent states for linear-attention layers

    where N = num_full_layers, M = num_linear_layers.

    Public inputs  (after make_stateful removes state ports):
        input_ids, attention_mask, position_ids, beam_idx
    Public output:
        logits
    """

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.model   = model.model   # QwenTextModel
        self.lm_head = model.lm_head
        self.cfg     = model.config

        cfg = self.cfg
        self._full_indices   = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
        self._linear_indices = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
        self.num_full   = len(self._full_indices)
        self.num_linear = len(self._linear_indices)

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, S]  int64
        attention_mask: torch.Tensor,   # [B, S]  int64  (unused; required by GenAI)
        position_ids:   torch.Tensor,   # [B, S]  int64
        beam_idx:       torch.Tensor,   # [B]     int64
        *state_tensors: torch.Tensor,   # flat ordered state (see docstring)
    ) -> tuple[torch.Tensor, ...]:      # (logits, *updated_states)

        # ── 1. Reorder state for beam search ──────────────────────────
        state = [s.index_select(0, beam_idx) for s in state_tensors]

        # ── 2. Unpack flat state ───────────────────────────────────────
        nf, nl = self.num_full, self.num_linear
        k_caches    = list(state[0         : nf])
        v_caches    = list(state[nf        : 2 * nf])
        conv_states = list(state[2 * nf    : 2 * nf + nl])
        rec_states  = list(state[2 * nf + nl : 2 * nf + 2 * nl])

        # ── 3. Embed ───────────────────────────────────────────────────
        x = self.model.embed_tokens(input_ids)

        # ── 4. RoPE ────────────────────────────────────────────────────
        pos       = position_ids.long()
        cos       = self.model.rope_cos[pos].to(x.dtype)
        sin       = self.model.rope_sin[pos].to(x.dtype)
        write_pos = position_ids[0, 0]

        # ── 5. Decoder layers ──────────────────────────────────────────
        full_i = lin_i = 0
        k_outs = []; v_outs = []; conv_outs = []; rec_outs = []

        for layer in self.model.layers:
            if layer.layer_type == "full_attention":
                x, k_new, v_new = layer.forward_full(
                    x, cos, sin, k_caches[full_i], v_caches[full_i], write_pos
                )
                k_outs.append(k_new)
                v_outs.append(v_new)
                full_i += 1
            else:
                x, conv_new, rec_new = layer.forward_linear(
                    x, conv_states[lin_i], rec_states[lin_i]
                )
                conv_outs.append(conv_new)
                rec_outs.append(rec_new)
                lin_i += 1

        # ── 6. LM Head ────────────────────────────────────────────────
        logits = self.lm_head(self.model.norm(x))

        # ── 7. Return logits + flat updated states ─────────────────────
        return (logits, *k_outs, *v_outs, *conv_outs, *rec_outs)
