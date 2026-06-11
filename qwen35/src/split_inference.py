"""OV-traceable PyTorch wrapper for OpenVINO GenAI / NPU deployment.

This wrapper exposes the complete Qwen3.5-4B forward pass (embed + 32 layers
+ final norm + lm_head) as a SINGLE nn.Module with a flat, explicit signature
that is compatible with openvino_genai.LLMPipeline and Intel NPU.

The key differences from the split-IR wrappers:

1. ALL state tensors are flattened into positional inputs and outputs so
   OpenVINO can bind them as ReadValue / Assign variables (stateful model).

2. `beam_idx` (shape [batch]) is accepted and used to reorder state tensors
   at the top of forward(). This is mandatory for beam search in GenAI.

3. `attention_mask` is accepted (shape [batch, seq_len]) — currently not used
   in single-token decode but required by the GenAI pipeline signature.

4. `position_ids` (shape [batch, 1]) drives the RoPE cache lookup.

State layout (flat, in order):
    For each full_attention layer i  (there are num_full_layers of them):
        k_cache_i  : (batch, num_kv_heads, max_seq, head_dim)
        v_cache_i  : (batch, num_kv_heads, max_seq, head_dim)
    For each linear_attention layer j  (there are num_linear_layers of them):
        conv_state_j      : (batch, conv_dim, conv_kernel_size)
        recurrent_state_j : (batch, num_v_heads, head_k_dim, head_v_dim)

Inputs (in order, all named for OV graph ports):
    input_ids      : [batch, seq]    int64
    attention_mask : [batch, seq]    int64   (unused internally, required by GenAI)
    position_ids   : [batch, seq]    int64
    beam_idx       : [batch]         int64
    k_cache_0      : [batch, kv_heads, max_seq, head_dim]   float16
    v_cache_0      : ...
    k_cache_3      : ...
    ...  (one pair per full_attention layer)
    conv_state_0   : [batch, conv_dim, kernel_size]   float16
    recurrent_state_0 : [batch, v_heads, k_dim, v_dim]   float16
    ...  (one pair per linear_attention layer)

Outputs (in order):
    logits         : [batch, seq, vocab_size]
    k_cache_0      : updated key cache   (same shape as input)
    v_cache_0      : ...
    ...
    conv_state_0   : ...
    recurrent_state_0 : ...
"""
from __future__ import annotations

import torch
from torch import nn

from .configuration import Qwen35Config
from .modeling import QwenDecoderLayer, QwenForCausalLM, QwenTextModel


class QwenGenAIWrapper(nn.Module):
    """Single-module wrapper compatible with openvino_genai.LLMPipeline.

    Flat explicit state I/O lets OpenVINO's make_stateful() bind each
    (state_in, state_out) pair into a ReadValue / Assign node so the state
    is managed inside the graph — no host orchestration needed at runtime.
    """

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.model   = model.model      # QwenTextModel
        self.lm_head = model.lm_head
        self.cfg     = model.config

        cfg = self.cfg
        self._full_indices   = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
        self._linear_indices = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
        self.num_full   = len(self._full_indices)
        self.num_linear = len(self._linear_indices)

    # ------------------------------------------------------------------
    # forward — fully explicit signature (no *args, no lists)
    # Total number of state args = 2*num_full + 2*num_linear
    # For Qwen3.5-4B (32 layers, 8 full, 24 linear) = 16 + 48 = 64 state args
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, S]  int64
        attention_mask: torch.Tensor,   # [B, S]  int64   (unused, required by GenAI)
        position_ids:   torch.Tensor,   # [B, S]  int64
        beam_idx:       torch.Tensor,   # [B]     int64
        *state_tensors: torch.Tensor,   # flat ordered state (see module docstring)
    ) -> tuple[torch.Tensor, ...]:      # logits + flat updated state

        cfg = self.cfg

        # ── 1. Reorder state for beam search ──────────────────────────
        # beam_idx tells us which beam each batch slot came from.
        # We index_select every state tensor along dim 0.
        state = [s.index_select(0, beam_idx) for s in state_tensors]

        # ── 2. Unpack flat state ───────────────────────────────────────
        nf, nl = self.num_full, self.num_linear
        k_caches        = list(state[0        : nf])
        v_caches        = list(state[nf       : 2*nf])
        conv_states     = list(state[2*nf     : 2*nf + nl])
        rec_states      = list(state[2*nf + nl: 2*nf + 2*nl])

        # ── 3. Embedding ───────────────────────────────────────────────
        x = self.model.embed_tokens(input_ids)

        # ── 4. RoPE  ───────────────────────────────────────────────────
        pos = position_ids.long()
        cos = self.model.rope_cos[pos].to(x.dtype)   # [B, S, rotary_dim]
        sin = self.model.rope_sin[pos].to(x.dtype)
        write_pos = position_ids[0, 0]

        # ── 5. Decoder layers ──────────────────────────────────────────
        full_i = lin_i = 0
        k_outs   = []
        v_outs   = []
        conv_outs = []
        rec_outs  = []

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
        x      = self.model.norm(x)
        logits = self.lm_head(x)

        # ── 7. Flatten outputs  ────────────────────────────────────────
        # Order must match input order so make_stateful can pair them:
        #   k_caches, v_caches, conv_states, rec_states
        return (logits, *k_outs, *v_outs, *conv_outs, *rec_outs)
