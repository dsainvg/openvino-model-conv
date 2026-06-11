"""Split inference wrappers and state definitions for Qwen3.5-4B.

Enables layer-by-layer OpenVINO conversion and low-memory execution.
"""
from __future__ import annotations

import torch
from torch import nn

from .modeling import QwenDecoderLayer, QwenForCausalLM


class QwenEmbedWrapper(nn.Module):
    """Wraps embedding layer: input_ids -> hidden_states."""

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.embed = model.model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class QwenLayerWrapper(nn.Module):
    """Wraps a single QwenDecoderLayer to support flat arguments mapping."""

    def __init__(self, layer: QwenDecoderLayer):
        super().__init__()
        self.layer = layer
        self.layer_type = layer.layer_type

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.layer_type == "full_attention":
            # args: x, cos, sin, k_cache, v_cache, write_pos
            x, cos, sin, k_cache, v_cache, write_pos = args
            x_out, k_out, v_out = self.layer.forward_full(
                x, cos, sin, k_cache, v_cache, write_pos
            )
            return x_out, k_out, v_out
        else:
            # args: x, conv_state, recurrent_state
            x, conv_state, recurrent_state = args
            x_out, conv_out, rec_out = self.layer.forward_linear(
                x, conv_state, recurrent_state
            )
            return x_out, conv_out, rec_out


class QwenLMHeadWrapper(nn.Module):
    """Wraps final norm and lm_head: hidden_states -> logits."""

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(x))
