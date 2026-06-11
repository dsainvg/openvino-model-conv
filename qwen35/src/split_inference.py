"""Explicit split inference wrappers for Qwen3.5-4B.

Uses explicit signatures instead of *args to ensure compatibility with Dynamo ONNX exporter.
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
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(x))
