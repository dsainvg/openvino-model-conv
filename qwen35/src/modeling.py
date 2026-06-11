"""OV-traceable PyTorch port of Qwen3.5-4B (model_type=qwen3_5).

Architecture: hybrid decoder with alternating Gated DeltaNet (linear
attention) and standard GQA full-attention layers, every
full_attention_interval-th layer is full attention. Dense SwiGLU FFN in
every layer (no MoE in the 4B variant).

Decode-only (seq_len=1). All state is expressed as explicit input/output
tensors so every tensor is a named OV graph port.

Reference: transformers/models/qwen3_5/modeling_qwen3_5.py
(pure-torch fallback paths). Stateful in-place ops replaced by functional IO.

State layout per layer (decode):
  full_attention layer i:
      k_cache_i: (batch, num_kv_heads, max_seq, head_dim)
      v_cache_i: (batch, num_kv_heads, max_seq, head_dim)
  linear_attention layer i:
      conv_state_i:      (batch, conv_dim, conv_kernel_size)
      recurrent_state_i: (batch, num_v_heads, head_k_dim, head_v_dim)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import Qwen35Config


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class QwenRMSNorm(nn.Module):
    """RMSNorm: out = x * rsqrt(mean(x²) + eps) * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * (1.0 + self.weight.float())).to(in_dtype)


class QwenRMSNormGated(nn.Module):
    """RMSNorm + silu gate: out = rms_norm(x) * weight * silu(gate)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float() * F.silu(gate.float())).to(in_dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Rotary position embedding (partial)
# ---------------------------------------------------------------------------


def build_rope_cache(
    rotary_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    cos_b = cos.unsqueeze(-2)
    sin_b = sin.unsqueeze(-2)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_out = q_rot * cos_b + _rotate_half(q_rot) * sin_b
    k_out = k_rot * cos_b + _rotate_half(k_rot) * sin_b
    return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)


# ---------------------------------------------------------------------------
# Dense SwiGLU MLP
# ---------------------------------------------------------------------------


class QwenMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Full-attention layer (GQA, partial RoPE, output gate)
# ---------------------------------------------------------------------------


class QwenAttention(nn.Module):
    """GQA full-attention with partial RoPE and sigmoid output gate.
    Decode path only (seq_len=1). KV cache grows by one row each call.
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.num_heads    = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim     = config.head_dim
        self.rotary_dim   = config.partial_rotary_dim
        self.scaling      = self.head_dim ** -0.5

        # q_proj output = 2 * H * D  (second half is sigmoid gate)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim,  bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim,  bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,     bias=config.attention_bias)
        self.q_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, 1, hidden)
        cos: torch.Tensor,             # (B, 1, rotary_dim)
        sin: torch.Tensor,
        k_cache_in: torch.Tensor,      # (B, kv_heads, max_seq, head_dim)
        v_cache_in: torch.Tensor,
        write_pos: torch.Tensor,       # int scalar
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.shape

        q_full = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim * 2)
        query, gate = q_full.chunk(2, dim=-1)
        query = self.q_norm(query)
        k = self.k_norm(self.k_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim)
        query, k = _apply_partial_rope(query, k, cos, sin)

        query = query.transpose(1, 2)
        k     = k.transpose(1, 2)
        v     = v.transpose(1, 2)

        max_seq = k_cache_in.shape[2]
        positions  = torch.arange(max_seq, device=k_cache_in.device)
        write_mask = (positions == write_pos.to(positions.dtype)).view(1, 1, max_seq, 1)
        k_cache_out = torch.where(write_mask, k.expand_as(k_cache_in), k_cache_in)
        v_cache_out = torch.where(write_mask, v.expand_as(v_cache_in), v_cache_in)

        if self.num_kv_groups > 1:
            k_full = k_cache_out.repeat_interleave(self.num_kv_groups, dim=1)
            v_full = v_cache_out.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_full = k_cache_out
            v_full = v_cache_out

        valid_mask = positions <= write_pos.to(positions.dtype)
        attn_bias = torch.where(
            valid_mask,
            torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full((), float("-inf"), dtype=query.dtype, device=query.device),
        ).view(1, 1, 1, max_seq)

        scores  = torch.matmul(query, k_full.transpose(-1, -2)) * self.scaling + attn_bias
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        attn    = torch.matmul(weights, v_full)

        attn = attn.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        attn = attn * torch.sigmoid(gate.reshape(batch, seq, self.num_heads * self.head_dim))
        return self.o_proj(attn), k_cache_out, v_cache_out


# ---------------------------------------------------------------------------
# Linear-attention layer (Gated DeltaNet, single-token decode)
# ---------------------------------------------------------------------------


class QwenGatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention, single-token decode step.

    State:
        conv_state:      (B, conv_dim, conv_kernel_size)  circular buffer
        recurrent_state: (B, num_v_heads, head_k_dim, head_v_dim)
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.num_v_heads      = config.linear_num_value_heads
        self.num_k_heads      = config.linear_num_key_heads
        self.head_k_dim       = config.linear_key_head_dim
        self.head_v_dim       = config.linear_value_head_dim
        self.key_dim          = config.linear_key_dim
        self.value_dim        = config.linear_value_dim
        self.conv_dim         = config.linear_conv_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.num_v_per_k      = config.num_v_per_k

        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim,   bias=False)
        self.in_proj_z   = nn.Linear(config.hidden_size, self.value_dim,  bias=False)
        self.in_proj_b   = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a   = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log   = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm     = QwenRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,    # (B, 1, hidden)
        conv_state_in: torch.Tensor,    # (B, conv_dim, K)
        recurrent_state_in: torch.Tensor,  # (B, num_v_heads, head_k_dim, head_v_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).squeeze(1)  # (B, conv_dim)
        z   = self.in_proj_z(hidden_states).reshape(batch, self.num_v_heads, self.head_v_dim)
        b   = self.in_proj_b(hidden_states).squeeze(1)
        a   = self.in_proj_a(hidden_states).squeeze(1)

        # Causal conv1d step: shift state left, append new token projection
        conv_state_out = torch.cat([conv_state_in[:, :, 1:], mixed_qkv.unsqueeze(-1)], dim=-1)
        w = self.conv1d.weight.squeeze(1)
        conv_out = F.silu((conv_state_out * w.unsqueeze(0)).sum(dim=-1))

        q, k, v = torch.split(conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(batch, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch, self.num_v_heads, self.head_v_dim)

        if self.num_v_per_k > 1:
            q = q.repeat_interleave(self.num_v_per_k, dim=1)
            k = k.repeat_interleave(self.num_v_per_k, dim=1)

        q = _l2norm(q) * (self.head_k_dim ** -0.5)
        k = _l2norm(k)

        beta  = torch.sigmoid(b)
        g     = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        q32, k32, v32 = q.float(), k.float(), v.float()
        state = recurrent_state_in.float()

        decay = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = state * decay
        kv_mem = (state * k32.unsqueeze(-1)).sum(dim=-2)
        delta  = (v32 - kv_mem) * beta.float().unsqueeze(-1)
        state  = state + k32.unsqueeze(-1) * delta.unsqueeze(-2)
        out    = (state * q32.unsqueeze(-1)).sum(dim=-2)
        recurrent_state_out = state.to(recurrent_state_in.dtype)

        out     = out.to(hidden_states.dtype)
        out_flat = out.reshape(batch * self.num_v_heads, self.head_v_dim)
        z_flat   = z.reshape(batch * self.num_v_heads, self.head_v_dim)
        out_flat = self.norm(out_flat, z_flat)
        out      = out_flat.reshape(batch, 1, self.num_v_heads * self.head_v_dim)
        return self.out_proj(out), conv_state_out, recurrent_state_out


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx  = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "full_attention":
            self.attn: nn.Module = QwenAttention(config)
        elif self.layer_type == "linear_attention":
            self.attn = QwenGatedDeltaNet(config)
        else:
            raise ValueError(f"Unknown layer_type {self.layer_type!r}")

        self.input_layernorm          = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = QwenMLP(config.hidden_size, config.intermediate_size)

    def forward_full(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_layernorm(x)
        attn_out, k_out, v_out = self.attn(h, cos, sin, k_cache, v_cache, write_pos)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k_out, v_out

    def forward_linear(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_layernorm(x)
        attn_out, conv_out, rec_out = self.attn(h, conv_state, recurrent_state)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, conv_out, rec_out


# ---------------------------------------------------------------------------
# Full text model + LM head
# ---------------------------------------------------------------------------


class QwenTextModel(nn.Module):
    """Embed + N decoder layers + final norm.
    All KV / recurrent state is passed as explicit tensors (no hidden state).
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config       = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers       = nn.ModuleList(QwenDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm         = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        cos, sin = build_rope_cache(
            rotary_dim=config.partial_rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._full_layer_indices   = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]
        self._linear_layer_indices = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]

    @property
    def num_full_layers(self) -> int:
        return len(self._full_layer_indices)

    @property
    def num_linear_layers(self) -> int:
        return len(self._linear_layer_indices)

    def empty_state(
        self,
        batch: int,
        max_seq: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> dict:
        c = self.config
        return {
            "k_caches": [
                torch.zeros(batch, c.num_key_value_heads, max_seq, c.head_dim, dtype=dtype, device=device)
                for _ in range(self.num_full_layers)
            ],
            "v_caches": [
                torch.zeros(batch, c.num_key_value_heads, max_seq, c.head_dim, dtype=dtype, device=device)
                for _ in range(self.num_full_layers)
            ],
            "conv_states": [
                torch.zeros(batch, c.linear_conv_dim, c.linear_conv_kernel_dim, dtype=dtype, device=device)
                for _ in range(self.num_linear_layers)
            ],
            "rec_states": [
                torch.zeros(
                    batch, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                    dtype=dtype, device=device,
                )
                for _ in range(self.num_linear_layers)
            ],
        }

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, 1)
        position_ids: torch.Tensor,     # (B, 1) int
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
        conv_states: list[torch.Tensor],
        rec_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list, list, list, list]:
        x = self.embed_tokens(input_ids)

        cos = self.rope_cos[position_ids.long()].to(x.dtype)
        sin = self.rope_sin[position_ids.long()].to(x.dtype)
        write_pos = position_ids[0, 0]

        k_out   = list(k_caches)
        v_out   = list(v_caches)
        conv_out = list(conv_states)
        rec_out  = list(rec_states)

        full_i = lin_i = 0
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                x, k_new, v_new = layer.forward_full(x, cos, sin, k_out[full_i], v_out[full_i], write_pos)
                k_out[full_i] = k_new
                v_out[full_i] = v_new
                full_i += 1
            else:
                x, conv_new, rec_new = layer.forward_linear(x, conv_out[lin_i], rec_out[lin_i])
                conv_out[lin_i] = conv_new
                rec_out[lin_i]  = rec_new
                lin_i += 1

        return self.norm(x), k_out, v_out, conv_out, rec_out


class QwenForCausalLM(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config  = config
        self.model   = QwenTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def empty_state(self, batch: int, max_seq: int, dtype=torch.float32, device="cpu") -> dict:
        return self.model.empty_state(batch, max_seq, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
        conv_states: list[torch.Tensor],
        rec_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list, list, list, list]:
        x, k_out, v_out, conv_out, rec_out = self.model(
            input_ids, position_ids, k_caches, v_caches, conv_states, rec_states
        )
        return self.lm_head(x), k_out, v_out, conv_out, rec_out
