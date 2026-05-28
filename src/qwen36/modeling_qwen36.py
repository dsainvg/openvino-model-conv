"""OV-traceable port of Qwen3.6-35B-A3B (qwen3_5_moe).

Decode-only (seq_len=1) forward with all state expressed as explicit input
and output tensors. Multimodal vision tower and MTP head are NOT ported here
-- text generation only.

Reference: venv-qwen/Lib/site-packages/transformers/models/qwen3_5_moe/
modeling_qwen3_5_moe.py. The math in this file mirrors the pure-torch
fallbacks (torch_causal_conv1d_update L218, torch_recurrent_gated_delta_rule
L323) but with stateful in-place updates replaced by functional IO.

State layout per layer (decode):
  full_attention layer i:
      k_cache_i:  (batch, num_kv_heads, max_seq, head_dim)
      v_cache_i:  (batch, num_kv_heads, max_seq, head_dim)
      attn_pos_i: int scalar -- current write position
  linear_attention layer i:
      conv_state_i:      (batch, conv_dim, conv_kernel_size)
      recurrent_state_i: (batch, num_v_heads, head_k_dim, head_v_dim)

Phase 1 uses a "compute-all + mask" MoE that is OV-traceable but O(num_experts)
in cost. Phase 2 will replace it with a Python-orchestrated selective compute
where experts live outside the main IR.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_qwen36 import Qwen36Config


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class QwenRMSNorm(nn.Module):
    """Qwen3.6 RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = norm * (1.0 + self.weight.float())
        return out.to(in_dtype)


class QwenRMSNormGated(nn.Module):
    """RMSNorm followed by silu-gating from a second tensor.

    out = (x * rsqrt(mean(x^2) + eps)) * weight * silu(gate)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = norm * self.weight.float() * F.silu(gate.float())
        return out.to(in_dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Rotary position embedding (standard, partial). mRoPE collapses to this when
# position IDs are identical across the three sections (text-only decode).
# ---------------------------------------------------------------------------


def build_rope_cache(
    rotary_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) of shape (max_position, rotary_dim) for partial RoPE."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = positions[:, None] * inv_freq[None, :]  # (P, rotary_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (P, rotary_dim)
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
    """q, k: (..., heads, head_dim).
    cos, sin: (..., rotary_dim).
    Applies rotation to the first rotary_dim channels; leaves the rest alone."""
    rotary_dim = cos.shape[-1]
    cos_b = cos.unsqueeze(-2)  # broadcast over heads dim
    sin_b = sin.unsqueeze(-2)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_out = q_rot * cos_b + _rotate_half(q_rot) * sin_b
    k_out = k_rot * cos_b + _rotate_half(k_rot) * sin_b
    return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)


# ---------------------------------------------------------------------------
# MLP / Expert / MoE
# ---------------------------------------------------------------------------


class QwenMLP(nn.Module):
    """SwiGLU MLP -- shared expert and any non-routed MLP use this."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenExpertFFN(nn.Module):
    """One routed expert. Same SwiGLU form as QwenMLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenMoEBlock(nn.Module):
    """Router + N routed experts + 1 gated shared expert.

    Phase 1 'compute-all + mask': every routed expert runs on every token,
    then the output is weighted by topk_weights * (topk_idx == e). Trace
    friendly but O(num_experts) per token. Acceptable for a small toy
    config; Phase 2 will replace this with per-expert subgraphs.
    """

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            QwenExpertFFN(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.num_experts)
        )
        self.shared_expert = QwenMLP(config.hidden_size, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden). Returns same shape."""
        batch, seq, hidden = x.shape
        flat = x.reshape(batch * seq, hidden)

        router_logits = self.gate(flat)  # (BS, E)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_idx = router_probs.topk(self.top_k, dim=-1)  # both (BS, K)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(flat.dtype)

        out = torch.zeros_like(flat)
        for e_idx, expert in enumerate(self.experts):
            e_out = expert(flat)  # (BS, hidden)
            mask = (topk_idx == e_idx).to(flat.dtype)  # (BS, K)
            w = (mask * topk_weights).sum(dim=-1, keepdim=True)  # (BS, 1)
            out = out + e_out * w

        shared = self.shared_expert(flat) * torch.sigmoid(self.shared_expert_gate(flat))
        return (out + shared).reshape(batch, seq, hidden)


# ---------------------------------------------------------------------------
# Full attention (decode, single token, explicit KV cache IO)
# ---------------------------------------------------------------------------


class QwenAttention(nn.Module):
    """Standard GQA attention with partial RoPE and an output-side sigmoid gate.

    Decode path only: input is one token, KV cache grows by one row each call.
    """

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.partial_rotary_dim = config.partial_rotary_dim
        self.scaling = self.head_dim ** -0.5

        # q_proj outputs 2 * num_heads * head_dim (the second half is the gate)
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, 1, hidden)
        cos: torch.Tensor,  # (B, 1, rotary_dim)
        sin: torch.Tensor,  # (B, 1, rotary_dim)
        k_cache_in: torch.Tensor,  # (B, num_kv_heads, max_seq, head_dim)
        v_cache_in: torch.Tensor,  # (B, num_kv_heads, max_seq, head_dim)
        write_pos: torch.Tensor,  # int scalar tensor: index to write at (0..max_seq-1)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.shape

        # Q projection produces (..., 2 * H * head_dim) which splits into query + gate.
        q_full = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim * 2)
        query, gate = q_full.chunk(2, dim=-1)
        # Q/K RMSNorm before RoPE.
        query = self.q_norm(query)
        k = self.k_norm(self.k_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim)
        # Partial RoPE.
        query, k = _apply_partial_rope(query, k, cos, sin)

        # Rearrange to (B, H, S, D).
        query = query.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Write new K/V into the cache at write_pos. Functional update (no in-place):
        # use a one-hot mask along the seq dim and combine via where().
        max_seq = k_cache_in.shape[2]
        positions = torch.arange(max_seq, device=k_cache_in.device)
        write_mask = (positions == write_pos.to(positions.dtype)).view(1, 1, max_seq, 1)
        # k has seq=1 at dim 2; broadcast to max_seq via where().
        k_cache_out = torch.where(write_mask, k.expand_as(k_cache_in), k_cache_in)
        v_cache_out = torch.where(write_mask, v.expand_as(v_cache_in), v_cache_in)

        # GQA: expand KV heads to match Q heads via repeat_interleave.
        if self.num_kv_groups > 1:
            k_full = k_cache_out.repeat_interleave(self.num_kv_groups, dim=1)
            v_full = v_cache_out.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_full = k_cache_out
            v_full = v_cache_out

        # Causal mask: only positions <= write_pos contribute.
        valid_mask = positions <= write_pos.to(positions.dtype)  # (max_seq,)
        attn_bias = torch.where(
            valid_mask, torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full((), float("-inf"), dtype=query.dtype, device=query.device),
        ).view(1, 1, 1, max_seq)

        # Scaled dot product. query: (B, H, 1, D); k_full: (B, H, S, D)
        scores = torch.matmul(query, k_full.transpose(-1, -2)) * self.scaling  # (B, H, 1, S)
        scores = scores + attn_bias
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        attn = torch.matmul(weights, v_full)  # (B, H, 1, D)

        # (B, H, 1, D) -> (B, 1, H*D) and apply output gate.
        attn = attn.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        attn = attn * torch.sigmoid(gate.reshape(batch, seq, self.num_heads * self.head_dim))
        out = self.o_proj(attn)
        return out, k_cache_out, v_cache_out


# ---------------------------------------------------------------------------
# Gated DeltaNet (linear attention) -- decode step (seq_len=1) only
# ---------------------------------------------------------------------------


class QwenGatedDeltaNet(nn.Module):
    """Mamba/DeltaNet hybrid linear attention. Single-token decode step.

    State:
        conv_state:      (B, conv_dim, conv_kernel_size) circular buffer
        recurrent_state: (B, num_v_heads, head_k_dim, head_v_dim)
    """

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = config.linear_key_dim
        self.value_dim = config.linear_value_dim
        self.conv_dim = config.linear_conv_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.num_v_per_k = config.num_v_per_k

        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

        # Depthwise causal conv1d (no bias in upstream; padding handled via state).
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm = QwenRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, 1, hidden)
        conv_state_in: torch.Tensor,  # (B, conv_dim, K)
        recurrent_state_in: torch.Tensor,  # (B, num_v_heads, head_k_dim, head_v_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).squeeze(1)  # (B, conv_dim)
        z = self.in_proj_z(hidden_states).reshape(batch, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states).squeeze(1)  # (B, num_v_heads)
        a = self.in_proj_a(hidden_states).squeeze(1)  # (B, num_v_heads)

        # --- Causal conv1d step (mirrors torch_causal_conv1d_update at seq_len=1) ---
        # Shift state left by 1 and append the new column.
        conv_state_out = torch.cat([conv_state_in[:, :, 1:], mixed_qkv.unsqueeze(-1)], dim=-1)
        # Depthwise weighted sum: weight (conv_dim, 1, K) -> (conv_dim, K)
        w = self.conv1d.weight.squeeze(1)
        conv_out = (conv_state_out * w.unsqueeze(0)).sum(dim=-1)  # (B, conv_dim)
        conv_out = F.silu(conv_out)

        # Split into q, k, v along conv_dim.
        q, k, v = torch.split(conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(batch, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch, self.num_v_heads, self.head_v_dim)

        # Expand q, k to match v's head count (GQA-style replication).
        if self.num_v_per_k > 1:
            q = q.repeat_interleave(self.num_v_per_k, dim=1)
            k = k.repeat_interleave(self.num_v_per_k, dim=1)

        # qk_l2norm
        q = _l2norm(q)
        k = _l2norm(k)
        # Scale q by 1/sqrt(head_k_dim) per torch reference.
        q = q * (self.head_k_dim ** -0.5)

        # beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
        beta = torch.sigmoid(b)  # (B, num_v_heads)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        # Cast to f32 for the recurrent step.
        q32, k32, v32 = q.float(), k.float(), v.float()
        beta32 = beta.float()
        state = recurrent_state_in.float()

        # decay: state = state * exp(g), per head
        decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        state = state * decay
        # kv_mem = sum over head_k of state * k[..., None]  -> (B, H, head_v)
        kv_mem = (state * k32.unsqueeze(-1)).sum(dim=-2)
        # delta = (v - kv_mem) * beta
        delta = (v32 - kv_mem) * beta32.unsqueeze(-1)
        # state update: state += k[..., None] * delta[..., None, :]
        state = state + k32.unsqueeze(-1) * delta.unsqueeze(-2)
        # out = sum over head_k of state * q[..., None]  -> (B, H, head_v)
        out = (state * q32.unsqueeze(-1)).sum(dim=-2)
        recurrent_state_out = state.to(recurrent_state_in.dtype)

        # Per-head RMSNormGated then merge heads.
        out = out.to(hidden_states.dtype)
        # Reshape so the norm applies per (B*H, head_v_dim).
        out_flat = out.reshape(batch * self.num_v_heads, self.head_v_dim)
        z_flat = z.reshape(batch * self.num_v_heads, self.head_v_dim)
        out_flat = self.norm(out_flat, z_flat)
        out = out_flat.reshape(batch, 1, self.num_v_heads * self.head_v_dim)

        out = self.out_proj(out)
        return out, conv_state_out, recurrent_state_out


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: Qwen36Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "full_attention":
            self.attn: nn.Module = QwenAttention(config)
        elif self.layer_type == "linear_attention":
            self.attn = QwenGatedDeltaNet(config)
        else:
            raise ValueError(f"Unknown layer_type {self.layer_type}")
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = QwenMoEBlock(config)

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
    """Embed + N decoder layers + final norm. Per-layer state is passed as
    *args so each tensor is a distinct OV graph input."""

    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(QwenDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Precompute RoPE cache as a buffer so it's part of the traced graph.
        cos, sin = build_rope_cache(
            rotary_dim=config.partial_rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Layer-index bookkeeping
        self._full_layer_indices = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]
        self._linear_layer_indices = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]

    @property
    def num_full_layers(self) -> int:
        return len(self._full_layer_indices)

    @property
    def num_linear_layers(self) -> int:
        return len(self._linear_layer_indices)

    def empty_state(self, batch: int, max_seq: int, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        """Build a fresh zeroed state dict for one batch.

        Returns dict with keys:
          k_caches:   list[Tensor]  (num_full_layers,) of (B, num_kv_heads, max_seq, head_dim)
          v_caches:   same shape
          conv_states: list[Tensor] (num_linear_layers,) of (B, conv_dim, K)
          rec_states:  list[Tensor] (num_linear_layers,) of (B, num_v_heads, head_k_dim, head_v_dim)
        """
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
        input_ids: torch.Tensor,  # (B, 1)
        position_ids: torch.Tensor,  # (B, 1) int
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
        conv_states: list[torch.Tensor],
        rec_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        x = self.embed_tokens(input_ids)  # (B, 1, hidden)

        # Gather RoPE cos/sin for this single position.
        batch, seq = position_ids.shape
        cos = self.rope_cos[position_ids.long()]  # (B, 1, rotary_dim)
        sin = self.rope_sin[position_ids.long()]
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        write_pos = position_ids[0, 0]  # scalar; all rows share position for batched decode

        k_out_list = list(k_caches)
        v_out_list = list(v_caches)
        conv_out_list = list(conv_states)
        rec_out_list = list(rec_states)

        full_i = 0
        lin_i = 0
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "full_attention":
                x, k_new, v_new = layer.forward_full(
                    x, cos, sin, k_out_list[full_i], v_out_list[full_i], write_pos
                )
                k_out_list[full_i] = k_new
                v_out_list[full_i] = v_new
                full_i += 1
            else:
                x, conv_new, rec_new = layer.forward_linear(
                    x, conv_out_list[lin_i], rec_out_list[lin_i]
                )
                conv_out_list[lin_i] = conv_new
                rec_out_list[lin_i] = rec_new
                lin_i += 1

        x = self.norm(x)
        return x, k_out_list, v_out_list, conv_out_list, rec_out_list


class QwenForCausalLM(nn.Module):
    def __init__(self, config: Qwen36Config):
        super().__init__()
        self.config = config
        self.model = QwenTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def empty_state(self, batch: int, max_seq: int, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        return self.model.empty_state(batch, max_seq, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
        conv_states: list[torch.Tensor],
        rec_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        x, k_out, v_out, conv_out, rec_out = self.model(
            input_ids, position_ids, k_caches, v_caches, conv_states, rec_states
        )
        logits = self.lm_head(x)
        return logits, k_out, v_out, conv_out, rec_out
