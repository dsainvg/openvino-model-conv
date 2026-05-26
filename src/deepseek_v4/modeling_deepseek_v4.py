"""DeepSeek-V4 modeling, ported from deepseek-ai/DeepSeek-V4-Flash inference/model.py.

Differences from the reference:
- Pure PyTorch. No TileLang kernels, no FP4/FP8 dtypes (BF16/FP32 throughout).
- Single-rank: no torch.distributed; ParallelEmbedding/ColumnParallel/RowParallel are plain nn.Linear/nn.Embedding.
- KV cache is exposed via past_key_values: a list of per-layer Block-input tensors
  [B, S_past, hc_mult, dim]. On decode (past given), each Block concatenates past+new along
  the sequence dim internally so the K/V projection and compressor see the full history,
  but Q is computed only for new positions and the block returns outputs only for new positions.
  This is not perfectly O(1) per step (projection still scans the past) but it gives a faithful
  cache API and avoids re-running the full stack per generated token.
- Sparse attention is implemented via index-gather + dense softmax over the gathered slice.
  The "sparse" part is the indices selected by the indexer; the kernel itself is dense.
- Sinkhorn balancing is an in-graph for loop with `hc_sinkhorn_iters` iterations.
- No Hadamard rotation (it was for FP8 QAT prep in the reference; not needed here).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_deepseek_v4 import DeepseekV4Config


# ------------------------------------------------------------------------------
# RoPE / YaRN
# ------------------------------------------------------------------------------
def _yarn_find_correction_dim(num_rotations, dim, base, max_seq_len):
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(min_v, max_v, dim, dtype=torch.float32):
    if min_v == max_v:
        max_v = max_v + 0.001
    linear = (torch.arange(dim, dtype=dtype) - min_v) / (max_v - min_v)
    return torch.clamp(linear, 0, 1)


def precompute_rope_cos_sin(
    rope_dim: int,
    seq_len: int,
    base: float = 10000.0,
    yarn_factor: float = 1.0,
    yarn_original_max_pos: int = 0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    dtype: torch.dtype = torch.float32,
):
    """Returns (cos, sin) of shape [seq_len, rope_dim/2]. Pure real-valued — no complex tensors."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    if yarn_original_max_pos > 0 and yarn_factor != 1.0:
        low, high = _yarn_find_correction_range(beta_fast, beta_slow, rope_dim, base, yarn_original_max_pos)
        smooth = 1.0 - _yarn_linear_ramp(low, high, rope_dim // 2)
        inv_freq = inv_freq / yarn_factor * (1.0 - smooth) + inv_freq * smooth
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [S, rope_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rotary_emb_inplace_slice(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply real-valued RoPE to last `2*cos.size(-1)` dims of x.
    x shape: [B, L, *middle, 2*D] where the L dim is x.size(1).
    cos/sin shape: [L, D]. Returns rotated tensor (same shape as x)."""
    *prefix, two_d = x.shape
    L = cos.size(0)
    d = cos.size(-1)
    assert two_d == 2 * d, f"x last dim {two_d} != 2*cos last dim {2*d}"
    assert x.size(1) == L, f"x sequence dim {x.size(1)} != cos length {L}"
    # Pairs: [B, L, *middle, D, 2]
    x_pairs = x.float().reshape(*prefix, d, 2)
    x_real, x_imag = x_pairs[..., 0], x_pairs[..., 1]
    # Broadcast cos/sin from [L, D] to [1, L, *1's, D] matching x_real rank.
    n_middle = x_real.dim() - 2 - 1  # exclude leading [B, L] and trailing [D]
    bcast_shape = [1, L] + [1] * n_middle + [d]
    cos_b = cos.reshape(bcast_shape).to(x_real.dtype)
    sin_b = sin.reshape(bcast_shape).to(x_real.dtype)
    out_real = x_real * cos_b - x_imag * sin_b
    out_imag = x_real * sin_b + x_imag * cos_b
    out = torch.stack([out_real, out_imag], dim=-1).reshape(*prefix, two_d)
    return out.to(x.dtype)


# ------------------------------------------------------------------------------
# RMSNorm
# ------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Stored fp32 for numerical stability; weights cast to input dtype on output.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (self.weight * xf).to(dtype)


# ------------------------------------------------------------------------------
# Compressor (KV compression by learned gated pooling over `compress_ratio` tokens)
# ------------------------------------------------------------------------------
class Compressor(nn.Module):
    """Pooled KV compression. Prefill-only: no across-call state; we recompute over the full sequence.
    Outputs compressed KV at length seq_len // compress_ratio."""

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int):
        super().__init__()
        assert compress_ratio in (4, 128), f"unsupported compress_ratio {compress_ratio}"
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coff = 1 + self.overlap
        # Learned positional encoding for tokens within a compression window.
        self.ape = nn.Parameter(torch.zeros(compress_ratio, coff * head_dim, dtype=torch.float32))
        self.wkv = nn.Linear(self.dim, coff * head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(self.dim, coff * head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [B, S, dim], cos/sin: [S, rope_dim/2]. Returns compressed kv: [B, S/ratio, head_dim]."""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        d = self.head_dim
        rd = self.rope_dim
        cutoff = (seqlen // ratio) * ratio
        if cutoff == 0:
            return x.new_zeros(bsz, 0, d)

        x32 = x.float()
        kv = self.wkv(x32)[:, :cutoff]     # [B, cutoff, coff*d]
        score = self.wgate(x32)[:, :cutoff]
        kv = kv.unflatten(1, (-1, ratio))     # [B, n_chunks, ratio, coff*d]
        score = score.unflatten(1, (-1, ratio)) + self.ape  # broadcast ape over chunks

        if self.overlap:
            # Build [B, n_chunks, 2*ratio, d]:
            #   second half (ratio:) = current chunk's "second half" of coff*d
            #   first  half (:ratio) = previous chunk's "first half" of coff*d
            b, n, _, _ = kv.size()
            kv_full = kv.new_zeros(b, n, 2 * ratio, d)
            kv_full[:, :, ratio:] = kv[:, :, :, d:]
            kv_full[:, 1:, :ratio] = kv[:, :-1, :, :d]
            sc_full = score.new_full((b, n, 2 * ratio, d), float("-inf"))
            sc_full[:, :, ratio:] = score[:, :, :, d:]
            sc_full[:, 1:, :ratio] = score[:, :-1, :, :d]
            kv, score = kv_full, sc_full

        # Weighted pool over the (overlapping or normal) ratio axis.
        kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [B, n_chunks, d]
        kv = self.norm(kv.to(x.dtype))

        # Apply RoPE to the last `rope_dim` dims, sampled at chunk-end positions.
        # Reference uses freqs_cis[:cutoff:ratio] — i.e., one cos/sin per output chunk.
        idx = torch.arange(0, cutoff, ratio, device=cos.device)
        kv_rope = apply_rotary_emb_inplace_slice(
            kv[..., -rd:], cos[idx], sin[idx]
        )
        kv = torch.cat([kv[..., : -rd], kv_rope], dim=-1)
        return kv


# ------------------------------------------------------------------------------
# Indexer (selects top-k positions from compressed KV for sparse attention)
# ------------------------------------------------------------------------------
class Indexer(nn.Module):
    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.index_topk = config.index_topk
        self.compress_ratio = compress_ratio

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5
        # Indexer uses its own KV compressor (head_dim = index_head_dim).
        self.compressor = Compressor(config, compress_ratio, self.head_dim)

    def forward(
        self,
        x_full: torch.Tensor,
        qr_new: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        cos_new: torch.Tensor,
        sin_new: torch.Tensor,
        offset: int,
        seqlen_new: int,
    ) -> torch.Tensor:
        """Returns topk_idxs: [B, S_new, index_topk] of int32 — indexer scores for the LAST
        `seqlen_new` positions of `x_full`. Indices reference positions in the gathered-KV
        array starting from `offset`. Out-of-range positions are -1.

        x_full / cos_full / sin_full cover the full sequence (past + new); qr_new / cos_new /
        sin_new cover only the new tokens. When seqlen_new == x_full.size(1) this reduces to
        the prefill path."""
        bsz, seqlen_total, _ = x_full.size()
        ratio = self.compress_ratio
        rd = self.rope_dim
        start_new = seqlen_total - seqlen_new

        q = self.wq_b(qr_new).unflatten(-1, (self.n_heads, self.head_dim))  # [B, S_new, H, d]
        q_rope = apply_rotary_emb_inplace_slice(q[..., -rd:], cos_new, sin_new)
        q = torch.cat([q[..., : -rd], q_rope], dim=-1)

        kv = self.compressor(x_full, cos_full, sin_full)  # [B, S_total/ratio, d]
        weights = self.weights_proj(x_full[:, start_new:]).float() * (self.softmax_scale * self.n_heads ** -0.5)

        # index_score[b, s, h, t] = q[b,s,h,:] · kv[b,t,:]
        index_score = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
        index_score = (F.relu(index_score) * weights.unsqueeze(-1)).sum(dim=2)  # [B, S_new, T]

        # Causal mask: absolute position s can only attend to compressed-chunks with end <= s.
        n_t = index_score.size(-1)
        s_idx = torch.arange(start_new, start_new + seqlen_new, device=index_score.device)
        t_idx = torch.arange(n_t, device=index_score.device)
        mask = t_idx.unsqueeze(0) >= ((s_idx + 1) // ratio).unsqueeze(1)
        index_score = index_score.masked_fill(mask.unsqueeze(0), float("-inf"))

        k = min(self.index_topk, n_t)
        topk_idxs = index_score.topk(k, dim=-1)[1]  # [B, S_new, k]
        invalid = topk_idxs >= ((s_idx + 1) // ratio).unsqueeze(0).unsqueeze(-1)
        topk_idxs = torch.where(invalid, torch.full_like(topk_idxs, -1), topk_idxs + offset)
        return topk_idxs.to(torch.int32)


# ------------------------------------------------------------------------------
# Sliding-window topk indices (no learned scoring, just window-aligned)
# ------------------------------------------------------------------------------
def sliding_window_topk_idxs(window: int, bsz: int, seqlen_total: int, seqlen_new: int, device) -> torch.Tensor:
    """Returns [B, seqlen_new, window] int32: window indices for the last `seqlen_new`
    positions of a sequence of length `seqlen_total`. Indices are absolute positions
    in [0, seqlen_total). Out-of-range positions are -1."""
    start = seqlen_total - seqlen_new
    base = torch.arange(start, start + seqlen_new, device=device).unsqueeze(1)  # [S_new, 1]
    win = torch.arange(window, device=device).unsqueeze(0)                       # [1, W]
    idxs = base - window + 1 + win                                                # [S_new, W]
    idxs = torch.where(idxs < 0, torch.full_like(idxs, -1), idxs)
    idxs = torch.where(idxs > base, torch.full_like(idxs, -1), idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1).to(torch.int32).contiguous()


def dense_compress_topk_idxs(ratio: int, bsz: int, seqlen_total: int, seqlen_new: int, offset: int, device) -> torch.Tensor:
    """For ratio==128 (no learned indexer), select all causally-valid compressed positions
    for the last `seqlen_new` positions of a sequence of length `seqlen_total`."""
    n_t = seqlen_total // ratio
    start = seqlen_total - seqlen_new
    s_idx = torch.arange(start, start + seqlen_new, device=device).unsqueeze(1)  # [S_new, 1]
    t_idx = torch.arange(n_t, device=device).unsqueeze(0)                         # [1, T]
    valid = t_idx < ((s_idx + 1) // ratio)
    idxs = torch.where(valid, t_idx + offset, torch.full_like(t_idx, -1))
    idxs = idxs.unsqueeze(0).expand(bsz, seqlen_new, -1).to(torch.int32).contiguous()
    return idxs


# ------------------------------------------------------------------------------
# Sparse attention (dense gather + softmax — "sparse" only in indices)
# ------------------------------------------------------------------------------
def sparse_attn_dense(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    q:           [B, S, H, d]
    kv:          [B, T, d]                (single shared K/V per position; MQA with 1 KV head)
    attn_sink:   [H]                       fp32 — extra learned key with score=sink
    topk_idxs:   [B, S, K] int32; -1 marks invalid
    Returns o:   [B, S, H, d]
    """
    bsz, seqlen, n_heads, head_dim = q.shape
    K = topk_idxs.size(-1)
    # Gather: kv_gather[b, s, k, d] = kv[b, topk_idxs[b, s, k], d], with -1 → zero key.
    safe_idxs = topk_idxs.clamp(min=0).long()                          # [B, S, K]
    invalid = (topk_idxs < 0).unsqueeze(-1)                             # [B, S, K, 1]
    # Expand kv to [B, S, T, d] is too expensive. Use gather with broadcasting:
    #   kv:        [B, T, d]  -> broadcast over S
    #   gather idx: [B, S, K, d]
    gather_idx = safe_idxs.unsqueeze(-1).expand(bsz, seqlen, K, head_dim)
    kv_b = kv.unsqueeze(1).expand(bsz, seqlen, kv.size(1), head_dim)   # [B, S, T, d]
    kv_gathered = torch.gather(kv_b, 2, gather_idx)                     # [B, S, K, d]
    kv_gathered = kv_gathered.masked_fill(invalid, 0.0)

    # Attention scores: [B, S, H, K]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), kv_gathered.float()) * softmax_scale
    # Mask invalid keys.
    invalid_hk = (topk_idxs < 0).unsqueeze(2)                           # [B, S, 1, K]
    scores = scores.masked_fill(invalid_hk, float("-inf"))

    # attn_sink as one extra key with logit attn_sink[h] (no value contribution; sums into denom only).
    sink = attn_sink.view(1, 1, n_heads, 1).expand(bsz, seqlen, n_heads, 1).float()
    scores_with_sink = torch.cat([scores, sink], dim=-1)                # [B, S, H, K+1]
    probs = scores_with_sink.softmax(dim=-1)
    probs_keys = probs[..., :K]                                          # drop sink for value mixing

    o = torch.einsum("bshk,bskd->bshd", probs_keys, kv_gathered.float())
    return o.to(q.dtype)


# ------------------------------------------------------------------------------
# Attention block (MLA-like with Q/O LoRA, sliding window + optional compression)
# ------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_id]
        self.eps = config.rms_norm_eps

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        # Single shared K/V projection (MQA with 1 KV head).
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(
            (self.n_heads * self.head_dim) // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            self.indexer = Indexer(config, self.compress_ratio) if self.compress_ratio == 4 else None
        else:
            self.compressor = None
            self.indexer = None

    def forward(
        self,
        x_full: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        seqlen_new: Optional[int] = None,
    ) -> torch.Tensor:
        """Run attention for the LAST `seqlen_new` positions of `x_full`. If `seqlen_new`
        is None or equals `x_full.size(1)`, this is the prefill path (compute for all
        positions). x_full / cos_full / sin_full always span the full sequence so that
        the K/V projection and compressor see the full history."""
        bsz, seqlen_total, _ = x_full.size()
        if seqlen_new is None:
            seqlen_new = seqlen_total
        start_new = seqlen_total - seqlen_new
        x_new = x_full[:, start_new:]
        cos_new = cos_full[start_new:]
        sin_new = sin_full[start_new:]
        rd = self.rope_dim
        win = self.window_size

        # Q (new positions only)
        qr = self.q_norm(self.wq_a(x_new))                                # [B, S_new, q_lora]
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))    # [B, S_new, H, d]
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + self.eps).to(q.dtype)
        q_rope = apply_rotary_emb_inplace_slice(q[..., -rd:], cos_new, sin_new)
        q = torch.cat([q[..., : -rd], q_rope], dim=-1)

        # K/V (full sequence, shared single head)
        kv = self.wkv(x_full)                                              # [B, S_total, d]
        kv = self.kv_norm(kv)
        kv_rope = apply_rotary_emb_inplace_slice(kv[..., -rd:], cos_full, sin_full)
        kv = torch.cat([kv[..., : -rd], kv_rope], dim=-1)

        # Build the gather pool for new positions.
        win_idxs = sliding_window_topk_idxs(win, bsz, seqlen_total, seqlen_new, x_full.device)
        if self.compress_ratio:
            kv_compress = self.compressor(x_full, cos_full, sin_full)     # [B, T, d]
            offset = kv.size(1)
            if self.indexer is not None:
                comp_idxs = self.indexer(
                    x_full, qr, cos_full, sin_full, cos_new, sin_new, offset, seqlen_new
                )
            else:
                comp_idxs = dense_compress_topk_idxs(
                    self.compress_ratio, bsz, seqlen_total, seqlen_new, offset, x_full.device
                )
            topk_idxs = torch.cat([win_idxs, comp_idxs], dim=-1)
            kv_pool = torch.cat([kv, kv_compress], dim=1)
        else:
            topk_idxs = win_idxs
            kv_pool = kv

        o = sparse_attn_dense(q, kv_pool, self.attn_sink, topk_idxs, self.softmax_scale)
        # Inverse RoPE on last 2*rd dims of o (reference applies conjugate freqs).
        o_rope = apply_rotary_emb_inplace_slice(o[..., -rd:], cos_new, -sin_new)
        o = torch.cat([o[..., : -rd], o_rope], dim=-1)

        # Output projection: grouped LoRA.
        o = o.reshape(bsz, seqlen_new, self.n_groups, -1)                 # [B, S_new, G, head_chunk]
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgc,grc->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


# ------------------------------------------------------------------------------
# MoE Gate + Expert + MoE block
# ------------------------------------------------------------------------------
class Gate(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_id: int):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.hash = layer_id < config.num_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        if self.hash:
            self.register_buffer(
                "tid2eid",
                torch.zeros(config.vocab_size, config.num_experts_per_tok, dtype=torch.int64),
                persistent=True,
            )
            self.bias = None
        else:
            self.tid2eid = None
            self.bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        # x: [N, dim] (N == B*S), input_ids: [N]
        scores = F.linear(x.float(), self.weight.float())               # [N, E]
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]                            # [N, topk]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]                  # [N, topk]
        weights = original_scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, -self.swiglu_limit, self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        return self.w2(h.to(x.dtype))


class MoE(nn.Module):
    """Compute-all-experts MoE: graph-friendly (no Python-side dispatch).
    Cost = E * Expert_cost; for the toy E is small. For real inference the loop-over-experts
    pattern is faster but does not export cleanly."""

    def __init__(self, config: DeepseekV4Config, layer_id: int):
        super().__init__()
        self.dim = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.gate = Gate(config, layer_id)
        self.experts = nn.ModuleList(
            [Expert(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)
             for _ in range(config.n_routed_experts)]
        )
        assert config.n_shared_experts == 1, "only n_shared_experts=1 is supported"
        self.shared_experts = Expert(
            config.hidden_size, config.moe_intermediate_size, config.swiglu_limit
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, S, dim]; input_ids: [B, S]
        b, s, d = x.shape
        x_flat = x.reshape(b * s, d)
        ids_flat = input_ids.reshape(b * s)
        weights, indices = self.gate(x_flat, ids_flat)                  # [N, topk] each

        # Build a sparse [N, E] gate matrix g where g[n, e] = weights[n, j] if indices[n, j] == e else 0.
        N = b * s
        E = self.n_routed_experts
        gate_mat = x_flat.new_zeros(N, E, dtype=torch.float32)
        gate_mat.scatter_(1, indices, weights.float())

        # Run all experts, then take a weighted sum.
        # expert_out[e]: [N, d] — stack along E.
        outs = torch.stack([self.experts[e](x_flat) for e in range(E)], dim=1)  # [N, E, d]
        y = (gate_mat.unsqueeze(-1) * outs.float()).sum(dim=1)
        y = y + self.shared_experts(x_flat).float()
        return y.to(x.dtype).reshape(b, s, d)


# ------------------------------------------------------------------------------
# Hyper-Connections (mHC)
# ------------------------------------------------------------------------------
def hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    hc_mult: int, sinkhorn_iters: int, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes: [B, S, (2+hc_mult)*hc_mult] float32
       Returns (pre[B,S,hc_mult], post[B,S,hc_mult], comb[B,S,hc_mult,hc_mult])."""
    H = hc_mult
    pre_mix = mixes[..., :H]                                            # [B, S, H]
    post_mix = mixes[..., H : 2 * H]
    comb_mix = mixes[..., 2 * H :].view(*mixes.shape[:-1], H, H)        # [B, S, H, H]

    pre = torch.sigmoid(pre_mix * hc_scale[0] + hc_base[:H]) + eps
    post = 2 * torch.sigmoid(post_mix * hc_scale[1] + hc_base[H : 2 * H])
    comb = comb_mix * hc_scale[2] + hc_base[2 * H :].view(H, H)

    # Sinkhorn balancing: row-softmax then alternating row/col normalization.
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class Block(nn.Module):
    """V4 transformer block with Hyper-Connections.
    Hidden state shape throughout is [B, S, hc_mult, dim]."""

    def __init__(self, config: DeepseekV4Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps

        self.attn = Attention(config, layer_id)
        self.ffn = MoE(config, layer_id)
        self.attn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, self.norm_eps)

        H = self.hc_mult
        mix_hc = (2 + H) * H
        hc_dim = H * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))

    def _hc_pre(self, x: torch.Tensor, hc_fn, hc_scale, hc_base):
        """x: [B, S, H, dim] -> (y: [B, S, dim] reduced via pre weights, post: [B,S,H], comb: [B,S,H,H])."""
        b, s, H, dim = x.shape
        x_flat = x.reshape(b, s, H * dim).float()
        rsq = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsq
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)                  # [B, S, dim]
        return y.to(x.dtype), post, comb

    def _hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [B,S,dim]; residual: [B,S,H,dim]; post: [B,S,H]; comb: [B,S,H,H]
        # y[b,s,h,d] = post[b,s,h] * x[b,s,d] + sum_k comb[b,s,k,h] * residual[b,s,k,d]
        # (note: reference summed comb over the second axis with comb.unsqueeze(-1) * residual.unsqueeze(-2),
        #  which effectively pairs residual[k] with comb[k, h] for each output h.)
        new_branch = post.unsqueeze(-1) * x.unsqueeze(-2)                # [B,S,H,dim] (broadcast over H)
        from_res = torch.einsum("bskh,bskd->bshd", comb.float(), residual.float())
        return (new_branch.float() + from_res).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        input_ids_new: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        past_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """When past_x is None: prefill — `x` covers the full sequence; output also full.
        When past_x is given: decode — `x` covers only new tokens (shape [B, S_new, H, dim]).
        Internally we concat past+new to give the K/V projection and compressor full context,
        but the returned block output covers only the new positions. Caller is responsible
        for using `present_x` (returned) as the next call's `past_x`."""
        # Always take the concat path so OpenVINO can trace past_x as a regular input.
        # past_x with sequence length 0 yields the prefill behavior bit-for-bit.
        if past_x is None:
            past_x = x.new_zeros(x.size(0), 0, x.size(2), x.size(3))
        x_full = torch.cat([past_x, x], dim=1)
        seqlen_new = x.size(1)
        present_x = x_full

        # Attention sub-block — pre-mix and norm over the full sequence.
        y_full, post_full, comb_full = self._hc_pre(x_full, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        y_full = self.attn_norm(y_full)
        y_new = self.attn(y_full, cos_full, sin_full, seqlen_new=seqlen_new)
        # _hc_post mixes attention output back with residual; both must align to NEW positions only.
        residual_new = x_full[:, -seqlen_new:]
        post_new = post_full[:, -seqlen_new:]
        comb_new = comb_full[:, -seqlen_new:]
        x_new = self._hc_post(y_new, residual_new, post_new, comb_new)

        # FFN sub-block — operates per-token, so we can run it on new positions only.
        # But _hc_pre needs the full context to compute mixes consistent with prefill semantics?
        # No: _hc_pre is per-position (no cross-position interaction), so running on new
        # positions only is correct and matches prefill bit-for-bit.
        y2, post2, comb2 = self._hc_pre(x_new, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        y2 = self.ffn_norm(y2)
        y2 = self.ffn(y2, input_ids_new)
        x_new = self._hc_post(y2, x_new, post2, comb2)

        # Update present_x to reflect this layer's new INPUT-stream tokens.
        # Caller will use this as past_x next call.
        return x_new, present_x


# ------------------------------------------------------------------------------
# Top-level model
# ------------------------------------------------------------------------------
class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # HC-head reduce parameters (mirror reference Transformer.hc_head_*).
        H = config.hc_mult
        hc_dim = H * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(H, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.zeros(H, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

        # Precompute RoPE for max_seq_len once. The forward picks the [:S] slice.
        rope_scaling = config.rope_scaling or {}
        cos_w, sin_w = precompute_rope_cos_sin(
            rope_dim=config.qk_rope_head_dim,
            seq_len=config.max_position_embeddings,
            base=config.rope_theta,
            yarn_factor=rope_scaling.get("factor", 1.0),
            yarn_original_max_pos=rope_scaling.get("original_max_position_embeddings", 0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
        )
        # NOTE: reference uses different rope_theta for compressed (compress_rope_theta=160000)
        # vs window-only attention (rope_theta=10000). For toy PoC we use one set.
        self.register_buffer("rope_cos", cos_w, persistent=False)
        self.register_buffer("rope_sin", sin_w, persistent=False)

        self.post_init()

    def _hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H, dim] -> [B, S, dim]
        b, s, H, dim = x.shape
        x_flat = x.reshape(b, s, H * dim).float()
        rsq = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsq                 # [B, S, H]
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(x.dtype)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """`past_key_values` is a list of per-layer cached block inputs, each shape
        [B, S_past, hc_mult, dim]. If given, `input_ids` is interpreted as new tokens only
        and the model returns logits / last_hidden_state for those new positions. If
        `use_cache=True`, the returned `past_key_values` contains the updated per-layer
        inputs (length = num_hidden_layers, each [B, S_past + S_new, hc_mult, dim])."""
        bsz, seqlen_new = input_ids.shape
        seqlen_past = 0 if past_key_values is None else past_key_values[0].size(1)
        seqlen_total = seqlen_past + seqlen_new

        h = self.embed(input_ids)                                       # [B, S_new, dim]
        h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()  # [B, S_new, H, dim]

        # RoPE for the full sequence; layers slice as needed.
        cos_full = self.rope_cos[:seqlen_total]
        sin_full = self.rope_sin[:seqlen_total]

        present_key_values: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            past_x = None if past_key_values is None else past_key_values[i]
            h, present_x = layer(h, input_ids, cos_full, sin_full, past_x=past_x)
            if use_cache:
                present_key_values.append(present_x)

        h = self._hc_head_reduce(h)                                     # [B, S_new, dim]
        h = self.norm(h)
        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=present_key_values if use_cache else None,
        )


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values)
