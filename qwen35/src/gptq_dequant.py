"""GPTQ int4 dequantization for the Qwen3.5-4B checkpoint.

Format verified against actual GPTQ safetensors:
- qweight: int32, shape (in_features // 8, out_features).
- qzeros: int32, shape (num_groups, out_features // 8).
- scales: float16, shape (num_groups, out_features).
- g_idx: int32, shape (in_features,).
"""
from __future__ import annotations

from pathlib import Path
import torch

_INT4_MASK = 0xF
_INT4_PER_INT32 = 8
_DEFAULT_GROUP_SIZE = 128


def _shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(0, 32, 4, dtype=torch.int32, device=device)


def unpack_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack int4 weights packed along input dim."""
    if qweight.dtype != torch.int32:
        raise TypeError(f"qweight must be int32, got {qweight.dtype}")
    rows_packed, out_features = qweight.shape
    expanded = qweight.unsqueeze(-1)  # (R, O, 1)
    unpacked = (expanded >> _shifts(qweight.device)) & _INT4_MASK  # (R, O, 8)
    unpacked = unpacked.permute(0, 2, 1).contiguous()
    return unpacked.reshape(rows_packed * _INT4_PER_INT32, out_features)


def unpack_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack int4 zeros packed along output dim."""
    if qzeros.dtype != torch.int32:
        raise TypeError(f"qzeros must be int32, got {qzeros.dtype}")
    num_groups, cols_packed = qzeros.shape
    expanded = qzeros.unsqueeze(-1)  # (G, Cp, 1)
    unpacked = (expanded >> _shifts(qzeros.device)) & _INT4_MASK  # (G, Cp, 8)
    return unpacked.reshape(num_groups, cols_packed * _INT4_PER_INT32)


def dequantize_gptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor | None = None,
    group_size: int = _DEFAULT_GROUP_SIZE,
    out_dtype: torch.dtype = torch.float32,
    zero_offset: int = 1,
) -> torch.Tensor:
    """Dequantize one GPTQ int4 linear weight."""
    raw = unpack_qweight(qweight)  # (in, out)
    unpacked_zeros = unpack_qzeros(qzeros)  # (G, out)
    in_features, out_features = raw.shape

    if g_idx is None:
        g_idx = torch.arange(in_features, device=raw.device) // group_size
    g_idx_long = g_idx.long()

    row_zeros = unpacked_zeros[g_idx_long] + zero_offset  # (in, out)
    row_scales = scales[g_idx_long].to(out_dtype)  # (in, out)
    return (raw - row_zeros).to(out_dtype) * row_scales


def dequantize_linear_for_nn(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor | None = None,
    group_size: int = _DEFAULT_GROUP_SIZE,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Returns (out_features, in_features) for direct assignment to nn.Linear.weight."""
    w = dequantize_gptq(qweight, qzeros, scales, g_idx, group_size, out_dtype)
    return w.t().contiguous()


def load_gptq_linear(
    weight_map: dict[str, str],
    model_dir: Path | str,
    prefix: str,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a single GPTQ-stored linear from safetensors."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    names = {leaf: f"{prefix}.{leaf}" for leaf in ("qweight", "qzeros", "scales", "g_idx")}
    shards = {leaf: model_dir / weight_map[full] for leaf, full in names.items()}

    tensors: dict[str, torch.Tensor] = {}
    handles: dict[Path, object] = {}
    try:
        for leaf, shard_path in shards.items():
            if shard_path not in handles:
                handles[shard_path] = safe_open(str(shard_path), framework="pt").__enter__()
            tensors[leaf] = handles[shard_path].get_tensor(names[leaf])
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)

    return dequantize_linear_for_nn(
        tensors["qweight"],
        tensors["qzeros"],
        tensors["scales"],
        tensors["g_idx"],
        out_dtype=out_dtype,
    )
