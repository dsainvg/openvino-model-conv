"""GPTQ int4 dequantization for the palmfuture Qwen3.6 checkpoint.

Format verified against the actual safetensors (gptqmodel:6.0.3, sym=True,
group_size=128, desc_act=False, pack_dtype=int32):

- qweight: int32, shape (in_features // 8, out_features). 8 int4 weights
  packed along the input axis. Bits (j*4..j*4+4) of qweight[i, o] hold
  the int4 value at input row (i*8 + j) for output column o.

- qzeros: int32, shape (num_groups, out_features // 8). 8 int4 zeros
  packed along the output axis. Each int4 stores the zero MINUS ONE
  (gptqmodel/auto-gptq convention) so that 0 maps to -1 = 0xF.
  Effective zero = unpacked_zero + 1.

- scales: float16, shape (num_groups, out_features). Per-(group, output)
  scale factor.

- g_idx: int32, shape (in_features,). Group index per input row. With
  desc_act=False this is arange(in_features) // group_size.

Dequant: w[i, o] = (raw_int4[i, o] - effective_zero[g_idx[i], o]) * scale[g_idx[i], o]
                 = (raw_int4[i, o] - unpacked_zero[g_idx[i], o] - 1) * scale[g_idx[i], o]

For sym=True all unpacked zeros are 7, so effective zero is uniformly 8
(midpoint of int4 range [0, 15]).
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
    """Unpack int4 weights packed along input dim.

    Input  (rows_packed, out_features)  int32, 8 int4s per packed entry.
    Output (rows_packed * 8, out_features) int32, values in [0, 15].
    """
    if qweight.dtype != torch.int32:
        raise TypeError(f"qweight must be int32, got {qweight.dtype}")
    rows_packed, out_features = qweight.shape
    expanded = qweight.unsqueeze(-1)  # (R, O, 1)
    unpacked = (expanded >> _shifts(qweight.device)) & _INT4_MASK  # (R, O, 8)
    # (R, O, 8) -> (R, 8, O) so that within each R block the 8 input rows
    # appear in increasing order along dim 1, then flatten to (R*8, O).
    unpacked = unpacked.permute(0, 2, 1).contiguous()
    return unpacked.reshape(rows_packed * _INT4_PER_INT32, out_features)


def unpack_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack int4 zeros packed along output dim.

    Input  (num_groups, cols_packed) int32.
    Output (num_groups, cols_packed * 8) int32, values in [0, 15].
    """
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
    """Dequantize one GPTQ int4 linear weight.

    Returns a tensor of shape (in_features, out_features) in `out_dtype`.
    Convert to (out_features, in_features) yourself if you need it for
    nn.Linear.weight.

    `zero_offset` is added to the unpacked qzeros before subtraction. The
    gptqmodel/auto-gptq convention is 1 (stored zero is value - 1, so 0
    encodes as 15). Pass 0 if you have a checkpoint that stores raw zeros.
    """
    raw = unpack_qweight(qweight)  # (in, out) int32 in [0, 15]
    unpacked_zeros = unpack_qzeros(qzeros)  # (G, out) int32 in [0, 15]
    in_features, out_features = raw.shape

    if g_idx is None:
        g_idx = torch.arange(in_features, device=raw.device) // group_size
    if g_idx.shape != (in_features,):
        raise ValueError(f"g_idx shape {g_idx.shape} != ({in_features},)")
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
    """Same as dequantize_gptq but returns (out_features, in_features) for
    direct assignment to `nn.Linear.weight.data` (PyTorch's convention)."""
    w = dequantize_gptq(qweight, qzeros, scales, g_idx, group_size, out_dtype)
    return w.t().contiguous()


# ---------------------------------------------------------------------------
# Safetensors helpers
# ---------------------------------------------------------------------------


def build_shard_index(model_dir: Path | str) -> dict[str, str]:
    """Read model.safetensors.index.json and return tensor_name -> shard filename."""
    import json

    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as fh:
        return json.load(fh)["weight_map"]


def load_gptq_linear(
    weight_map: dict[str, str],
    model_dir: Path | str,
    prefix: str,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a single GPTQ-stored linear from safetensors.

    `prefix` is the dotted weight prefix (e.g. `model.language_model.layers.0
    .mlp.experts.0.down_proj`) such that `prefix + ".qweight"` etc exist in
    the weight_map.

    Returns (out_features, in_features) ready for nn.Linear.weight.data.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    names = {leaf: f"{prefix}.{leaf}" for leaf in ("qweight", "qzeros", "scales", "g_idx")}
    shards = {leaf: model_dir / weight_map[full] for leaf, full in names.items()}

    # Open each shard once. In practice all four are in the same shard.
    tensors: dict[str, torch.Tensor] = {}
    handles: dict[Path, object] = {}
    try:
        for leaf, shard_path in shards.items():
            if shard_path not in handles:
                handles[shard_path] = safe_open(shard_path, framework="pt").__enter__()
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


def load_plain_tensor(
    weight_map: dict[str, str],
    model_dir: Path | str,
    name: str,
) -> torch.Tensor:
    """Load a non-quantized tensor (BF16/FP16/etc) by full dotted name."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    shard = model_dir / weight_map[name]
    with safe_open(shard, framework="pt") as fh:
        return fh.get_tensor(name)
