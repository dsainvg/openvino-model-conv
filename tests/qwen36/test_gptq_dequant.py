"""Tests for src/qwen36/gptq_dequant.py.

The "real" tests load tensors from the on-disk Qwen3.6 GPTQ checkpoint at
the path below. They are skipped if the checkpoint is not present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.qwen36.gptq_dequant import (  # noqa: E402
    build_shard_index,
    dequantize_gptq,
    load_gptq_linear,
    unpack_qweight,
    unpack_qzeros,
)


def _i32(values):
    """Build an int32 tensor from values that may overflow signed int32 (PyTorch
    refuses 0x88888888 etc directly; go through int64 then cast)."""
    return torch.tensor(values, dtype=torch.int64).to(torch.int32)


MODEL_DIR = Path(r"C:\Users\intel\models\qwen36-35b-int4")
HAVE_MODEL = MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors.index.json").exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="Qwen3.6 GPTQ checkpoint not on disk")


# ---------------------------------------------------------------------------
# Pure unit tests for the bit-unpacking primitives
# ---------------------------------------------------------------------------


def test_unpack_qweight_known_value():
    """0x76543210 should unpack to [0, 1, 2, 3, 4, 5, 6, 7] along the row dim."""
    packed = torch.tensor([[0x76543210]], dtype=torch.int32)
    unpacked = unpack_qweight(packed)
    assert unpacked.shape == (8, 1)
    assert unpacked.flatten().tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_unpack_qweight_two_columns_two_packed_rows():
    """Verify the packed-row -> input-row mapping is correct across multiple
    packed rows and columns."""
    # Row 0 col 0: [0..7], row 0 col 1: [8..15 mod 16] = [8,9,10,11,12,13,14,15]
    # Row 1 col 0: [15,14..8], row 1 col 1: [7,6..0]
    packed = _i32(
        [
            [0x76543210, 0xFEDCBA98],
            [0x89ABCDEF, 0x01234567],
        ]
    )
    unpacked = unpack_qweight(packed)
    assert unpacked.shape == (16, 2)
    assert unpacked[:8, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert unpacked[:8, 1].tolist() == [8, 9, 10, 11, 12, 13, 14, 15]
    assert unpacked[8:, 0].tolist() == [15, 14, 13, 12, 11, 10, 9, 8]
    assert unpacked[8:, 1].tolist() == [7, 6, 5, 4, 3, 2, 1, 0]


def test_unpack_qzeros_known_value():
    """0x77777777 -> all 7s along the output dim (gptqmodel sym=True case)."""
    packed = torch.tensor([[0x77777777]], dtype=torch.int32)
    unpacked = unpack_qzeros(packed)
    assert unpacked.shape == (1, 8)
    assert unpacked.flatten().tolist() == [7] * 8


def test_dequantize_gptq_sym_zero_is_8():
    """For sym=True the effective zero (unpacked + 1) is 8. With raw=8 and
    scale=1 we should get exactly zeros out."""
    # qzeros packs 8 outputs per int32, so smallest legal out_features is 8.
    qweight = _i32([[0x88888888] * 8])  # 1 packed row x 8 output cols, all raw=8
    qzeros = _i32([[0x77777777]])  # 1 group x 1 packed col -> 8 output cols of 7
    scales = torch.ones((1, 8), dtype=torch.float16)
    g_idx = torch.zeros(8, dtype=torch.int32)
    out = dequantize_gptq(qweight, qzeros, scales, g_idx)
    assert out.shape == (8, 8)
    assert torch.all(out == 0.0), out


def test_dequantize_gptq_known_nonzero():
    """Verify (raw - 8) * scale gives the right magnitude on a hand-crafted case."""
    # Pack int4 [0, 8, 15, 0, ...] into one int32 along the row axis.
    # Per unpack convention: lsb -> row 0. So [0, 8, 15, 0, 0, 0, 0, 0]
    # packs to 0x00000F80 (row 0=0x0, row 1=0x8, row 2=0xF, rest=0).
    packed = (0 << 0) | (8 << 4) | (15 << 8)
    # out_features must be a multiple of 8 (qzeros packing constraint).
    # Replicate the same packed col across 8 output columns so we can test
    # one column and trust the others by symmetry.
    qweight = _i32([[packed] * 8])
    qzeros = _i32([[0x77777777]])  # effective zero = 8 on all outputs
    scales = torch.full((1, 8), 2.0, dtype=torch.float16)
    g_idx = torch.zeros(8, dtype=torch.int32)
    out = dequantize_gptq(qweight, qzeros, scales, g_idx)
    # row 0: (0 - 8) * 2 = -16; row 1: (8 - 8) * 2 = 0; row 2: (15 - 8) * 2 = 14
    assert out[0, 0].item() == -16.0
    assert out[1, 0].item() == 0.0
    assert out[2, 0].item() == 14.0
    assert out[3, 0].item() == -16.0  # raw=0 again


# ---------------------------------------------------------------------------
# Real-checkpoint integration test
# ---------------------------------------------------------------------------


@needs_model
def test_load_one_real_expert_down_proj():
    weight_map = build_shard_index(MODEL_DIR)
    prefix = "model.language_model.layers.0.mlp.experts.0.down_proj"
    w = load_gptq_linear(weight_map, MODEL_DIR, prefix, out_dtype=torch.float32)
    # down_proj: maps intermediate (512) -> hidden (2048); nn.Linear weight is (out, in)
    assert w.shape == (2048, 512), w.shape
    assert w.dtype == torch.float32
    # Sanity: not all zero, not all nan, reasonable magnitude for BF16-trained weights
    assert torch.isfinite(w).all()
    assert w.abs().max().item() > 0.0
    assert w.abs().max().item() < 1.0  # quantized weights for FFNs are typically O(0.1)
    # Mean should be close to 0 for a well-trained projection
    assert abs(w.mean().item()) < 0.01


@needs_model
def test_load_one_real_expert_gate_proj():
    weight_map = build_shard_index(MODEL_DIR)
    prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    w = load_gptq_linear(weight_map, MODEL_DIR, prefix, out_dtype=torch.float32)
    # gate_proj: hidden (2048) -> intermediate (512); nn.Linear weight is (out, in)
    assert w.shape == (512, 2048), w.shape
    assert torch.isfinite(w).all()
    assert w.abs().max().item() > 0.0


@needs_model
def test_multiple_experts_have_different_weights():
    """If we mis-indexed shards we might load the same data for every expert."""
    weight_map = build_shard_index(MODEL_DIR)
    w0 = load_gptq_linear(weight_map, MODEL_DIR, "model.language_model.layers.0.mlp.experts.0.gate_proj")
    w1 = load_gptq_linear(weight_map, MODEL_DIR, "model.language_model.layers.0.mlp.experts.1.gate_proj")
    assert not torch.allclose(w0, w1), "expert 0 and expert 1 gate_proj are identical -- loader bug?"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
