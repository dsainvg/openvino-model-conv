"""Unit tests for the FP4 and FP8 dequantization helpers in
scripts/load_real_v4_weights.py. Verified with synthetic tensors of known
content; we cannot run the loader on real weights from this 64GB host.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import torch

from load_real_v4_weights import (
    FP4_TABLE,
    dequant_fp4,
    dequant_fp8,
    map_real_to_ours,
)


def test_fp4_known_byte():
    """Byte 0x12 -> low=2, high=1 -> [FP4_TABLE[2], FP4_TABLE[1]] = [1.0, 0.5]."""
    # one row, 32 packed bytes = 64 unpacked vals; one block of 32 cols means
    # we get 2 blocks of 32 cols across the 64 vals -> scale shape (1, 2).
    # Make every byte 0x12 so unpacked is alternating [1.0, 0.5, 1.0, 0.5, ...].
    packed = torch.full((1, 32), 0x12, dtype=torch.int8)
    scale = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    out = dequant_fp4(packed, scale)
    assert out.shape == (1, 64)
    expected = torch.tensor([1.0, 0.5] * 32, dtype=torch.float32).unsqueeze(0)
    assert torch.allclose(out, expected), f"{out[0, :8]} vs {expected[0, :8]}"
    print("  fp4_known_byte OK")


def test_fp4_sign_bit():
    """FP4_TABLE[8..15] are negatives. byte 0x89 -> low=9 (-0.5), high=8 (-0.0)."""
    packed = torch.full((1, 32), 0x89 - 256, dtype=torch.int8)  # 0x89 as signed int8
    scale = torch.ones((1, 2), dtype=torch.float32)
    out = dequant_fp4(packed, scale)
    expected = torch.tensor([-0.5, -0.0] * 32, dtype=torch.float32).unsqueeze(0)
    assert torch.allclose(out, expected)
    print("  fp4_sign_bit OK")


def test_fp4_per_block_scale():
    """Different scale per 32-col block must apply correctly."""
    # 0x21: low=1 -> 0.5, high=2 -> 1.0. Two blocks of 32 cols: scale [2, 4].
    packed = torch.full((1, 32), 0x21, dtype=torch.int8)
    scale = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    out = dequant_fp4(packed, scale)
    block0 = out[0, :32]
    block1 = out[0, 32:]
    expected0 = torch.tensor([0.5, 1.0] * 16) * 2.0
    expected1 = torch.tensor([0.5, 1.0] * 16) * 4.0
    assert torch.allclose(block0, expected0)
    assert torch.allclose(block1, expected1)
    print("  fp4_per_block_scale OK")


def test_fp8_passthrough_unit_scale():
    """FP8 dequant with unit scale should match weight.float() pointwise."""
    torch.manual_seed(0)
    out_dim, in_dim = 256, 256
    raw = torch.randn(out_dim, in_dim, dtype=torch.float32) * 6
    fp8 = raw.to(torch.float8_e4m3fn)
    scale = torch.ones((out_dim // 128, in_dim // 128), dtype=torch.float32)
    out = dequant_fp8(fp8, scale)
    assert out.shape == (out_dim, in_dim)
    assert torch.equal(out, fp8.float())
    print("  fp8_passthrough_unit_scale OK")


def test_fp8_block_scale():
    """Two-block-by-two-block weight with distinct per-block scales."""
    out_dim, in_dim = 256, 256
    fp8 = torch.ones(out_dim, in_dim, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    out = dequant_fp8(fp8, scale)
    # block (0,0): rows 0:128, cols 0:128 -> 1*1 = 1.0
    assert torch.allclose(out[:128, :128], torch.full((128, 128), 1.0))
    assert torch.allclose(out[:128, 128:], torch.full((128, 128), 2.0))
    assert torch.allclose(out[128:, :128], torch.full((128, 128), 3.0))
    assert torch.allclose(out[128:, 128:], torch.full((128, 128), 4.0))
    print("  fp8_block_scale OK")


def test_name_mapping_skips():
    assert map_real_to_ours("mtp.0.attn.wq_a.weight") is None
    assert map_real_to_ours("layers.0.ffn.gate.tid2eid") is None
    assert map_real_to_ours("layers.5.ffn.gate.bias") is None
    print("  name_mapping_skips OK")


def test_name_mapping_top_level():
    assert map_real_to_ours("embed.weight") == "model.embed.weight"
    assert map_real_to_ours("head.weight") == "lm_head.weight"
    assert map_real_to_ours("norm.weight") == "model.norm.weight"
    assert map_real_to_ours("hc_head_base") == "model.hc_head_base"
    assert map_real_to_ours("hc_head_scale") == "model.hc_head_scale"
    print("  name_mapping_top_level OK")


def test_name_mapping_layer_keys():
    assert map_real_to_ours("layers.0.attn.wq_a.weight") == "model.layers.0.attn.wq_a.weight"
    assert map_real_to_ours("layers.10.attn.compressor.wkv.weight") == \
        "model.layers.10.attn.compressor.wkv.weight"
    assert map_real_to_ours("layers.3.ffn.experts.42.w1.weight") == \
        "model.layers.3.ffn.experts.42.w1.weight"
    assert map_real_to_ours("layers.0.hc_attn_scale") == "model.layers.0.hc_attn_scale"
    print("  name_mapping_layer_keys OK")


def test_name_mapping_unrecognized_raises():
    try:
        map_real_to_ours("something_unknown.weight")
    except KeyError:
        print("  name_mapping_unrecognized_raises OK")
        return
    raise AssertionError("expected KeyError")


def main():
    print("FP4 tests")
    test_fp4_known_byte()
    test_fp4_sign_bit()
    test_fp4_per_block_scale()
    print("FP8 tests")
    test_fp8_passthrough_unit_scale()
    test_fp8_block_scale()
    print("Name-mapping tests")
    test_name_mapping_skips()
    test_name_mapping_top_level()
    test_name_mapping_layer_keys()
    test_name_mapping_unrecognized_raises()
    print("\nDEQUANT TESTS: PASSED")


if __name__ == "__main__":
    main()
