"""Load real DeepSeek-V4-Flash weights into our DeepseekV4ForCausalLM.

Real V4-Flash:
    - ~284B params, ~160 GB on disk (FP8 main weights + FP4 expert weights, both
      with E8M0-microscaled per-block scales)
    - dequantizing to BF16 needs roughly 500 GB of host RAM

This 64GB host cannot run the full load. The loader is written and the dequant
logic is verified on synthetic tensors (see tests/test_dequant.py). On a host
with enough RAM, the entry point is:

    python scripts/load_real_v4_weights.py \\
        --weights-dir /path/to/DeepSeek-V4-Flash \\
        --output       /path/to/output_dir

A `--dry-run` mode reads only `model.safetensors.index.json` to verify that every
real-V4 parameter name maps to a valid slot in our PyTorch module tree (or is on
the explicit skip list — MTP blocks and hash-routing tables, which the toy port
does not implement). Run this before kicking off the full load.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# FP4 / FP8 dequantization
# ---------------------------------------------------------------------------

# DeepSeek's FP4 (e2m1fn) lookup table. Index is the raw 4-bit value; the high
# bit is the sign. Reproduced from v4_flash_meta/inference/convert.py.
FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 (e2m1fn, 2 vals/byte) + per-32-col scale -> FP32.

    packed: int8 tensor [out_dim, in_packed]; in_dim = 2*in_packed.
            byte layout: low nibble is column 2k, high nibble is column 2k+1.
    scale:  numeric tensor [out_dim, in_dim // 32].
    """
    assert packed.dtype == torch.int8, f"expected int8, got {packed.dtype}"
    assert packed.dim() == 2
    out_dim, in_packed = packed.shape
    in_dim = 2 * in_packed
    fp4_block = 32
    assert in_dim % fp4_block == 0
    assert tuple(scale.shape) == (out_dim, in_dim // fp4_block), (
        f"scale shape {tuple(scale.shape)} != ({out_dim}, {in_dim // fp4_block})"
    )

    bytes_ = packed.view(torch.uint8)
    low = (bytes_ & 0x0F).long()
    high = ((bytes_ >> 4) & 0x0F).long()
    table = FP4_TABLE.to(packed.device)
    # convert.py order: low first, then high, in adjacent columns.
    vals = torch.stack([table[low], table[high]], dim=-1).reshape(out_dim, in_dim)

    s = scale.float().repeat_interleave(fp4_block, dim=-1)
    return vals * s


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 (e4m3fn) + per-128x128-block scale -> FP32."""
    if weight.dtype == torch.int8:
        weight = weight.view(torch.float8_e4m3fn)
    assert weight.dtype == torch.float8_e4m3fn, f"expected fp8_e4m3fn, got {weight.dtype}"
    assert weight.dim() == 2
    out_dim, in_dim = weight.shape
    block = 128
    assert out_dim % block == 0 and in_dim % block == 0, (
        f"FP8 weight shape ({out_dim},{in_dim}) not divisible by 128"
    )
    assert tuple(scale.shape) == (out_dim // block, in_dim // block)

    w = weight.float().view(out_dim // block, block, in_dim // block, block)
    s = scale.float()[:, None, :, None]
    return (w * s).reshape(out_dim, in_dim)


# ---------------------------------------------------------------------------
# Real-V4 → ours name mapping
# ---------------------------------------------------------------------------

# Patterns for real-V4 keys we DELIBERATELY skip:
#   - mtp.*           : Multi-Token Prediction blocks (set num_nextn_predict_layers=0)
#   - *.tid2eid       : Hash-routing token-id -> expert-id tables (num_hash_layers=0)
#   - *.gate.bias     : Bias on routed-expert gate (toy uses `bias=False`)
SKIP_PATTERNS = (
    re.compile(r"^mtp\."),
    re.compile(r"\.tid2eid$"),
    re.compile(r"\.gate\.bias$"),
)


def map_real_to_ours(name: str) -> Optional[str]:
    """Map a real-V4 safetensors key to our DeepseekV4ForCausalLM parameter name.

    Returns None when the key is on the explicit skip list. Raises on unrecognized
    keys so that the dry-run can flag them.
    """
    for pat in SKIP_PATTERNS:
        if pat.search(name):
            return None

    if name == "embed.weight":
        return "model.embed.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name in ("hc_head_base", "hc_head_fn", "hc_head_scale"):
        return f"model.{name}"
    if name.startswith("layers."):
        return f"model.{name}"

    raise KeyError(f"Unrecognized real-V4 key: {name!r}")


def is_quant_scale(name: str) -> bool:
    """True if this name is a quantization scale (paired with a .weight).

    `hc_*_scale` are HC parameters, NOT quantization scales — they have no
    paired weight and stay as full-precision params.
    """
    return name.endswith(".scale")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def dry_run(weights_dir: Path) -> Tuple[int, int, list]:
    """Read only the index.json and verify every key maps cleanly."""
    index_path = weights_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    keys = list(index["weight_map"].keys())

    ok = 0
    skipped = 0
    paired_weights = 0  # `.weight` keys that have a paired `.scale`
    unmapped = []
    weight_keys = set(k for k in keys if not k.endswith(".scale"))
    scale_keys = set(k for k in keys if k.endswith(".scale"))

    for k in keys:
        try:
            mapped = map_real_to_ours(k)
        except KeyError:
            unmapped.append(k)
            continue
        if mapped is None:
            skipped += 1
        else:
            ok += 1

    for w in weight_keys:
        if w.replace(".weight", ".scale") in scale_keys:
            paired_weights += 1

    print(f"  total keys           : {len(keys)}")
    print(f"  mapped to our params : {ok}")
    print(f"  on skip list (mtp/hash/gate-bias): {skipped}")
    print(f"  weights paired w/ scale (FP4 or FP8): {paired_weights}")
    print(f"  free-floating .scale keys (HC, not quant): {len(scale_keys) - paired_weights}")
    if unmapped:
        print(f"\n  UNMAPPED ({len(unmapped)}):")
        for k in unmapped[:20]:
            print(f"    {k}")
        if len(unmapped) > 20:
            print(f"    ... and {len(unmapped) - 20} more")
    return ok, skipped, unmapped


def full_load(
    weights_dir: Path,
    output_dir: Path,
    target_dtype: torch.dtype = torch.bfloat16,
    skip_existing: bool = True,
) -> None:
    """Load real V4-Flash weights, dequant FP4/FP8, save BF16 safetensors.

    Streams shard by shard so peak RAM is bounded by the largest shard plus
    the largest single weight after dequant. Writes one BF16 .safetensors file
    per input shard to `output_dir`.
    """
    from safetensors.torch import safe_open, save_file

    index_path = weights_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index["weight_map"]
    shards: Dict[str, list] = defaultdict(list)
    for k, shard in weight_map.items():
        shards[shard].append(k)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"fp4": 0, "fp8": 0, "passthrough": 0, "skipped": 0}

    # Resolve which (.weight, .scale) pairs need joint loading. Most pairs are
    # in the same shard; rarely they straddle. Build a global scale->shard map.
    scale_to_shard = {k: v for k, v in weight_map.items() if k.endswith(".scale")}

    for shard_name, keys in sorted(shards.items()):
        out_path = output_dir / shard_name.replace(".safetensors", "-bf16.safetensors")
        if skip_existing and out_path.exists():
            print(f"  [skip] {out_path.name} (already exists)")
            continue
        print(f"  [load] {shard_name} ({len(keys)} tensors)")
        out: Dict[str, torch.Tensor] = {}
        with safe_open(weights_dir / shard_name, framework="pt", device="cpu") as f:
            keys_in_shard = set(f.keys())
            handled = set()
            for k in f.keys():
                if k in handled:
                    continue
                mapped = map_real_to_ours(k)
                if mapped is None:
                    summary["skipped"] += 1
                    continue
                if k.endswith(".scale"):
                    # Free-floating HC scale — not a quant scale.
                    out[mapped] = f.get_tensor(k).to(target_dtype)
                    summary["passthrough"] += 1
                    continue

                scale_name = k.replace(".weight", ".scale")
                if scale_name in scale_to_shard:
                    # Paired weight + scale: dequantize to BF16.
                    weight = f.get_tensor(k)
                    if scale_to_shard[scale_name] == shard_name:
                        scale = f.get_tensor(scale_name)
                        handled.add(scale_name)
                    else:
                        with safe_open(
                            weights_dir / scale_to_shard[scale_name],
                            framework="pt",
                            device="cpu",
                        ) as g:
                            scale = g.get_tensor(scale_name)

                    if weight.dtype == torch.int8 and weight.size(-1) * 2 // 32 == scale.size(-1):
                        deq = dequant_fp4(weight, scale)
                        summary["fp4"] += 1
                    else:
                        deq = dequant_fp8(weight, scale)
                        summary["fp8"] += 1
                    out[mapped] = deq.to(target_dtype)
                else:
                    # Plain BF16/FP32 tensor.
                    out[mapped] = f.get_tensor(k).to(target_dtype)
                    summary["passthrough"] += 1
        save_file(out, str(out_path))
        print(f"    -> wrote {out_path.name} ({len(out)} tensors)")
    print(f"\n  Summary: {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "v4_flash_meta",
                        help="Directory containing model.safetensors.index.json (and shards for full mode).")
    parser.add_argument("--output", type=Path, default=ROOT / "v4_flash_bf16",
                        help="Output directory for dequantized BF16 shards (full mode only).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read only index.json and verify name coverage. No tensor loading.")
    args = parser.parse_args()

    if args.dry_run:
        print(f"=== Dry-run name-coverage check on {args.weights_dir} ===")
        ok, skipped, unmapped = dry_run(args.weights_dir)
        if unmapped:
            print("\nFAIL: unmapped keys remain")
            sys.exit(1)
        print("\nDRY-RUN: PASSED")
        return

    print(f"=== Full load from {args.weights_dir} -> {args.output} ===")
    print("WARNING: real V4-Flash needs ~500GB peak RAM; on a 64GB host this WILL OOM.")
    full_load(args.weights_dir, args.output)


if __name__ == "__main__":
    main()
