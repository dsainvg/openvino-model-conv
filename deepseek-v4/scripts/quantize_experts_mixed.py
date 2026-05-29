# -*- coding: utf-8 -*-
"""2.4 Step 2 -- Mixed-precision expert quantization (HOBBIT / MxMoE approach).

Reads the expert activation stats from collect_expert_stats.py, then quantizes
each expert IR to different precision levels based on hot/cold classification:

  - Hot experts  → INT4 (NNCF INT4_ASYM, group_size=32) — best quality
  - Cold experts → INT4 ultra-compressed (NNCF INT4_SYM, group_size=8) — smallest size
                   or MXFP4 (E2M1 + E8M0 microscale) when targeting CPU

The script processes the split expert IRs (from split_to_expert_irs.py or
load_real_v4_weights.py --per-expert-ir), producing a parallel directory of
quantized IRs that run_with_expert_offload.py can consume.

Output layout:
    <output_dir>/expert_L{i}_E{e}.xml   — quantized IR (hot=INT4, cold=INT4-tiny)
    <output_dir>/manifest.json           — maps each (layer, expert) to its precision

Usage:
    # Toy (validates pipeline):
    python scripts/quantize_experts_mixed.py \
        --stats expert_stats_toy.json \
        --ir-dir ov_ir_toy/expert_split \
        --output ov_ir_toy/expert_split_mixed

    # Real V4-Flash:
    python scripts/quantize_experts_mixed.py \
        --stats expert_stats_real.json \
        --ir-dir /path/to/v4_flash_expert_irs \
        --output /path/to/v4_flash_expert_mixed \
        --cold-mode int4-tiny
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def quantize_expert_ir(
    core, ir_path: Path, out_path: Path, mode: str
) -> dict:
    """Quantize a single expert IR.

    Modes:
      - "int4"      : NNCF INT4_ASYM, group_size=32 (high quality, ~4x compression)
      - "int4-tiny"  : NNCF INT4_SYM, group_size=8 (aggressive, ~5-6x compression)
      - "mxfp4"     : NNCF MXFP4, group_size=32 (CPU-optimized microscale)
      - "int8"      : NNCF INT8_ASYM (baseline, ~2x compression)
      - "copy"      : No quantization, just copy the IR as-is
    """
    import nncf
    import openvino as ov

    model = core.read_model(str(ir_path))

    # Get original size.
    bin_path = ir_path.with_suffix(".bin")
    orig_size = ir_path.stat().st_size + (bin_path.stat().st_size if bin_path.exists() else 0)

    if mode == "copy":
        ov.save_model(model, str(out_path), compress_to_fp16=False)
        out_bin = out_path.with_suffix(".bin")
        new_size = out_path.stat().st_size + (out_bin.stat().st_size if out_bin.exists() else 0)
        return {"mode": mode, "orig_bytes": orig_size, "new_bytes": new_size, "ratio": 1.0}

    if mode == "int4":
        compressed = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT4_ASYM,
            group_size=32,
            ratio=1.0,
        )
    elif mode == "int4-tiny":
        compressed = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT4_SYM,
            group_size=8,
            ratio=1.0,
        )
    elif mode == "mxfp4":
        compressed = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.MXFP4,
            group_size=32,
            ratio=1.0,
        )
    elif mode == "int8":
        compressed = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT8_ASYM,
        )
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(compressed, str(out_path), compress_to_fp16=False)

    out_bin = out_path.with_suffix(".bin")
    new_size = out_path.stat().st_size + (out_bin.stat().st_size if out_bin.exists() else 0)
    ratio = orig_size / max(new_size, 1)
    return {"mode": mode, "orig_bytes": orig_size, "new_bytes": new_size, "ratio": round(ratio, 2)}


def main():
    parser = argparse.ArgumentParser(
        description="Mixed-precision expert quantization based on activation stats."
    )
    parser.add_argument("--stats", type=Path, required=True,
                        help="Expert stats JSON from collect_expert_stats.py")
    parser.add_argument("--ir-dir", type=Path, required=True,
                        help="Directory with FP32 expert IRs (expert_L*_E*.xml)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for quantized expert IRs")
    parser.add_argument("--hot-mode", type=str, default="int4",
                        choices=["int4", "int8", "mxfp4", "copy"],
                        help="Quantization mode for hot experts (default: int4)")
    parser.add_argument("--cold-mode", type=str, default="int4-tiny",
                        choices=["int4-tiny", "int4", "mxfp4", "int8", "copy"],
                        help="Quantization mode for cold experts (default: int4-tiny)")
    parser.add_argument("--use-global", action="store_true",
                        help="Use global classification instead of per-layer")
    parser.add_argument("--copy-backbone", action="store_true",
                        help="Also copy backbone IRs (embed, pre_moe, post_moe, final) to output dir")
    args = parser.parse_args()

    # Load stats.
    with open(args.stats) as f:
        stats = json.load(f)

    meta = stats["meta"]
    L = meta["num_layers"]
    E = meta["num_experts"]
    print(f"=== Mixed-Precision Expert Quantization ===")
    print(f"  layers={L} experts={E} hot_mode={args.hot_mode} cold_mode={args.cold_mode}")

    # Build hot set.
    if args.use_global:
        global_hot = set(stats["global_classification"]["hot"])
        print(f"  using global classification: {len(global_hot)} hot experts")
    else:
        print(f"  using per-layer classification")

    import openvino as ov
    core = ov.Core()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"meta": meta, "hot_mode": args.hot_mode, "cold_mode": args.cold_mode, "experts": {}}

    total_orig = 0
    total_new = 0
    hot_count = 0
    cold_count = 0
    t0 = time.perf_counter()

    for i in range(L):
        if args.use_global:
            hot_set = global_hot
        else:
            hot_set = set(stats["classification"][str(i)]["hot"])

        for e in range(E):
            ir_path = args.ir_dir / f"expert_L{i}_E{e}.xml"
            if not ir_path.exists():
                print(f"  WARNING: {ir_path} not found, skipping")
                continue

            is_hot = e in hot_set
            mode = args.hot_mode if is_hot else args.cold_mode
            out_path = args.output / f"expert_L{i}_E{e}.xml"

            result = quantize_expert_ir(core, ir_path, out_path, mode)
            total_orig += result["orig_bytes"]
            total_new += result["new_bytes"]

            key = f"L{i}_E{e}"
            manifest["experts"][key] = {
                "layer": i, "expert": e, "hot": is_hot,
                "mode": mode, "ratio": result["ratio"],
            }
            if is_hot:
                hot_count += 1
            else:
                cold_count += 1

        elapsed = time.perf_counter() - t0
        print(f"  layer {i}: {E} experts quantized ({elapsed:.1f}s elapsed)")

    # Copy backbone IRs if requested.
    if args.copy_backbone:
        import shutil
        backbone_files = ["embed.xml", "embed.bin", "final.xml", "final.bin"]
        for i in range(L):
            backbone_files.extend([
                f"pre_moe_L{i}.xml", f"pre_moe_L{i}.bin",
                f"post_moe_L{i}.xml", f"post_moe_L{i}.bin",
            ])
        for fname in backbone_files:
            src = args.ir_dir / fname
            if src.exists():
                shutil.copy2(src, args.output / fname)
        print(f"  copied {len(backbone_files)} backbone files")

    # Save manifest.
    manifest["summary"] = {
        "hot_count": hot_count,
        "cold_count": cold_count,
        "total_orig_mb": round(total_orig / (1024 * 1024), 2),
        "total_new_mb": round(total_new / (1024 * 1024), 2),
        "overall_ratio": round(total_orig / max(total_new, 1), 2),
    }
    manifest_path = args.output / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\n=== Summary ===")
    print(f"  hot experts:  {hot_count} @ {args.hot_mode}")
    print(f"  cold experts: {cold_count} @ {args.cold_mode}")
    print(f"  size: {manifest['summary']['total_orig_mb']:.1f} MB -> "
          f"{manifest['summary']['total_new_mb']:.1f} MB "
          f"({manifest['summary']['overall_ratio']:.1f}x)")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  manifest: {manifest_path}")
    print(f"\nMIXED-PRECISION QUANTIZATION: PASSED")


if __name__ == "__main__":
    main()
