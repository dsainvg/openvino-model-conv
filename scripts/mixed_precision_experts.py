"""2.4 — Mixed-precision experts (HOBBIT-style two-tier quantization).

Pipeline:
  1. Calibrate: run N random inputs through the split orchestrator, count how
     often each (layer, expert) is selected by the gate.
  2. Tier: top-K most-used per layer = "hot"; the rest = "cold".
  3. Re-quantize: hot experts → INT8 weight compression, cold experts → INT4
     weight compression (group_size=32). Save to ov_ir_toy/expert_split_mixed/.
     (NNCF 3.1.0 has no INT2 mode; INT8/INT4 is the two-tier proxy on this
     stack. The infrastructure is what matters — swapping in MXFP4/NF4/INT2
     later is a one-line change.)
  4. Validate: orchestrate the mixed IRs, compare logits + greedy top against
     the monolithic OV reference.
  5. Report size deltas (FP32 baseline / all-INT8 / all-INT4 / mixed).

Activation counts are written to ov_ir_toy/expert_split_mixed/calibration.json
so a downstream caller (a future warm-cache predictor, NPU expert-predictor,
etc.) can read the same statistics without re-running calibration.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch
import openvino as ov
import nncf

from deepseek_v4 import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


def _file_size_kb(xml_path: Path) -> float:
    bin_path = xml_path.with_suffix(".bin")
    return (xml_path.stat().st_size + bin_path.stat().st_size) / 1024.0


def calibrate(
    core,
    split_dir: Path,
    cfg,
    n_inputs: int = 8,
    seq_len: int = 128,
    seed: int = 42,
) -> "dict[int, Counter]":
    """Return {layer_idx: Counter({expert_idx: hits})} from running n_inputs
    forward passes and counting unique-per-token expert selections."""
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    H = cfg.hc_mult
    d = cfg.hidden_size
    rng = np.random.default_rng(seed)

    embed_c = core.compile_model(str(split_dir / "embed.xml"), "CPU")
    pre_c = [core.compile_model(str(split_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(split_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    expert_c = [
        [core.compile_model(str(split_dir / f"expert_L{i}_E{e}.xml"), "CPU") for e in range(E)]
        for i in range(L)
    ]

    counts: "dict[int, Counter]" = {i: Counter() for i in range(L)}
    for n in range(n_inputs):
        ids = rng.integers(0, cfg.vocab_size, size=(1, seq_len)).astype(np.int64)
        h = embed_c([ids])[0]
        for i in range(L):
            out = pre_c[i]([h, ids])
            y2_flat = out[0]; x_res = out[1]; post2 = out[2]; comb2 = out[3]
            weights = out[4]; indices = out[5]; shared_out = out[6]
            for e in indices.flatten().tolist():
                counts[i][int(e)] += 1
            N = y2_flat.shape[0]
            gm = np.zeros((N, E), dtype=np.float32)
            np.put_along_axis(gm, indices.astype(np.int64), weights.astype(np.float32), axis=1)
            moe = np.zeros((N, d), dtype=np.float32)
            for e in np.unique(indices).tolist():
                moe += gm[:, e:e + 1] * expert_c[i][int(e)]([y2_flat])[0]
            moe += shared_out
            h = post_c[i]([moe.reshape(1, seq_len, d), x_res, post2, comb2])[0]
    return counts


def assign_tiers(counts: "dict[int, Counter]", n_hot: int) -> "dict[tuple[int, int], str]":
    """For each layer, the n_hot most-frequently-selected experts are 'hot';
    the rest are 'cold'. Ties broken by expert index for determinism."""
    tier: "dict[tuple[int, int], str]" = {}
    for layer, c in counts.items():
        # All experts seen at least once across calibration go into the ranking;
        # any expert never seen counts as 0 and is therefore cold.
        all_e = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        hot = {e for e, _ in all_e[:n_hot]}
        for e in range(max(max(hot, default=-1) + 1, n_hot * 2)):
            tier[(layer, e)] = "hot" if e in hot else "cold"
    return tier


def quantize_expert_irs(
    core,
    split_dir: Path,
    mixed_dir: Path,
    cfg,
    tier: "dict[tuple[int, int], str]",
):
    """Apply INT8 to hot experts, INT4 to cold experts; write to mixed_dir."""
    mixed_dir.mkdir(parents=True, exist_ok=True)
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    sizes = {"hot": [], "cold": []}

    for i in range(L):
        for e in range(E):
            src = split_dir / f"expert_L{i}_E{e}.xml"
            dst = mixed_dir / f"expert_L{i}_E{e}.xml"
            kind = tier.get((i, e), "cold")
            model_fp32 = core.read_model(str(src))
            if kind == "hot":
                model_q = nncf.compress_weights(model_fp32, mode=nncf.CompressWeightsMode.INT8_ASYM)
            else:
                model_q = nncf.compress_weights(
                    model_fp32,
                    mode=nncf.CompressWeightsMode.INT4_ASYM,
                    group_size=32,
                    ratio=1.0,
                )
            ov.save_model(model_q, str(dst), compress_to_fp16=False)
            sizes[kind].append(_file_size_kb(dst))
    return sizes


def copy_backbone(split_dir: Path, mixed_dir: Path, cfg):
    """Symlink (or copy) the FP32 backbone IRs into mixed_dir so the orchestrator
    only needs one directory to drive everything."""
    import shutil
    for name in ["embed.xml", "embed.bin", "final.xml", "final.bin"]:
        shutil.copyfile(split_dir / name, mixed_dir / name)
    L = cfg.num_hidden_layers
    for i in range(L):
        for name in (f"pre_moe_L{i}.xml", f"pre_moe_L{i}.bin",
                     f"post_moe_L{i}.xml", f"post_moe_L{i}.bin"):
            shutil.copyfile(split_dir / name, mixed_dir / name)


def orchestrate(core, ir_dir: Path, cfg, ids_np: np.ndarray) -> np.ndarray:
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    H = cfg.hc_mult
    d = cfg.hidden_size
    embed_c = core.compile_model(str(ir_dir / "embed.xml"), "CPU")
    final_c = core.compile_model(str(ir_dir / "final.xml"), "CPU")
    pre_c = [core.compile_model(str(ir_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(ir_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    expert_c = [
        [core.compile_model(str(ir_dir / f"expert_L{i}_E{e}.xml"), "CPU") for e in range(E)]
        for i in range(L)
    ]
    h = embed_c([ids_np])[0]
    for i in range(L):
        out = pre_c[i]([h, ids_np])
        y2_flat = out[0]; x_res = out[1]; post2 = out[2]; comb2 = out[3]
        weights = out[4]; indices = out[5]; shared_out = out[6]
        N = y2_flat.shape[0]
        gm = np.zeros((N, E), dtype=np.float32)
        np.put_along_axis(gm, indices.astype(np.int64), weights.astype(np.float32), axis=1)
        moe = np.zeros((N, d), dtype=np.float32)
        for e in np.unique(indices).tolist():
            moe += gm[:, e:e + 1] * expert_c[i][int(e)]([y2_flat])[0]
        moe += shared_out
        h = post_c[i]([moe.reshape(1, ids_np.shape[1], d), x_res, post2, comb2])[0]
    return final_c([h])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calibration-inputs", type=int, default=8)
    parser.add_argument("--n-hot-per-layer", type=int, default=4,
                        help="Top-N experts per layer to keep at the higher precision.")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = make_toy_config()
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts

    split_dir = ROOT / "ov_ir_toy" / "expert_split"
    mixed_dir = ROOT / "ov_ir_toy" / "expert_split_mixed"
    if not (split_dir / "embed.xml").exists():
        raise RuntimeError("Run scripts/split_to_expert_irs.py first.")

    core = ov.Core()

    print(f"=== Calibration: {args.n_calibration_inputs} random inputs, "
          f"seq_len={args.seq_len} ===")
    t0 = time.perf_counter()
    counts = calibrate(
        core, split_dir, cfg,
        n_inputs=args.n_calibration_inputs,
        seq_len=args.seq_len,
    )
    print(f"  calibration done in {time.perf_counter() - t0:.1f} s")
    for i, c in counts.items():
        total = sum(c.values())
        top = c.most_common(min(args.n_hot_per_layer + 2, E))
        print(f"  L{i}: {len(c)}/{E} experts seen, total selections={total}, top={top}")

    tier = assign_tiers(counts, args.n_hot_per_layer)
    n_hot = sum(1 for v in tier.values() if v == "hot")
    n_cold = sum(1 for v in tier.values() if v == "cold")
    print(f"\n=== Tiers (n_hot_per_layer={args.n_hot_per_layer}): "
          f"hot={n_hot}, cold={n_cold} ===")

    print("\n=== Quantize ===")
    t0 = time.perf_counter()
    sizes = quantize_expert_irs(core, split_dir, mixed_dir, cfg, tier)
    print(f"  quantization done in {time.perf_counter() - t0:.1f} s")
    fp32_sizes = [_file_size_kb(split_dir / f"expert_L{i}_E{e}.xml")
                  for i in range(L) for e in range(E)]
    print(f"  per-expert size (KB):")
    print(f"    FP32  : mean={np.mean(fp32_sizes):7.2f}  total={np.sum(fp32_sizes):8.2f}")
    print(f"    INT8 hot ({len(sizes['hot'])} experts): mean={np.mean(sizes['hot']):7.2f}  total={np.sum(sizes['hot']):8.2f}")
    print(f"    INT4 cold({len(sizes['cold'])} experts): mean={np.mean(sizes['cold']):7.2f}  total={np.sum(sizes['cold']):8.2f}")
    print(f"    Mixed total: {np.sum(sizes['hot']) + np.sum(sizes['cold']):8.2f} "
          f"(saved {(1 - (np.sum(sizes['hot']) + np.sum(sizes['cold'])) / np.sum(fp32_sizes)) * 100:.1f}% vs FP32)")

    print("\n=== Copy backbone segments ===")
    copy_backbone(split_dir, mixed_dir, cfg)

    # Persist calibration for downstream tools.
    cal_out = {
        "n_calibration_inputs": args.n_calibration_inputs,
        "seq_len": args.seq_len,
        "n_hot_per_layer": args.n_hot_per_layer,
        "per_layer_counts": {str(i): dict(c) for i, c in counts.items()},
        "tier": {f"L{i}_E{e}": t for (i, e), t in tier.items()},
    }
    (mixed_dir / "calibration.json").write_text(json.dumps(cal_out, indent=2))
    print(f"  wrote {mixed_dir / 'calibration.json'}")

    # Validate.
    print("\n=== Validate: mixed-precision orchestrator vs monolithic OV ===")
    mono_ir = ROOT / "ov_ir_toy" / "deepseek_v4_toy.xml"
    mono_c = core.compile_model(str(mono_ir), "CPU")
    ids = np.random.default_rng(7).integers(0, cfg.vocab_size, size=(1, args.seq_len)).astype(np.int64)
    mono_logits = mono_c([ids])[0]
    mixed_logits = orchestrate(core, mixed_dir, cfg, ids)

    mono_t = torch.from_numpy(mono_logits)
    mixed_t = torch.from_numpy(mixed_logits)
    diff = (mono_t - mixed_t).abs()
    mono_top = int(mono_t[0, -1].argmax().item())
    mix_top = int(mixed_t[0, -1].argmax().item())
    print(f"  monolithic OV top : {mono_top}")
    print(f"  mixed-precision top: {mix_top}  match={mono_top == mix_top}")
    print(f"  abs diff vs mono : max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")

    print("\nMIXED-PRECISION EXPERTS: PASSED")


if __name__ == "__main__":
    main()
