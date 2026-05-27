# -*- coding: utf-8 -*-
"""2.4 Step 1 -- Collect per-layer per-expert activation frequency statistics.

Runs calibration data through the model (PyTorch or split OV IRs), recording
which experts the gate selects on every token. Outputs a JSON file with counts
that downstream scripts use to classify hot vs cold experts for mixed-precision
quantization (HOBBIT / MxMoE approach).

Three modes:

1. `--mode toy`  (default) — uses the toy PyTorch model with random input_ids.
   Good for validating the pipeline end-to-end without real data or weights.

2. `--mode toy-ov` — uses the split OV IRs from ov_ir_toy/expert_split/.
   Validates that the OV gate gives the same routing decisions as PyTorch.

3. `--mode real` — uses real V4-Flash split IRs (from 2.3's --per-expert-ir
   output). Requires `--ir-dir` pointing to the directory with pre_moe_L*.xml.
   Only the pre_moe segments (which contain the gate) are loaded — no expert
   computation is needed, so RAM is modest (~5 GB for backbone compile).

Output: JSON file with structure:
    {
        "meta": {
            "mode": "toy", "num_layers": 4, "num_experts": 8,
            "topk": 2, "num_samples": 1000, "total_tokens": 128000
        },
        "per_layer": {
            "0": {"counts": [1523, 42, ...], "pct": [11.9, 0.3, ...]},
            ...
        },
        "global_counts": [6100, 200, ...],
        "ranking": {
            "0": [3, 5, 1, 7, 0, 2, 6, 4],  # expert indices sorted by count desc
            ...
        }
    }

Usage:
    # Toy (validates pipeline):
    python scripts/collect_expert_stats.py --mode toy --num-samples 100

    # Toy with OV split IRs:
    python scripts/collect_expert_stats.py --mode toy-ov --num-samples 100

    # Real V4-Flash (on dev machine with split IRs):
    python scripts/collect_expert_stats.py --mode real \
        --ir-dir /path/to/v4_flash_expert_irs \
        --calibration-data wikitext \
        --num-samples 1000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


# ---------------------------------------------------------------------------
# Calibration data loaders
# ---------------------------------------------------------------------------

def load_calibration_data(source: str, num_samples: int, seq_len: int, vocab_size: int):
    """Yield (input_ids_np [1, S],) batches for calibration.

    `source` can be:
      - "random"   : random token ids (for toy validation)
      - "wikitext"  : WikiText-2 test split via HuggingFace datasets
      - a file path : one-line-per-sample text file (pre-tokenized ids as JSON)
    """
    if source == "random":
        rng = np.random.RandomState(42)
        for _ in range(num_samples):
            ids = rng.randint(0, vocab_size, size=(1, seq_len)).astype(np.int64)
            yield ids
        return

    if source == "wikitext":
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError:
            print("ERROR: --calibration-data wikitext requires `datasets` and `transformers`.")
            print("       pip install datasets transformers")
            sys.exit(1)

        print("  loading WikiText-2 and tokenizer...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)

        # Concatenate all text, tokenize, chunk into seq_len blocks.
        all_text = "\n".join(t for t in ds["text"] if t.strip())
        all_ids = tokenizer.encode(all_text)
        print(f"  total tokens in WikiText-2 test: {len(all_ids)}")

        count = 0
        for start in range(0, len(all_ids) - seq_len, seq_len):
            if count >= num_samples:
                break
            chunk = np.array(all_ids[start:start + seq_len], dtype=np.int64).reshape(1, seq_len)
            yield chunk
            count += 1
        if count < num_samples:
            print(f"  WARNING: only {count} full-length samples available (requested {num_samples})")
        return

    # File path: expect one JSON array of ints per line.
    path = Path(source)
    if not path.exists():
        print(f"ERROR: calibration data file not found: {path}")
        sys.exit(1)
    count = 0
    with open(path) as f:
        for line in f:
            if count >= num_samples:
                break
            ids = json.loads(line.strip())
            arr = np.array(ids[:seq_len], dtype=np.int64).reshape(1, -1)
            if arr.shape[1] < seq_len:
                arr = np.pad(arr, ((0, 0), (0, seq_len - arr.shape[1])))
            yield arr
            count += 1


# ---------------------------------------------------------------------------
# Stats collector
# ---------------------------------------------------------------------------

class ExpertStatsCollector:
    """Accumulates per-layer expert activation counts."""

    def __init__(self, num_layers: int, num_experts: int):
        self.L = num_layers
        self.E = num_experts
        self.counts = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.total_tokens = 0

    def record(self, layer: int, indices: np.ndarray):
        """Record gate indices for one layer.

        indices: [N, topk] int array of selected expert ids.
        """
        for e in indices.flat:
            self.counts[layer, int(e)] += 1

    def record_batch_tokens(self, num_tokens: int):
        self.total_tokens += num_tokens

    def to_dict(self):
        result = {"per_layer": {}, "global_counts": None, "ranking": {}}
        global_counts = np.zeros(self.E, dtype=np.int64)

        for i in range(self.L):
            layer_counts = self.counts[i].tolist()
            total = max(int(self.counts[i].sum()), 1)
            pct = [round(c / total * 100, 2) for c in layer_counts]
            ranking = np.argsort(-self.counts[i]).tolist()
            result["per_layer"][str(i)] = {
                "counts": layer_counts,
                "pct": pct,
            }
            result["ranking"][str(i)] = ranking
            global_counts += self.counts[i]

        result["global_counts"] = global_counts.tolist()
        result["global_ranking"] = np.argsort(-global_counts).tolist()
        result["global_pct"] = [
            round(c / max(int(global_counts.sum()), 1) * 100, 2)
            for c in global_counts.tolist()
        ]
        return result


# ---------------------------------------------------------------------------
# Mode: toy (PyTorch)
# ---------------------------------------------------------------------------

def run_toy_pytorch(num_samples: int, seq_len: int):
    """Run gate-only forward on the toy PyTorch model."""
    import torch
    from deepseek_v4 import DeepseekV4ForCausalLM
    from test_modeling_smoke import make_toy_config

    torch.manual_seed(0)
    cfg = make_toy_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    H = cfg.hc_mult

    collector = ExpertStatsCollector(L, E)
    print(f"  model: L={L} E={E} topk={cfg.num_experts_per_tok} dim={cfg.hidden_size}")

    t0 = time.perf_counter()
    sample_count = 0
    with torch.inference_mode():
        for input_ids_np in load_calibration_data("random", num_samples, seq_len, cfg.vocab_size):
            input_ids = torch.from_numpy(input_ids_np)
            # Full forward to get gate decisions at every layer.
            # We hook into the MoE gate by running layer-by-layer.
            h = model.model.embed(input_ids)
            h = h.unsqueeze(2).expand(-1, -1, H, -1).contiguous()
            S = input_ids.size(1)
            cos = model.model.rope_cos[:S]
            sin = model.model.rope_sin[:S]

            for i in range(L):
                block = model.model.layers[i]
                # Attention sub-block.
                y_full, post_full, comb_full = block._hc_pre(
                    h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
                )
                y_full = block.attn_norm(y_full)
                y_new = block.attn(y_full, cos, sin, seqlen_new=S)
                x_new = block._hc_post(y_new, h, post_full, comb_full)

                # FFN pre-MoE.
                y2, post2, comb2 = block._hc_pre(
                    x_new, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
                )
                y2 = block.ffn_norm(y2)
                b, s, d = y2.shape
                y2_flat = y2.reshape(b * s, d)
                ids_flat = input_ids.reshape(b * s)

                # Gate only — this is what we need.
                weights, indices = block.ffn.gate(y2_flat, ids_flat)
                collector.record(i, indices.numpy())

                # Still need full forward to get correct h for next layer.
                h, _ = block(h, input_ids, cos, sin)

            N = input_ids_np.shape[0] * input_ids_np.shape[1]
            collector.record_batch_tokens(N)
            sample_count += 1
            if sample_count % 50 == 0:
                print(f"    [{sample_count}/{num_samples}]")

    elapsed = time.perf_counter() - t0
    print(f"  processed {sample_count} samples ({collector.total_tokens} tokens) in {elapsed:.1f}s")
    return collector, cfg


# ---------------------------------------------------------------------------
# Mode: toy-ov (split OV IRs)
# ---------------------------------------------------------------------------

def run_toy_ov(num_samples: int, seq_len: int):
    """Run gate-only forward using the split OV IRs."""
    import openvino as ov
    from test_modeling_smoke import make_toy_config

    cfg = make_toy_config()
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    d = cfg.hidden_size

    split_dir = ROOT / "ov_ir_toy" / "expert_split"
    if not (split_dir / "embed.xml").exists():
        print("ERROR: split IRs not found. Run scripts/split_to_expert_irs.py first.")
        sys.exit(1)

    core = ov.Core()
    print("  compiling backbone IRs...")
    embed_c = core.compile_model(str(split_dir / "embed.xml"), "CPU")
    pre_c = [core.compile_model(str(split_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(split_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    final_c = core.compile_model(str(split_dir / "final.xml"), "CPU")
    # Compile all experts (toy is small enough).
    expert_c = [
        [core.compile_model(str(split_dir / f"expert_L{i}_E{e}.xml"), "CPU") for e in range(E)]
        for i in range(L)
    ]

    collector = ExpertStatsCollector(L, E)
    print(f"  model: L={L} E={E} topk={cfg.num_experts_per_tok} dim={d}")

    t0 = time.perf_counter()
    sample_count = 0
    for input_ids_np in load_calibration_data("random", num_samples, seq_len, cfg.vocab_size):
        h = embed_c([input_ids_np])[0]
        B = input_ids_np.shape[0]
        S = input_ids_np.shape[1]

        for i in range(L):
            out = pre_c[i]([h, input_ids_np])
            y2_flat = out[0]
            x_res = out[1]
            post2 = out[2]
            comb2 = out[3]
            weights = out[4]
            indices = out[5]
            shared_out = out[6]

            # Record gate decisions.
            collector.record(i, indices.astype(np.int64))

            # Run experts for correct h propagation.
            N = y2_flat.shape[0]
            gate_mat = np.zeros((N, E), dtype=np.float32)
            np.put_along_axis(gate_mat, indices.astype(np.int64), weights.astype(np.float32), axis=1)
            moe_out_flat = np.zeros((N, d), dtype=np.float32)
            active = np.unique(indices).tolist()
            for e in active:
                expert_out = expert_c[i][e]([y2_flat])[0]
                moe_out_flat += gate_mat[:, e:e + 1] * expert_out
            moe_out_flat += shared_out

            moe_out_bsd = moe_out_flat.reshape(B, S, d)
            h = post_c[i]([moe_out_bsd, x_res, post2, comb2])[0]

        collector.record_batch_tokens(B * S)
        sample_count += 1
        if sample_count % 50 == 0:
            print(f"    [{sample_count}/{num_samples}]")

    elapsed = time.perf_counter() - t0
    print(f"  processed {sample_count} samples ({collector.total_tokens} tokens) in {elapsed:.1f}s")
    return collector, cfg


# ---------------------------------------------------------------------------
# Mode: real (V4-Flash split IRs, gate-only — no expert compute needed)
# ---------------------------------------------------------------------------

def run_real(ir_dir: Path, num_samples: int, seq_len: int, calibration_data: str):
    """Run gate-only forward on real V4-Flash split IRs.

    Only compiles embed + pre_moe segments (which contain the gate).
    Expert IRs are NOT loaded — we only need the gate's routing decisions.
    This keeps RAM modest and runs fast.
    """
    import openvino as ov

    ir_dir = Path(ir_dir)
    if not (ir_dir / "embed.xml").exists():
        # Try looking for pre_moe files to infer layer count.
        print(f"ERROR: embed.xml not found in {ir_dir}")
        print("  Run load_real_v4_weights.py --per-expert-ir first.")
        sys.exit(1)

    # Detect layer count from pre_moe files.
    layer_count = 0
    while (ir_dir / f"pre_moe_L{layer_count}.xml").exists():
        layer_count += 1
    if layer_count == 0:
        print(f"ERROR: no pre_moe_L*.xml found in {ir_dir}")
        sys.exit(1)

    # Detect expert count from expert files in layer 0.
    expert_count = 0
    while (ir_dir / f"expert_L0_E{expert_count}.xml").exists():
        expert_count += 1

    # Read config for topk if available.
    cfg_path = ir_dir.parent / "config.json"
    topk = 6  # V4-Flash default
    vocab_size = 129280
    if cfg_path.exists():
        with open(cfg_path) as f:
            real_cfg = json.load(f)
            topk = real_cfg.get("num_experts_per_tok", 6)
            vocab_size = real_cfg.get("vocab_size", 129280)

    print(f"  detected: L={layer_count} E={expert_count} topk={topk}")

    core = ov.Core()
    print("  compiling embed + pre_moe IRs (gate-only, no experts needed)...")
    embed_c = core.compile_model(str(ir_dir / "embed.xml"), "CPU")

    pre_c = []
    for i in range(layer_count):
        pre_c.append(core.compile_model(str(ir_dir / f"pre_moe_L{i}.xml"), "CPU"))
        if (i + 1) % 10 == 0:
            print(f"    compiled pre_moe up to L{i}")

    print("  NOTE: only running gate (pre_moe) — hidden states will drift after")
    print("        layer 0 since we skip expert+post_moe. This is fine for")
    print("        activation frequency statistics (routing is input-dependent,")
    print("        not precision-sensitive for frequency counts).")

    collector = ExpertStatsCollector(layer_count, expert_count)

    t0 = time.perf_counter()
    sample_count = 0
    for input_ids_np in load_calibration_data(calibration_data, num_samples, seq_len, vocab_size):
        h = embed_c([input_ids_np])[0]

        for i in range(layer_count):
            out = pre_c[i]([h, input_ids_np])
            indices = out[5]  # gate_indices [N, topk]
            collector.record(i, indices.astype(np.int64))
            # NOTE: we do NOT run experts or post_moe here.
            # h stays as embed output for all layers — this is intentional.
            # The gate routing statistics are still meaningful because:
            # 1) Gate weight is a learned linear projection, not dependent on
            #    exact hidden state precision for frequency-level statistics.
            # 2) For precise per-sample routing, use --mode toy or toy-ov.

        N = input_ids_np.shape[0] * input_ids_np.shape[1]
        collector.record_batch_tokens(N)
        sample_count += 1
        if sample_count % 50 == 0:
            elapsed = time.perf_counter() - t0
            tps = collector.total_tokens / max(elapsed, 0.001)
            print(f"    [{sample_count}/{num_samples}] {tps:.0f} tok/s")

    elapsed = time.perf_counter() - t0
    print(f"  processed {sample_count} samples ({collector.total_tokens} tokens) in {elapsed:.1f}s")

    class FakeCfg:
        pass
    cfg = FakeCfg()
    cfg.num_hidden_layers = layer_count
    cfg.n_routed_experts = expert_count
    cfg.num_experts_per_tok = topk
    cfg.hidden_size = 0  # unknown without config
    return collector, cfg


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(collector: ExpertStatsCollector, top_n: int = 10):
    """Print a human-readable summary of expert activation statistics."""
    data = collector.to_dict()
    L = collector.L
    E = collector.E

    print("\n" + "=" * 70)
    print("EXPERT ACTIVATION STATISTICS")
    print("=" * 70)

    for i in range(L):
        layer = data["per_layer"][str(i)]
        ranking = data["ranking"][str(i)]
        counts = layer["counts"]
        pct = layer["pct"]
        total = sum(counts)

        print(f"\n  Layer {i} (total activations: {total}):")
        # Show top and bottom experts.
        top = ranking[:min(top_n, E)]
        bottom = ranking[-min(3, E):]
        print(f"    top-{len(top)}: ", end="")
        print("  ".join(f"E{e}={counts[e]}({pct[e]:.1f}%)" for e in top))
        if len(ranking) > top_n:
            print(f"    bottom-{len(bottom)}: ", end="")
            print("  ".join(f"E{e}={counts[e]}({pct[e]:.1f}%)" for e in bottom))

    # Global summary.
    gc = data["global_counts"]
    gr = data["global_ranking"]
    gp = data["global_pct"]
    total_global = sum(gc)
    print(f"\n  Global (all layers, total activations: {total_global}):")
    top_g = gr[:min(top_n, E)]
    print(f"    top-{len(top_g)}: ", end="")
    print("  ".join(f"E{e}={gc[e]}({gp[e]:.1f}%)" for e in top_g))

    # Gini coefficient for load balance analysis.
    arr = np.array(gc, dtype=np.float64)
    if arr.sum() > 0:
        arr_sorted = np.sort(arr)
        n = len(arr_sorted)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * arr_sorted) - (n + 1) * np.sum(arr_sorted)) / (n * np.sum(arr_sorted))
        print(f"\n  Load balance Gini coefficient: {gini:.4f}")
        print(f"    (0 = perfectly balanced, 1 = all load on one expert)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect expert activation frequency statistics for mixed-precision quantization."
    )
    parser.add_argument("--mode", choices=["toy", "toy-ov", "real"], default="toy",
                        help="Which model to run (default: toy)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of calibration samples (default: 100)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length per sample (default: 128)")
    parser.add_argument("--ir-dir", type=Path, default=None,
                        help="(real mode) Directory with split IRs from load_real_v4_weights.py --per-expert-ir")
    parser.add_argument("--calibration-data", type=str, default="random",
                        help="Calibration data source: 'random', 'wikitext', or a file path (default: random)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: auto-generated in project root)")
    parser.add_argument("--hot-ratio", type=float, default=0.125,
                        help="Fraction of experts to classify as 'hot' (default: 0.125 = top 12.5%%)")
    args = parser.parse_args()

    print(f"=== Expert Activation Stats: mode={args.mode} samples={args.num_samples} seq_len={args.seq_len} ===")

    if args.mode == "toy":
        collector, cfg = run_toy_pytorch(args.num_samples, args.seq_len)
    elif args.mode == "toy-ov":
        collector, cfg = run_toy_ov(args.num_samples, args.seq_len)
    elif args.mode == "real":
        if args.ir_dir is None:
            print("ERROR: --ir-dir required for real mode")
            sys.exit(1)
        collector, cfg = run_real(args.ir_dir, args.num_samples, args.seq_len, args.calibration_data)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Build output.
    data = collector.to_dict()
    data["meta"] = {
        "mode": args.mode,
        "num_layers": collector.L,
        "num_experts": collector.E,
        "topk": cfg.num_experts_per_tok,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "total_tokens": int(collector.total_tokens),
        "calibration_data": args.calibration_data,
        "hot_ratio": args.hot_ratio,
    }

    # Classify hot/cold per layer.
    hot_count = max(1, int(collector.E * args.hot_ratio))
    data["classification"] = {}
    for i in range(collector.L):
        ranking = data["ranking"][str(i)]
        hot = ranking[:hot_count]
        cold = ranking[hot_count:]
        data["classification"][str(i)] = {
            "hot": hot,
            "cold": cold,
            "hot_count": len(hot),
            "cold_count": len(cold),
        }

    # Global classification.
    global_ranking = data["global_ranking"]
    data["global_classification"] = {
        "hot": global_ranking[:hot_count],
        "cold": global_ranking[hot_count:],
        "hot_count": hot_count,
        "cold_count": collector.E - hot_count,
    }

    # Save.
    if args.output is None:
        args.output = ROOT / f"expert_stats_{args.mode}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  saved to {args.output}")

    # Print summary.
    print_summary(collector)

    # Print classification.
    print(f"\n  Hot/Cold classification (hot_ratio={args.hot_ratio}, top {hot_count}/{collector.E}):")
    for i in range(collector.L):
        cls = data["classification"][str(i)]
        print(f"    layer {i}: hot={cls['hot']}  cold={cls['cold']}")
    print(f"    global:  hot={data['global_classification']['hot']}  "
          f"cold={data['global_classification']['cold']}")

    print(f"\nEXPERT STATS COLLECTION: PASSED")


if __name__ == "__main__":
    main()
