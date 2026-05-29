"""2.5 — Speculative expert prefetch.

Idea: while layer N's MoE is dispatching, a small predictor speculates layer
N+1's top-K experts and the orchestrator pre-compiles them in the background
so they're warm when layer N+1 starts.

Pipeline:
  1. Build per-layer linear predictor: y2_flat (layer N pre-MoE input) → next
     layer's router logits (shape [N, E]). Trained on (input, target) pairs
     collected from calibration: input = layer N y2_flat, target = layer N+1
     real gate weights.
  2. Export each predictor as a tiny OV IR.
  3. Orchestrator: at layer N, dispatch experts AND kick off predictor[N+1]
     + a ThreadPoolExecutor task that pre-warms predicted top-K experts of
     layer N+1 (compile_model is the expensive step that prefetch hides).
  4. Track predictor recall@K and cache-warm hit rate.

End-to-end output stays bit-exact with the FP32-split orchestrator because
prefetch is purely a latency optimization — the actual expert dispatch at
layer N+1 still uses the real gate output, never the predicted one.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov

from src import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


# ---------------------------------------------------------------------------
# Calibration: collect (y2_flat at layer i, gate weights at layer i+1) pairs.
# ---------------------------------------------------------------------------
def collect_pairs(core, split_dir: Path, cfg, n_inputs: int, seq_len: int, seed: int):
    """Run n_inputs through the split orchestrator and return per-layer pairs:
        pairs[i] = (X[N_total, dim], Y[N_total, E])
    where X is layer-i y2_flat tokens and Y is layer-(i+1) full gate weights
    (post-softmax via topk + scatter). For the last layer there's no next, so
    only L-1 layers contribute pairs.
    """
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    H = cfg.hc_mult
    d = cfg.hidden_size

    embed_c = core.compile_model(str(split_dir / "embed.xml"), "CPU")
    pre_c = [core.compile_model(str(split_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(split_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    expert_c = [
        [core.compile_model(str(split_dir / f"expert_L{i}_E{e}.xml"), "CPU") for e in range(E)]
        for i in range(L)
    ]

    rng = np.random.default_rng(seed)
    Xs = {i: [] for i in range(L - 1)}
    Ys = {i: [] for i in range(L - 1)}
    for _ in range(n_inputs):
        ids = rng.integers(0, cfg.vocab_size, size=(1, seq_len)).astype(np.int64)
        h = embed_c([ids])[0]
        # Cache per-layer (y2_flat, full_gate_weights) and consume after stepping forward.
        layer_inputs = []
        layer_gates = []
        for i in range(L):
            out = pre_c[i]([h, ids])
            y2_flat = out[0]; x_res = out[1]; post2 = out[2]; comb2 = out[3]
            weights = out[4]; indices = out[5]; shared_out = out[6]
            # Reconstruct full [N, E] gate matrix (zeros except at top-K).
            N = y2_flat.shape[0]
            gm = np.zeros((N, E), dtype=np.float32)
            np.put_along_axis(gm, indices.astype(np.int64), weights.astype(np.float32), axis=1)
            layer_inputs.append(y2_flat.copy())
            layer_gates.append(gm.copy())
            moe = np.zeros((N, d), dtype=np.float32)
            for e in np.unique(indices).tolist():
                moe += gm[:, e:e + 1] * expert_c[i][int(e)]([y2_flat])[0]
            moe += shared_out
            h = post_c[i]([moe.reshape(1, seq_len, d), x_res, post2, comb2])[0]
        for i in range(L - 1):
            Xs[i].append(layer_inputs[i])
            Ys[i].append(layer_gates[i + 1])

    pairs = {}
    for i in range(L - 1):
        pairs[i] = (np.concatenate(Xs[i], axis=0), np.concatenate(Ys[i], axis=0))
    return pairs


# ---------------------------------------------------------------------------
# Predictor: per-layer linear projection y2_flat -> gate logits.
# ---------------------------------------------------------------------------
class LinearPredictor(nn.Module):
    def __init__(self, dim: int, n_experts: int):
        super().__init__()
        self.proj = nn.Linear(dim, n_experts, bias=True)

    def forward(self, x):
        return self.proj(x)


def train_predictor(X: np.ndarray, Y: np.ndarray, n_experts: int, epochs: int = 200) -> LinearPredictor:
    """X: [N, dim], Y: [N, E] (target = real next-layer gate weights).
    MSE loss against the dense gate matrix — predicts both which experts are
    hot and their soft scores. At inference we just topk the prediction."""
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    pred = LinearPredictor(X.shape[1], n_experts)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(pred(X_t), Y_t)
        loss.backward()
        opt.step()
    return pred


def eval_recall(pred: LinearPredictor, X: np.ndarray, Y: np.ndarray, topk: int) -> float:
    """Per-token recall@topk: of the topk experts pred says will be hot, how
    many are actually in the real top-K of Y?"""
    with torch.no_grad():
        p = pred(torch.from_numpy(X).float()).numpy()
    pred_top = np.argpartition(-p, topk, axis=1)[:, :topk]
    true_top = np.argpartition(-Y, topk, axis=1)[:, :topk]
    pred_set = [set(r) for r in pred_top]
    true_set = [set(r) for r in true_top]
    hits = sum(len(p & t) for p, t in zip(pred_set, true_set))
    return hits / (topk * len(pred_set))


def export_predictor(pred: LinearPredictor, dim: int, save_path: Path):
    example = torch.zeros(1, dim, dtype=torch.float32)
    ov_model = ov.convert_model(
        pred.eval(),
        example_input=(example,),
        input=[([-1, dim], ov.Type.f32)],
    )
    ov_model.outputs[0].set_names({"router_logits"})
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(save_path), compress_to_fp16=False)


# ---------------------------------------------------------------------------
# Orchestrator with prefetch.
# ---------------------------------------------------------------------------
def orchestrate_with_prefetch(
    core, split_dir: Path, predictor_dir: Path, cfg,
    ids_np: np.ndarray, topk_predict: int,
):
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    d = cfg.hidden_size

    embed_c = core.compile_model(str(split_dir / "embed.xml"), "CPU")
    final_c = core.compile_model(str(split_dir / "final.xml"), "CPU")
    pre_c = [core.compile_model(str(split_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(split_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    # The dispatch cache: compiled experts keyed by (layer, expert).
    expert_cache: "dict[tuple[int, int], object]" = {}
    pred_c = [core.compile_model(str(predictor_dir / f"predictor_L{i}.xml"), "CPU")
              for i in range(L - 1)]

    pool = ThreadPoolExecutor(max_workers=4)
    prefetch_futures: "dict[tuple[int, int], object]" = {}
    stats = {
        "predicted_hits": 0,    # prefetched expert was actually selected
        "predicted_misses": 0,  # prefetched expert was NOT selected (wasted prefetch)
        "demand_loads": 0,      # selected expert was not prefetched (cache miss)
        "prefetch_warm_hits": 0,  # selected expert was prefetched and ready
    }

    def compile_expert(layer, expert):
        return core.compile_model(str(split_dir / f"expert_L{layer}_E{expert}.xml"), "CPU")

    def get_expert(layer, expert):
        key = (layer, expert)
        if key in prefetch_futures:
            cmp = prefetch_futures.pop(key).result()
            expert_cache[key] = cmp
            stats["prefetch_warm_hits"] += 1
            return cmp
        if key in expert_cache:
            return expert_cache[key]
        # demand-load (no prefetch)
        cmp = compile_expert(layer, expert)
        expert_cache[key] = cmp
        stats["demand_loads"] += 1
        return cmp

    h = embed_c([ids_np])[0]
    for i in range(L):
        out = pre_c[i]([h, ids_np])
        y2_flat = out[0]; x_res = out[1]; post2 = out[2]; comb2 = out[3]
        weights = out[4]; indices = out[5]; shared_out = out[6]

        # SPECULATE: predict layer i+1 top-K and start prefetching their IRs.
        if i + 1 < L:
            logits = pred_c[i]([y2_flat])[0]                        # [N, E]
            agg = logits.sum(axis=0)                                 # batch-level salience
            predicted = np.argpartition(-agg, topk_predict)[:topk_predict].tolist()
            for e in predicted:
                key = (i + 1, e)
                if key in expert_cache or key in prefetch_futures:
                    continue
                prefetch_futures[key] = pool.submit(compile_expert, i + 1, e)

        # DISPATCH: real experts for layer i.
        N = y2_flat.shape[0]
        gm = np.zeros((N, E), dtype=np.float32)
        np.put_along_axis(gm, indices.astype(np.int64), weights.astype(np.float32), axis=1)
        moe = np.zeros((N, d), dtype=np.float32)
        active = np.unique(indices).tolist()
        for e in active:
            exp_c = get_expert(i, int(e))
            moe += gm[:, e:e + 1] * exp_c([y2_flat])[0]
        moe += shared_out
        h = post_c[i]([moe.reshape(1, ids_np.shape[1], d), x_res, post2, comb2])[0]

    # Drain any remaining unused prefetches and count them as misses.
    for key, fut in prefetch_futures.items():
        fut.result()  # let the background threads finish cleanly
        stats["predicted_misses"] += 1

    pool.shutdown(wait=True)
    logits_out = final_c([h])[0]
    return logits_out, stats


def orchestrate_baseline(core, split_dir: Path, cfg, ids_np: np.ndarray):
    """No predictor, no prefetch — for output equivalence comparison."""
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    d = cfg.hidden_size

    embed_c = core.compile_model(str(split_dir / "embed.xml"), "CPU")
    final_c = core.compile_model(str(split_dir / "final.xml"), "CPU")
    pre_c = [core.compile_model(str(split_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(split_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    expert_c = [
        [core.compile_model(str(split_dir / f"expert_L{i}_E{e}.xml"), "CPU") for e in range(E)]
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
    parser.add_argument("--n-eval-inputs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--topk-predict", type=int, default=4,
                        help="How many experts to speculatively prefetch per layer.")
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = make_toy_config()
    L = cfg.num_hidden_layers
    E = cfg.n_routed_experts
    d = cfg.hidden_size

    split_dir = ROOT / "ov_ir_toy" / "expert_split"
    predictor_dir = ROOT / "ov_ir_toy" / "expert_split_predictors"
    if not (split_dir / "embed.xml").exists():
        raise RuntimeError("Run scripts/split_to_expert_irs.py first.")
    predictor_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()

    print(f"=== Calibration: collect (layer_i y2_flat, layer_i+1 gate) pairs ===")
    t0 = time.perf_counter()
    pairs = collect_pairs(core, split_dir, cfg,
                          n_inputs=args.n_calibration_inputs,
                          seq_len=args.seq_len, seed=42)
    cal_time = time.perf_counter() - t0
    for i, (X, Y) in pairs.items():
        print(f"  L{i} -> L{i+1}: {X.shape[0]} training pairs, X.shape={X.shape}, Y.shape={Y.shape}")
    print(f"  collected in {cal_time:.1f} s")

    print(f"\n=== Train per-layer linear predictors ({args.epochs} epochs each) ===")
    predictors = {}
    for i, (X, Y) in pairs.items():
        pred = train_predictor(X, Y, n_experts=E, epochs=args.epochs)
        recall = eval_recall(pred, X, Y, topk=cfg.num_experts_per_tok)
        predictors[i] = (pred, recall)
        export_predictor(pred, d, predictor_dir / f"predictor_L{i}.xml")
        size_kb = (predictor_dir / f"predictor_L{i}.bin").stat().st_size / 1024
        print(f"  L{i+1} predictor: recall@{cfg.num_experts_per_tok}={recall:.3f}  "
              f"params={d*E + E}  size={size_kb:.1f} KB")

    print(f"\n=== Held-out evaluation ({args.n_eval_inputs} inputs, "
          f"prefetch top-{args.topk_predict}) ===")
    rng = np.random.default_rng(99)
    total_stats = {"predicted_hits": 0, "predicted_misses": 0,
                   "demand_loads": 0, "prefetch_warm_hits": 0}
    greedy_match = 0
    for n in range(args.n_eval_inputs):
        ids = rng.integers(0, cfg.vocab_size, size=(1, args.seq_len)).astype(np.int64)
        baseline_out = orchestrate_baseline(core, split_dir, cfg, ids)
        prefetch_out, stats = orchestrate_with_prefetch(
            core, split_dir, predictor_dir, cfg, ids, topk_predict=args.topk_predict
        )
        for k, v in stats.items():
            total_stats[k] += v
        b_top = int(torch.from_numpy(baseline_out)[0, -1].argmax().item())
        p_top = int(torch.from_numpy(prefetch_out)[0, -1].argmax().item())
        greedy_match += int(b_top == p_top)
        bit_exact = bool(np.array_equal(baseline_out, prefetch_out))
        print(f"  input {n+1}: baseline_top={b_top} prefetch_top={p_top} "
              f"match={b_top == p_top} bit_exact={bit_exact}  stats={stats}")

    total_real_selections = total_stats["prefetch_warm_hits"] + total_stats["demand_loads"]
    warm_hit_rate = (total_stats["prefetch_warm_hits"] / total_real_selections
                     if total_real_selections else 0.0)
    total_prefetched = total_stats["prefetch_warm_hits"] + total_stats["predicted_misses"]
    waste_rate = (total_stats["predicted_misses"] / total_prefetched
                  if total_prefetched else 0.0)

    print("\n=== Summary ===")
    print(f"  greedy match across {args.n_eval_inputs} inputs: {greedy_match}/{args.n_eval_inputs}")
    print(f"  warm-hit rate     : {warm_hit_rate:.1%}  "
          f"({total_stats['prefetch_warm_hits']}/{total_real_selections} real expert dispatches "
          f"hit a warm prefetch)")
    print(f"  prefetch waste rate: {waste_rate:.1%}  "
          f"({total_stats['predicted_misses']}/{total_prefetched} prefetched experts unused)")
    print(f"  demand loads      : {total_stats['demand_loads']}")

    # Persist for downstream NPU-predictor work in 2.6.
    (predictor_dir / "predictor_stats.json").write_text(json.dumps({
        "n_calibration_inputs": args.n_calibration_inputs,
        "n_eval_inputs": args.n_eval_inputs,
        "seq_len": args.seq_len,
        "topk_predict": args.topk_predict,
        "per_layer_recall_at_topk": {
            f"L{i+1}": float(predictors[i][1]) for i in predictors
        },
        "warm_hit_rate": float(warm_hit_rate),
        "prefetch_waste_rate": float(waste_rate),
        "total_stats": total_stats,
        "greedy_match": greedy_match,
    }, indent=2))
    print(f"  wrote {predictor_dir / 'predictor_stats.json'}")

    assert greedy_match == args.n_eval_inputs, "prefetch path diverged from baseline"
    print("\nSPECULATIVE PREFETCH: PASSED")


if __name__ == "__main__":
    main()
