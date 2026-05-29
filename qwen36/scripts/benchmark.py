"""Phase 3 benchmark for the Qwen3.6 split-inference engine.

Measures the plan's core value proposition -- selective (top-k) expert
computation vs naive compute-all -- plus end-to-end decode throughput and
the quantization compression ratio. Runs on a configurable toy so it
finishes in seconds; the ratios it reports are the same ones that apply at
full scale (256 experts, top-8).

  selective speedup  ~= num_experts / top_k         (expert-compute only)
  quant compression  ~= fp32_bytes / int4_bytes     (routed-expert weights)

Out of scope here (documented, not run):
  - WikiText-2 perplexity: needs a reference run we can't produce on this
    Windows host (no gptqmodel; no BF16 checkpoint downloaded).
  - Multimodal: vision tower not ported (text-only).
  - CPU+iGPU split: requires the OV device plumbing from Phase 3.2 proper.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.configuration import make_toy_config  # noqa: E402
from src.modeling import QwenForCausalLM  # noqa: E402
from src.pipeline import Qwen36Pipeline  # noqa: E402


def _bench_config():
    """A toy big enough for stable timings but still seconds to run."""
    return make_toy_config(
        num_layers=4,
        full_attention_interval=4,  # 3 linear + 1 full
        hidden_size=256,
        num_experts=32,
        top_k=8,
        moe_intermediate=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
    )


def _time(fn, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def main() -> int:
    cfg = _bench_config()
    print("=" * 70)
    print("Qwen3.6 split-inference benchmark (toy proxy for full scale)")
    print("=" * 70)
    print(f"config: {cfg.num_hidden_layers} layers ({cfg.layer_types}), "
          f"{cfg.num_experts} experts top-{cfg.num_experts_per_tok}, hidden={cfg.hidden_size}")

    torch.manual_seed(0)
    model = QwenForCausalLM(cfg)
    model.eval()

    state_template = model.empty_state(batch=1, max_seq=16)
    input_ids = torch.tensor([[42]])
    position_ids = torch.tensor([[0]])

    def clone_state():
        return {k: [t.clone() for t in v] for k, v in state_template.items()}

    # --- Selective (pipeline) vs compute-all (monolithic forward) ---
    pipe = Qwen36Pipeline(model)

    def run_selective():
        pipe.step(input_ids, position_ids, clone_state())

    def run_compute_all():
        s = clone_state()
        model(input_ids, position_ids, s["k_caches"], s["v_caches"], s["conv_states"], s["rec_states"])

    iters = 20
    t_sel = _time(run_selective, iters)
    t_all = _time(run_compute_all, iters)

    print("\n--- Expert computation: selective vs compute-all ---")
    print(f"  compute-all (all {cfg.num_experts} experts/layer):  {t_all*1e3:7.2f} ms/step")
    print(f"  selective  (top-{cfg.num_experts_per_tok} experts/layer):   {t_sel*1e3:7.2f} ms/step")
    print(f"  measured speedup:        {t_all/t_sel:5.2f}x")
    print(f"  theoretical expert-FLOP: {cfg.num_experts/cfg.num_experts_per_tok:5.2f}x "
          f"(experts run: {cfg.num_experts_per_tok}/{cfg.num_experts})")
    print(f"  --> at full scale (256 experts, top-8): "
          f"{256/8:.0f}x fewer expert FLOPs")

    # --- Decode throughput ---
    print("\n--- Decode throughput (greedy, 32 new tokens) ---")
    pipe2 = Qwen36Pipeline(model)
    res = pipe2.generate([1, 2, 3, 4, 5], max_new_tokens=32, greedy=True)
    print(f"  steps: {res.steps}  total: {res.seconds*1e3:.1f} ms  "
          f"throughput: {res.tokens_per_second:.1f} tok/s")
    # TTFT: prefill of the prompt + first decode step
    prompt = [1, 2, 3, 4, 5]
    pipe3 = Qwen36Pipeline(model)
    t0 = time.perf_counter()
    pipe3.generate(prompt, max_new_tokens=1, greedy=True)
    ttft = time.perf_counter() - t0
    print(f"  TTFT (prompt len {len(prompt)} + 1 token): {ttft*1e3:.1f} ms")

    # --- LRU cache behaviour (capacity sweep) ---
    print("\n--- LRU cache hit rate vs capacity ---")
    from src.expert_manager import ExpertManager
    working_set = cfg.num_hidden_layers * cfg.num_experts_per_tok  # keys touched per step
    total_keys = cfg.num_hidden_layers * cfg.num_experts
    for cap in (cfg.num_experts_per_tok, working_set, total_keys):
        mgr = ExpertManager(lambda li, ei: model.model.layers[li].mlp.experts[ei], capacity=cap)
        Qwen36Pipeline(model, mgr).generate([1, 2, 3, 4, 5], max_new_tokens=32, greedy=True)
        tag = {cfg.num_experts_per_tok: "top_k", working_set: "1-step working set", total_keys: "all experts"}[cap]
        print(f"  capacity={cap:4d} ({tag:18s}): {mgr.stats}")
    print("  NOTE: toy uses random-init routing -> ~uniform expert selection,")
    print("        so cross-step locality is low. A trained model routes with")
    print("        heavy skew (hot experts), where a modest LRU + calibration")
    print("        prewarm gives high hit rates. See ExpertFrequency.prewarm.")

    # --- Quantization compression ---
    print("\n--- Quantization compression (routed-expert weights) ---")
    params_per_expert = (
        cfg.hidden_size * cfg.moe_intermediate_size * 2  # gate + up
        + cfg.moe_intermediate_size * cfg.hidden_size  # down
    )
    fp32_bytes = params_per_expert * 4
    int4_bytes = params_per_expert * 0.5  # 4-bit ideal
    # GPTQ adds scales (fp16, per group) + zeros (int4 packed) overhead.
    gs = 128
    n_groups = cfg.hidden_size // gs if cfg.hidden_size >= gs else 1
    print(f"  params/expert: {params_per_expert:,}")
    print(f"  fp32: {fp32_bytes/1024:.1f} KB   int4 (ideal): {int4_bytes/1024:.1f} KB   "
          f"ratio: {fp32_bytes/int4_bytes:.1f}x")
    print(f"  real checkpoint: palmfuture GPTQ-Int4 = 22.78 GB on disk vs "
          f"~70 GB BF16 -> {70/22.78:.2f}x")

    print("\n" + "=" * 70)
    print("Note: perplexity / multimodal / iGPU-split benchmarks are out of "
          "scope on this host (see module docstring).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
