"""Phase 4 interactive demo for the Qwen3.6 split-inference engine.

Shows, live per step: which experts fire in each layer, the LRU cache hit
rate, and the running tokens/sec. Two modes:

  (default) toy   -- random-weight toy model. Instant. Demonstrates the
                     engine + instrumentation, NOT real text (weights are
                     random, so generated token ids are meaningless).

  --real          -- loads the real Qwen3.6 weights and tokenizer and
                     generates actual text. WARNING: loading all 40 layers
                     (256 experts each) takes ~13 min and needs the full
                     checkpoint resident; intended for a workstation run,
                     not a quick smoke test. Use --layers to cap layers for
                     a faster (incorrect-output) instrumentation preview.

Usage:
    python scripts/qwen36_demo.py --prompt-tokens 1,2,3 --max-tokens 12
    python scripts/qwen36_demo.py --real --prompt "The capital of France is" --max-tokens 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.configuration import make_toy_config  # noqa: E402
from src.expert_manager import ExpertManager  # noqa: E402
from src.modeling import QwenForCausalLM  # noqa: E402
from src.pipeline import Qwen36Pipeline  # noqa: E402

MODEL_DIR = Path(r"C:\Users\intel\models\qwen36-35b-int4")


def _print_step(step: int, token: int, freq_snapshot, stats, tok_s: float, decoded: str | None):
    layers = sorted({layer for (layer, _e) in freq_snapshot})
    parts = []
    for layer in layers:
        experts = sorted(e for (l, e) in freq_snapshot if l == layer)
        parts.append(f"L{layer}={experts}")
    fired = "  ".join(parts)
    tok_repr = f"{token}" if decoded is None else f"{token} {decoded!r}"
    print(f"  step {step:3d} | tok={tok_repr:24s} | {fired} | "
          f"cache {stats.hit_rate:5.1%} | {tok_s:6.1f} tok/s")


def run_toy(args) -> int:
    cfg = make_toy_config(num_layers=4, full_attention_interval=4, num_experts=16, top_k=4)
    print(f"[toy] {cfg.num_hidden_layers} layers, {cfg.num_experts} experts top-{cfg.num_experts_per_tok}, "
          f"hidden={cfg.hidden_size} (random weights -> token ids are illustrative only)\n")
    torch.manual_seed(0)
    model = QwenForCausalLM(cfg)
    model.eval()

    prompt = [int(t) for t in args.prompt_tokens.split(",")] if args.prompt_tokens else [1, 2, 3]
    mgr = ExpertManager(lambda li, ei: model.model.layers[li].mlp.experts[ei], capacity=args.capacity)
    pipe = Qwen36Pipeline(model, mgr, collect_freq=True)

    _stream_generate(pipe, prompt, args.max_tokens, tokenizer=None, eos=None)
    return 0


def run_real(args) -> int:
    if not MODEL_DIR.exists():
        print(f"ERROR: checkpoint not found at {MODEL_DIR}")
        return 1
    from dataclasses import replace as dc_replace

    from transformers import AutoTokenizer

    from src.configuration import Qwen36Config
    from src.load_weights import build_shard_index, load_global_weights, load_layer_weights

    print(f"[real] loading tokenizer from {MODEL_DIR} ...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    cfg = Qwen36Config.from_pretrained_dir(MODEL_DIR)
    if args.layers is not None:
        cfg = dc_replace(cfg, num_hidden_layers=args.layers,
                         layer_types=cfg.layer_types[:args.layers])
        print(f"[real] WARNING: capping to {args.layers} layers -> outputs are NOT valid text, "
              "instrumentation preview only.")
    n_layers = cfg.num_hidden_layers

    model = QwenForCausalLM(cfg)
    model.eval()
    weight_map = build_shard_index(MODEL_DIR)
    print(f"[real] loading {n_layers} layers x {cfg.num_experts} experts (~{n_layers*19//60} min est.) ...")
    load_global_weights(model, weight_map, MODEL_DIR, out_dtype=torch.bfloat16)
    for i in range(n_layers):
        load_layer_weights(model.model.layers[i], i, weight_map, MODEL_DIR, out_dtype=torch.bfloat16)
        print(f"    layer {i+1}/{n_layers} loaded", end="\r")
    print()

    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.flatten().tolist()
    print(f"[real] prompt: {args.prompt!r} -> {len(prompt_ids)} tokens\n")

    mgr = ExpertManager(lambda li, ei: model.model.layers[li].mlp.experts[ei], capacity=args.capacity)
    pipe = Qwen36Pipeline(model, mgr, collect_freq=True)
    _stream_generate(pipe, prompt_ids, args.max_tokens, tokenizer=tok,
                     eos=cfg.eos_token_id, state_dtype=torch.bfloat16)
    return 0


def _freq_delta(before: dict, after: dict) -> list[tuple[int, int]]:
    """Keys whose count increased between two ExpertFrequency snapshots =
    the experts that fired in the step between them."""
    return [k for k, v in after.items() if v > before.get(k, 0)]


def _stream_generate(pipe, prompt, max_tokens, tokenizer, eos, state_dtype=torch.float32):
    import time

    max_seq = len(prompt) + max_tokens + 1
    state = pipe.model.empty_state(batch=1, max_seq=max_seq, dtype=state_dtype)

    print(f"prefilling {len(prompt)} prompt tokens ...")
    logits = None
    for pos, t in enumerate(prompt):
        logits, state = pipe.step(torch.tensor([[t]]), torch.tensor([[pos]]), state)

    print("generating:")
    generated: list[int] = []
    t_start = time.perf_counter()
    pos = len(prompt)
    for step in range(max_tokens):
        next_tok = int(logits[0, -1].float().argmax().item())
        generated.append(next_tok)
        decoded = tokenizer.decode([next_tok]) if tokenizer else None

        if eos is not None and next_tok == eos:
            print(f"  step {step+1:3d} | tok={next_tok} <eos>")
            break

        # Snapshot expert frequency around the step() that processes this token
        # so the display shows exactly the experts that fired.
        before = dict(pipe.expert_freq.counts) if pipe.expert_freq else {}
        logits, state = pipe.step(torch.tensor([[next_tok]]), torch.tensor([[pos]]), state)
        after = pipe.expert_freq.counts if pipe.expert_freq else {}
        fired = _freq_delta(before, after)

        elapsed = time.perf_counter() - t_start
        tok_s = (step + 1) / elapsed if elapsed > 0 else 0.0
        _print_step(step + 1, next_tok, fired, pipe.experts.stats, tok_s, decoded)
        pos += 1

    total = time.perf_counter() - t_start
    print(f"\nsummary: {len(generated)} tokens in {total*1e3:.0f} ms "
          f"= {len(generated)/total:.1f} tok/s | final cache: {pipe.experts.stats}")
    if tokenizer:
        print(f"\noutput: {tokenizer.decode(generated)!r}")
    if pipe.expert_freq:
        hot = pipe.expert_freq.hot_keys(top_n=5)
        print(f"hottest experts (layer,expert): {hot}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.6 split-inference demo")
    ap.add_argument("--real", action="store_true", help="load real weights + tokenizer")
    ap.add_argument("--prompt", default="The capital of France is", help="text prompt (real mode)")
    ap.add_argument("--prompt-tokens", default=None, help="comma-separated token ids (toy mode)")
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--capacity", type=int, default=64, help="expert LRU capacity")
    ap.add_argument("--layers", type=int, default=None, help="real mode: cap layers (preview only)")
    args = ap.parse_args()
    return run_real(args) if args.real else run_toy(args)


if __name__ == "__main__":
    sys.exit(main())
