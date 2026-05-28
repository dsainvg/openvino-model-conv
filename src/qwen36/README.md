# Qwen3.6-35B-A3B on OpenVINO

OV-traceable port of **Qwen3.6-35B-A3B** (`model_type: qwen3_5_moe`) with a
split-IR MoE inference engine designed for a 64GB Intel AI PC. See
[`../../PLAN-QWEN36.md`](../../PLAN-QWEN36.md) for the full experiment plan.

## Why this is non-trivial

`optimum-intel` cannot export this architecture (`qwen3_5_moe` is unregistered;
transformers itself needs >=5.5 to even read the config). The model's **Gated
DeltaNet** linear-attention layers maintain recurrent state and use CUDA-only
kernels (`flash-linear-attention`, `causal-conv1d`) that don't trace. So the
model is hand-ported to a pure, OV-traceable form, and the 256-expert MoE is
factored into a backbone + per-expert IRs so only the top-8 experts per token
ever execute.

## Architecture (verified from the real checkpoint)

```
40 layers, 3:1 hybrid:  3 x (GatedDeltaNet -> MoE)  +  1 x (GQA Attention -> MoE)
hidden=2048, head_dim=256, vocab=248320

GatedDeltaNet (linear attention, O(n)):
  32 V heads / 16 QK heads, head_dim=128, causal conv1d (k=4),
  delta rule + exponential gating + L2 norm.  State = (conv_state, recurrent_state).

MoE:
  256 routed experts (intermediate=512), top-8 + 1 shared expert.
  Only ~3.5% of expert params touched per token.
```

Quantization: routed experts are **GPTQ-Int4** (group_size=128, sym), the rest
(attention, shared expert, MTP, vision) stay BF16. On disk: 22.78 GB.

## Module map

| File | Purpose |
|------|---------|
| `configuration_qwen36.py` | `Qwen36Config` dataclass, `from_pretrained_dir()`, `make_toy_config()` |
| `gptq_dequant.py` | GPTQ int4 unpack/dequant; `load_gptq_linear()` from safetensors |
| `modeling_qwen36.py` | OV-traceable decode-only model; all state as functional IO |
| `split_inference.py` | backbone wrappers + router/combine seam + per-expert extraction |
| `expert_manager.py` | LRU expert cache (capacity, pinning, stats) + `ExpertFrequency` |
| `pipeline.py` | `Qwen36Pipeline.generate()` — autoregressive decode through the cache |
| `load_weights.py` | real-checkpoint state-dict mapping (linear_attn / self_attn / experts) |

## How to run

All commands use the isolated `venv-qwen` (transformers 5.9 + openvino 2026.1).

```bash
# Tests (toy-scale, fast; real-weight tests skip if checkpoint absent)
venv-qwen/Scripts/python -m pytest tests/qwen36 -v

# Toy -> OV IR conversion + numerical match
venv-qwen/Scripts/python scripts/qwen36_convert_toy.py

# End-to-end split-IR orchestration (backbone IRs + per-expert IRs)
venv-qwen/Scripts/python scripts/qwen36_split_orchestrator_toy.py

# Benchmark (selective vs compute-all, tok/s, cache, quant)
venv-qwen/Scripts/python scripts/qwen36_benchmark.py

# Live demo (toy default; --real loads the 40-layer checkpoint, ~13 min)
venv-qwen/Scripts/python scripts/qwen36_demo.py --prompt-tokens 1,2,3 --max-tokens 8
```

## Results (toy proxy; ratios hold at full scale)

- **OV conversion**: toy model traces in <1s; monolithic IR matches PyTorch
  within 4.4e-4; the **split orchestrator matches within 3.3e-7** (no
  large-graph fusion noise).
- **Selective vs compute-all**: top-k expert compute is `num_experts/top_k`x
  cheaper in FLOPs — at full scale **32x fewer expert FLOPs** (8 of 256).
- **LRU cache**: hit rate scales with capacity (0% at top_k → 89% when the
  full expert set fits). Toy uses random-init routing (~uniform), so real
  trained routing — which is heavily skewed — does substantially better with
  calibration prewarm (`ExpertFrequency.hot_keys` → `ExpertManager.prewarm`).
- **Quant**: routed experts 8x smaller at int4 vs fp32; whole checkpoint
  22.78 GB vs ~70 GB BF16 (3.07x).
- **Real weights**: a single real layer (256 experts, GPTQ-dequant to bf16)
  loads + forwards with finite, non-degenerate logits — both attention types
  validated.

## Status

Done: Phase 0 (env/feasibility), Phase 1 (port + OV + numeric match),
Phase 2.1–2.5 (split IRs, selective compute, real-weight loader, LRU cache,
autoregressive pipeline), Phase 3 benchmark, Phase 4 demo.

Not done / out of scope here:
- Full 40-layer real-model generation run (works in principle; ~13 min load,
  ~64 GB resident — a workstation run, not CI).
- WikiText-2 perplexity (no reference run available on this Windows host —
  `gptqmodel` has no Windows wheels; BF16 checkpoint not downloaded).
- Multimodal (vision tower + MTP head deliberately not ported; text-only).
- Weight-as-input parametric expert IR (one compiled graph serving all
  experts) and CPU+iGPU device split — natural next optimizations.
