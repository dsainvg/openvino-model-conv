# Upstream issue drafts — Qwen3.6 / qwen3_5_moe

Two drafts: one for `huggingface/optimum-intel` (export support), one for
`openvinotoolkit/openvino.genai` (a reusable MoE selective-compute pattern).
Pick one or post both. Reference implementation:
https://github.com/bob798/deepseek-v4-openvino (`src/qwen36/`).

---

## Target: `huggingface/optimum-intel`

**Title:** Native OpenVINO export for Qwen3.6 / Qwen3.5 MoE (`model_type=qwen3_5_moe`)

**Labels (suggested):** `enhancement`, `model support`, `openvino`

**Body:**

Qwen released the Qwen3.6-35B-A3B family (registered as `model_type="qwen3_5_moe"`,
class `Qwen3_5MoeForConditionalGeneration`). As of `optimum-intel` 1.27.0,
`optimum-cli export openvino` fails before it even reaches the exporter:

```
ValueError: The checkpoint you are trying to load has model type `qwen3_5_moe`
but Transformers does not recognize this architecture.
```

with `transformers>=5.5` (which does register the architecture) the export then
fails at the OpenVINO exporter config map, which registers `qwen2`, `qwen2_moe`,
`qwen3`, `qwen3_moe`, `qwen2_vl`, `qwen2_5_vl` but not `qwen3_5_moe`.

### What makes this architecture non-trivial

Unlike `qwen3_moe`, this family is a **3:1 hybrid**: 3 Gated-DeltaNet
(linear-attention) layers per 1 full GQA-attention layer. The DeltaNet layers:

- maintain recurrent state (`conv_state` + `recurrent_state`) updated in place
  inside a `Cache` object — not graph-traceable as written;
- default to CUDA-only kernels (`flash-linear-attention`, `causal-conv1d`) with
  pure-torch fallbacks (`torch_recurrent_gated_delta_rule`,
  `torch_causal_conv1d_update`) that *are* traceable.

A working OpenVINO export needs: (1) a model-patcher that forces the torch
fallbacks, (2) recurrent/conv state lifted to explicit model inputs/outputs
(stateful or KV-style), (3) the MoE routing expressed graph-friendly.

### Reference implementation

The linked repo has a self-contained, OV-traceable port: `QwenGatedDeltaNet`
with functional state IO, GQA attention with explicit KV cache, MoE as
compute-all+mask (single IR) **or** a backbone + per-expert split, plus a GPTQ
int4 dequantizer that reads the community checkpoints directly (no `gptqmodel`
runtime needed). `ov.convert_model` traces the whole thing; CPU and Arc iGPU
both compile it; logits match eager PyTorch within ~4e-4. Happy to help shape a
`Qwen3_5MoeOpenVINOConfig` + patcher PR mirroring the `qwen3_moe` one.

---

## Target: `openvinotoolkit/openvino.genai`

**Title:** Pattern: backbone + per-expert split IR for large-sparse MoE (top-k of 256)

**Labels (suggested):** `enhancement`, `feature request`, `LLM`

**Body:**

For MoE models where total params are large but active params per token are
small (Qwen3.6: 35B total / 3B active, 256 routed experts, top-8), compiling a
single IR with all experts unrolled is impractical. A split that the runtime
could support as a first-class pattern:

- **backbone IR** per layer: attention + router + shared expert, emitting the
  top-k expert indices/weights as outputs;
- **per-expert IRs** (or one weight-parameterized expert IR served from a
  cache): only the selected top-k run on each token;
- a thin orchestrator: backbone → top-k select → dispatch selected experts →
  weighted combine.

In my reference implementation this gives **bit-near-exact** parity with the
monolithic model (max abs-diff 3.3e-7, *tighter* than the monolithic OV path
because there is no large-graph fusion reordering), while only `top_k/num_experts`
of expert FLOPs execute (8/256 → 32x). An LRU expert cache + calibration prewarm
keeps the hot set resident; the dense backbone can run on the iGPU while the
sparse experts stay on CPU.

The ask: either guidance on the idiomatic way to express this in
openvino.genai's pipeline, or interest in a reusable `MoESplitPipeline` helper.
Reference: https://github.com/bob798/deepseek-v4-openvino (`src/qwen36/split_inference.py`,
`expert_manager.py`, `pipeline.py`).

---

## Notes before posting

- Verify version numbers against the latest `optimum-intel` / `transformers` at
  posting time (drafted against optimum-intel 1.27.0, transformers 5.9.0).
- The repo is a PoC on a 64GB Intel AI PC; full 40-layer real-weight runs are
  validated per-layer, not end-to-end in CI (≈13 min load).
- Multimodal (vision tower + MTP head) is intentionally out of scope in the
  reference port — mention only the text path upstream.
