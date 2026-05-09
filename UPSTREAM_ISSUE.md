# Upstream issue draft

Two near-identical drafts: one for `huggingface/optimum-intel`, one for
`openvinotoolkit/openvino`. Same body, different target audience and emphasis.
Pick one or post both.

---

## Target: `huggingface/optimum-intel`

**Title:** Native support for DeepSeek-V4 (`model_type=deepseek_v4`)

**Labels (suggested):** `enhancement`, `model support`, `openvino`

**Body:**

DeepSeek released V4-Flash (Apr 30, 2026) with a substantially redesigned architecture. As of `optimum-intel` 1.27.0, `OVModelForCausalLM.from_pretrained` does not recognize `model_type="deepseek_v4"`. Users have to bring their own pre-converted IR and load it with `trust_remote_code=True` and `use_cache=False`.

I built a working end-to-end conversion path on a toy V4 model and have a real-weight loader ready. This issue surfaces what would be needed to make V4 a first-class citizen in `optimum-intel`, mirroring the `deepseek_v3` registration that already exists.

### Reference implementation

https://github.com/bob798/deepseek-v4-openvino

Self-contained: pure-PyTorch port of the V4 architecture, `ov.convert_model` integration, NNCF INT8/INT4 weight compression, FP4/FP8 dequant + name mapping for real V4-Flash weights, `OVModelForCausalLM.from_pretrained(use_cache=False)` integration via `trust_remote_code`.

### What V4-Flash adds vs V3

The published reference uses TileLang JIT kernels + FP4 (`e2m1fn`) expert weights + FP8 (`e4m3fn`) main weights with E8M0 microscaling. None of that traces through `ov.convert_model` directly. My port replaces all of that with pure-PyTorch BF16 equivalents while preserving the architectural shape:

- **Hybrid sparse attention**: sliding window (size 128) + indexer-driven top-k over compressed-KV (`compress_ratio=4`) + dense compressed-KV (`compress_ratio=128`), selected per-layer.
- **Multi-Latent Attention** with Q/O LoRA + MQA + attention sink.
- **KV Compressor** (gated pooling) + a separate **Indexer** sub-network for top-k position selection.
- **Manifold-constrained Hyper-Connections (mHC)**: hidden state carries `hc_mult=4` parallel copies through every block, recombined via Sinkhorn iteration (20 in real V4-Flash).
- **MoE**: 256 routed + 1 shared expert, top-6, `sqrtsoftplus` scoring, `noaux_tc` topk, hash-routed first 3 layers.
- **YaRN RoPE scaling** factor=16 (65k → 1M context).

### What native registration would touch

Mirroring the existing `deepseek_v3` registration (refs cited from `optimum-intel` 1.27.0):

1. `optimum/exporters/onnx/model_configs.py` — `@register_tasks_manager_onnx("deepseek_v4", ...)` + `class DeepSeekV4OnnxConfig(LlamaOnnxConfig)`. Sibling of `DeepSeekV3OnnxConfig` at line ~442.
2. `optimum/exporters/openvino/model_configs.py` — `@register_in_tasks_manager("deepseek_v4", ...)` + `DeepSeekV4OpenVINOConfig`. Sibling of `DeepseekV3OpenVINOConfig` at line ~3633.
3. `optimum/exporters/openvino/model_patcher.py` — `deepseek_v4_attn_forward` and a `DeepseekPatcher` branch. Existing `deepseek_v3_attn_forward` at line ~3757 is the closest analogue.
4. `optimum/utils/input_generators.py` — a `DeepSeekV4DummyPastKeyValuesGenerator`. The trickiest piece: V4's per-layer different `compress_ratios` mean the KV cache layout is **not uniform across layers** — some layers have only sliding-window cache, some have window + ratio-4 compressed, some have window + ratio-128 compressed. `DeepSeekV3DummyPastKeyValuesGenerator` at line ~1191 only handles V3's MLA non-uniform-head-dim case; V4 needs per-layer cache shape support.
5. `optimum/exporters/openvino/utils.py` and `optimum/exporters/onnx/utils.py` — add `"deepseek_v4"` to the model lists at lines ~373 and ~42 respectively.

### Open questions for maintainers

1. **Per-layer KV cache shape.** Is there a precedent in `optimum-intel` for non-uniform KV cache structures across layers? If not, this would be a new pattern. I'm happy to draft the cache generator if there's a shape representation you'd want to standardize on.
2. **FP4 / FP8 microscaled weights.** OpenVINO 2026.1 doesn't yet have a runtime story for E8M0-microscaled FP4. The realistic path right now is to dequantize to BF16 on load (which I do in `scripts/load_real_v4_weights.py`). Is there a planned timeline for FP4 / E8M0 in OpenVINO that would change this calculus?
3. **Reasonable scope.** V4-Flash without MTP next-next-token blocks and without hash-routing tables is still a complete autoregressive LM. My port skips both (`num_nextn_predict_layers=0`, `num_hash_layers=0`). Is that an acceptable starting point for upstream, or would the first PR be expected to cover the full feature set?

### What I've already proven on a 64GB host

- Toy V4 (~1.34M params; sliding-window + ratio-4 indexer + ratio-128 dense + mHC + MoE) traces cleanly through `ov.convert_model`. IR loads + runs on CPU. PT/OV greedy tokens match for B=1.
- INT8 + INT4 weight compression via NNCF preserves greedy-token output (FP32 6.33 MB → INT8 3.03 MB → INT4 2.71 MB).
- `OVModelForCausalLM.from_pretrained(trust_remote_code=True, use_cache=False)` loads my pre-bundled IR end-to-end.
- FP4 (e2m1fn, 32-col E8M0 microscale) and FP8 (e4m3fn, 128×128 block) dequant verified on synthetic tensors. Name-mapping covers **67,569 / 69,187** keys in the published V4-Flash safetensors index, **1,618** explicit skips (MTP / hash routing / gate bias), **0** unmapped.
- Real V4-Flash full load gated only on RAM (~500 GB peak BF16; my host has 64 GB).

### What I'd ideally like

Either (or both) of:

- **(A)** Guidance on per-layer KV cache shape representation, after which I'm happy to draft the PR for `DeepSeekV4OnnxConfig` + `DeepSeekV4OpenVINOConfig` + the patcher + the cache generator.
- **(B)** Acknowledgement in the docs that the trust_remote_code + pre-converted IR pattern is the intended workflow for V4 until native support lands. My repo can serve as the reference implementation.

Happy to iterate on whichever direction is most useful.

### Repro

```bash
git clone https://github.com/bob798/deepseek-v4-openvino
cd deepseek-v4-openvino
python -m venv venv && source venv/bin/activate     # or .\venv\Scripts\activate on Windows
pip install torch==2.11.0 transformers==4.57.6 openvino==2026.1.0 optimum-intel==1.27.0 nncf
python tests/test_modeling_smoke.py
python scripts/convert_to_openvino.py
python scripts/quantize_with_nncf.py
python scripts/export_to_optimum_intel.py
python tests/test_dequant.py
python scripts/load_real_v4_weights.py --dry-run
```

Toy model end-to-end PoC runs in <30s on CPU.

---

## Target: `openvinotoolkit/openvino`

**Title:** DeepSeek-V4 conversion path: working PoC + open questions on FP4 / E8M0 / per-layer KV cache

**Labels (suggested):** `category: PT FE`, `feature`, `support_request`

**Body:**

(Same body as above, with the "What I'd ideally like" section reworded to emphasize the OpenVINO side.)

Specifically for the OpenVINO maintainers, two questions:

1. **FP4 (`e2m1fn`) and E8M0 microscaling**: is there a roadmap for native runtime support? If we could keep weights in FP4 + E8M0 instead of dequantizing to BF16 on load, the memory footprint of V4-Flash drops from ~500 GB to ~140 GB.
2. **Tracing behavior on graph-side Sinkhorn loops**: my mHC implementation contains a 20-iteration in-graph for-loop that `ov.convert_model` correctly unrolls. For V4 in production we'd want to investigate whether this can be expressed as a fused op or a Loop op for compilation efficiency. Any guidance welcome.

The PoC repo is the same: https://github.com/bob798/deepseek-v4-openvino
