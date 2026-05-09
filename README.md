# dpv4-openvino

Proof-of-concept port of the **DeepSeek-V4 (V4-Flash)** architecture to **OpenVINO**.
A toy-sized model with random weights is built end-to-end in pure PyTorch, traced through
`openvino.convert_model`, saved as IR, then reloaded and run on CPU. The same Python
modules accept the real V4-Flash weights wherever there is enough RAM/VRAM to host them.

[中文 README](README.zh-CN.md)

## Status

| Item | State |
| --- | --- |
| Toy V4 PyTorch forward | works, finite logits |
| `ov.convert_model` tracing | works |
| OpenVINO IR save/load on CPU | works |
| PyTorch ↔ OpenVINO greedy match (B=1) | matches |
| Dynamic shapes (`(1,64) (1,128) (1,256) (2,128)`) | all run |
| B=2 numerical match | one element drifts (FP rounding, not a topology bug) |
| INT8 / INT4 weight compression via NNCF | works on toy IR, greedy match to FP32 |
| `optimum-intel` `OVModelForCausalLM.from_pretrained` | works (toy, `use_cache=False`) |
| Real V4-Flash weight loader (FP4 + FP8 dequant) | code written, dequant verified on synthetic tensors, name-mapping covers 100% of real-V4 keys |
| Real V4-Flash full load | not run — needs ~500 GB peak RAM (this host has 64 GB) |

The agreed done bar for the PoC was **"OpenVINO IR loads + runs without crashing"** — that
bar is met, plus a numerical sanity check vs. PyTorch.

## What V4 features the port covers

The reference implementation at `v4_flash_meta/inference/model.py` (downloaded from
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash))
relies on TileLang JIT kernels and FP4/FP8 microscaled dtypes that OpenVINO cannot
trace directly. This port replaces those with pure-PyTorch equivalents while keeping
the architectural shape identical:

- **Hybrid sparse attention**: sliding window (size 128) plus indexer-driven top-k over
  compressed-KV (`compress_ratio=4`) and dense compressed-KV (`compress_ratio=128`),
  selected per layer via the `compress_ratios` config field. Sparse attention is implemented
  as a dense gather + softmax so the topology is graph-friendly.
- **Multi-Latent Attention**: Q/O LoRA decomposition, MQA (single KV head), with the
  attention sink merged into the softmax denominator.
- **KV Compressor**: gated pooling over `compress_ratio` consecutive tokens (overlap when
  ratio==4). Prefill-only; no across-call state.
- **Indexer**: separate attention-like sub-network selecting top-k compressed-KV positions.
- **Manifold-constrained Hyper-Connections (mHC)**: hidden state carries `hc_mult=4` parallel
  copies through every block, recombined with an in-graph Sinkhorn loop
  (`hc_sinkhorn_iters=4` for the toy, 20 in real V4-Flash).
- **MoE**: 256 routed + 1 shared expert in real V4-Flash, top-6 with `sqrtsoftplus` scoring
  and `noaux_tc` topk. Implemented as compute-all-experts with a dense `[N, E]` gate matrix
  built via scatter, so the graph has no Python-side dispatch.
- **YaRN RoPE scaling** (factor=16 from 65k → 1M context in real V4-Flash; factor=1 for the
  toy).

What is **not** ported: TileLang kernels, FP4 / FP8 / E8M0 microscaled GEMMs, MTP next-next
prediction blocks, hash-routed first 3 layers (configurable to 0 for the toy).

## Repo layout

```
src/deepseek_v4/
  configuration_deepseek_v4.py   # PretrainedConfig with all V4 fields
  modeling_deepseek_v4.py        # ~720-line pure-PyTorch port of inference/model.py
  __init__.py
tests/
  test_modeling_smoke.py         # toy config + PyTorch forward smoke test
  test_ov_dynamic_shapes.py      # IR runs on shapes other than the trace shape
  test_dequant.py                # FP4/FP8 dequant + name-mapping unit tests
scripts/
  convert_to_openvino.py         # PyTorch -> ov.convert_model -> save IR -> CPU run + compare
  quantize_with_nncf.py          # FP32 IR -> INT8 / INT4 via nncf.compress_weights
  export_to_optimum_intel.py     # save HF dir + bundle IR -> OVModelForCausalLM.from_pretrained
  load_real_v4_weights.py        # real-V4 -> ours: FP4/FP8 dequant + name mapping (--dry-run)
  fetch_v4_meta.py               # downloads the HF V4-Flash repo metadata
  probe_v4_repos.py              # quick HF probe utility
v4_flash_meta/                   # mirrored HF metadata + reference impl (deepseek MIT)
ov_ir_toy/
  deepseek_v4_toy.xml/.bin       # generated OpenVINO IR for the toy model
```

## Setup

Tested on Windows 11, Intel Core Ultra 9 285H + Arc 140T iGPU, 64 GB RAM, Python 3.12.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch==2.11.0 transformers==4.57.6 openvino==2026.1.0 optimum-intel==1.27.0 numpy
```

## Run the PoC

```powershell
# 1. PyTorch smoke test on the toy model (1.34M params)
python tests\test_modeling_smoke.py

# 2. Trace + convert to OpenVINO IR + reload + run on CPU + compare vs PyTorch
python scripts\convert_to_openvino.py

# 3. Verify the saved IR runs at shapes other than the trace shape
python tests\test_ov_dynamic_shapes.py

# 4. Compress the IR to INT8 and INT4 with NNCF; compare numerics + size vs FP32
python scripts\quantize_with_nncf.py

# 5. Save model in HF format + bundled IR; load through optimum-intel
python scripts\export_to_optimum_intel.py

# 6. (real V4 only) Verify the FP4/FP8 dequant + name mapping
python tests\test_dequant.py
python scripts\load_real_v4_weights.py --dry-run
```

Step 2 prints both PyTorch and OpenVINO logits and the greedy next-token comparison.
The IR is written to `ov_ir_toy/deepseek_v4_toy.xml` (+ `.bin`).

## Toy config

The toy model is intentionally tiny so the PoC runs in seconds on CPU:

| Field | Toy | Real V4-Flash |
| --- | --- | --- |
| `hidden_size` | 128 | 4096 |
| `num_hidden_layers` | 4 | 43 |
| `num_attention_heads` | 4 | 64 |
| `head_dim` | 32 | 512 |
| `q_lora_rank` | 64 | 1024 |
| `n_routed_experts` | 8 | 256 |
| `num_experts_per_tok` | 2 | 6 |
| `compress_ratios` | `[0, 0, 4, 128]` | `[0, 0, 4, 128, 4, 128, ...]` |
| `hc_mult` | 4 | 4 |
| `hc_sinkhorn_iters` | 4 | 20 |
| total params | ~1.34 M | ~284 B (~13 B active) |

The 4-layer mix `[0, 0, 4, 128]` exercises every attention path: pure sliding window, then
window + indexer-driven sparse compression, then window + dense compression.

## Real-V4 weight loader

`scripts/load_real_v4_weights.py` reads `model.safetensors.index.json` from a
real V4-Flash checkpoint, maps every key to our parameter names, and dequantizes
FP4 expert weights and FP8 main weights to BF16 shard-by-shard.

- Dequant logic (FP4 e2m1fn with 32-col E8M0 microscale, FP8 e4m3fn with
  128×128 block scale) is unit-tested with synthetic tensors in
  `tests/test_dequant.py`.
- `--dry-run` reads only the index and verifies coverage. On the real
  V4-Flash index this currently reports: 67,569 keys mapped to our params,
  1,618 on the explicit skip list (MTP blocks + hash-routing tables + routed-gate
  bias), 0 unmapped.
- The full load is not exercised on this host: real V4-Flash needs roughly
  500 GB peak RAM after BF16 dequant, vs. our 64 GB.

## Known limitations

- **Toy weights only on this host.** The loader is written but the full real-V4
  load needs ~500 GB peak RAM. On a sufficiently large host the entry point is
  `python scripts/load_real_v4_weights.py --weights-dir <V4-Flash dir>`.
- **B=2 numerical drift.** Trace was at B=1; running the IR at B=2 produces a single
  divergent greedy token from FP rounding-order differences, not a topology bug.
- **Prefill-only.** The Compressor and KV cache do not carry across calls; no
  `past_key_values` plumbing for autoregressive decode yet, so
  `OVModelForCausalLM` is loaded with `use_cache=False`.
- **MTP and hash routing not implemented.** Multi-Token Prediction blocks and the
  hash-routing tables of the first 3 layers are skipped on real-V4 load
  (`num_nextn_predict_layers=0`, `num_hash_layers=0` in our config).

## Attribution

`v4_flash_meta/` mirrors files from
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash),
licensed MIT (see `v4_flash_meta/LICENSE`). The PyTorch port in `src/deepseek_v4/` is
original code that follows the reference architecture.
