# TODO — DeepSeek-V4 on OpenVINO

Updated: 2026-05-26

Hardware: Intel Core Ultra 9 285H, 64GB RAM (shared with iGPU), Arc 140T iGPU, NPU

Upstream issues filed:
- huggingface/optimum-intel#1748
- openvinotoolkit/openvino#36071

本项目分两个方向并行推进。

---

# 方向一：Toy PoC 完善（架构验证 + upstream 贡献）

目标：把 toy 模型从 "能跑 prefill" 做到 "功能完整的 HF 集成"，推动 upstream 合入。

## P0 — 立刻可做

### 1.1 Autoregressive decode（KV cache） ✅ 2026-05-26
- ✅ `modeling_deepseek_v4.py`: per-layer Block-input cache via `past_key_values`
  (list of `[B, S_past, hc_mult, dim]`). Block always takes the concat path so the
  same IR serves both prefill (S_past=0) and decode (S_past>0). Attention, Indexer,
  and the topk-index helpers all take `seqlen_total / seqlen_new`.
- ✅ `scripts/convert_to_openvino_kv.py`: traces L past inputs + L present outputs,
  all dimensions dynamic. CPU prefill abs_max vs PT = 5.8e-2, greedy match.
- ✅ `tests/test_kv_cache.py`: PyTorch prefill vs step-by-step decode equivalence
  (abs_max 5e-7, greedy match across all decode steps).

### 1.2 MXFP4 量化路径验证（toy 上） ✅ 2026-05-26
- ✅ `scripts/quantize_with_nncf.py`: added `CompressWeightsMode.MXFP4` (E2M1 + E8M0
  microscale, group_size=32). Toy sizes — FP32 6.46 MB / INT8 3.17 / INT4 2.85 /
  **MXFP4 2.66 MB**; greedy top matches FP32 across all four modes.

## P1 — 等外部条件

### 1.3 optimum-intel 上游 PR
- 等 issue #1748 维护者回复 per-layer KV cache 方案
- 编写 `DeepSeekV4OnnxConfig` + `DeepSeekV4OpenVINOConfig` + patcher + cache generator
- 目标：`pip install optimum-intel` 后直接 `from_pretrained("deepseek-ai/DeepSeek-V4-Flash")`

### 1.4 真实权重全量加载（需大内存机器）
- 租云实例（AWS r7i.16xlarge 512GB ~$4/hr，一次转换 ~$10-20）
- 或 Intel DevCloud / 社区协作
- 转换完的 INT4 IR (~70-80GB) 下载回本机

## P2 — 锦上添花

### 1.5 MTP (Multi-Token Prediction)
- 没有 MTP 也是完整的 autoregressive LM，优先级低

### 1.6 Hash routing（前 3 层）
- 用标准 routing 替代不影响正确性，优先级低

---

# 方向二：Intel AI PC 推理引擎（64GB 跑真实 V4-Flash）

目标：在 64GB 消费级笔记本上运行 284B 参数的 V4-Flash，利用 OpenVINO + NPU 做出 ds4/llama.cpp 做不到的差异化方案。

## 完成的基础验证

### 2.0 Arc 140T iGPU 后端兼容性 ✅ 2026-05-26
- ✅ `scripts/bench_igpu.py`: compile + run every IR on CPU and Arc 140T iGPU,
  writes per-record JSON to `ov_ir_toy/igpu_bench_results.json`.
- ✅ 12/12 records OK. CPU↔GPU greedy match on every IR (FP32 / INT8 / INT4 / MXFP4 / KV).
- ✅ iGPU prefill 比 CPU 快 2-5×（MXFP4 prefill: 33.6 ms CPU → 8.8 ms GPU — 是所有
  组合中最快的）。单 token decode 在 iGPU 上反而比 CPU 慢约 2×（开销摊不开）。
- ✅ CPU↔GPU 数值漂移 abs_max ~5e-2 到 1.5e-1（iGPU FP16 路径的典型差异）。
- **结论**：iGPU 作为方向二的 backbone 后端是可行的（prefill 快），decode 阶段
  小 batch 选择留给 CPU/NPU。

## 核心思路

V4-Flash 有 256 个 routed expert，每个 token 只激活 top-6（~2.3%）。
**95% 以上的权重在任意时刻是冷的。** 不需要全部装进内存。

```
┌──────────── NPU (常驻, ~几MB) ─────────┐
│  Expert Predictor：预测下一层 top-6     │
│  → 异步通知 CPU prefetch               │
│  低功耗，不占 CPU/内存带宽             │
└────────────────────────────────────────┘
          ↓ prefetch signal
┌──────────── CPU + RAM (~55GB) ─────────┐
│  Backbone (embed + attn + mHC): ~5GB   │
│  Shared expert: ~0.5GB                 │
│  Hot experts LRU cache: ~40GB INT4     │
│  Cold expert INT2 fallback: ~10GB      │
└────────────────────────────────────────┘
          ↕ cache miss → async load
┌──────────── NVMe SSD (~130GB) ─────────┐
│  全部 256 experts @ INT4               │
│  + INT2 shadow copies for fast fallback│
└────────────────────────────────────────┘
```

差异化优势（vs ds4 / llama.cpp）：
- **NPU expert predictor**：零 CPU 开销的预测 + prefetch，Intel AI PC 独有
- **混合精度 expert**：hot INT4 + cold INT2，内存占用更低
- **OpenVINO 原生**：直接用 `compiled_model`，不需要手写 GEMM kernel
- **speculative prefetch**：在当前层计算时异步加载下一层 expert

## P0 — 基础设施 ✅ DONE (2026-05-26)

### 2.1 模型拆分：backbone + expert 分离 ✅
- ✅ `scripts/split_to_expert_irs.py`: emits embed.xml + per-layer pre_moe/post_moe
  + per-expert IR + final.xml (42 IRs total for the toy: 1 + 4*pre + 32 experts +
  4*post + 1).
- ✅ Backbone segments hold (gate + shared expert + HC + attention) — routed
  experts are fully isolated.
- ✅ Orchestrator validates numerical equivalence vs monolithic PT (greedy match;
  abs diff in line with monolithic OV's drift from PT).
- ✅ Found and fixed a pre-existing init bug: `nn.Parameter(torch.empty(...))` for
  HC mix matrices and Gate.weight was uninitialized memory, causing non-deterministic
  forward outputs across runs. Extended `_init_weights` to cover those.
- **关键论文参考**：HOBBIT (arXiv:2411.01433)

### 2.2 Expert offloading 基础版 ✅
- ✅ `scripts/run_with_expert_offload.py`: persistent backbone (embed + per-layer
  pre/post + final), compile-on-miss + LRU-evict for routed experts.
- ✅ Bit-exact match with the monolithic OV IR (abs_max = 0.0) — the split
  layout doesn't introduce its own numerical drift.
- ✅ With LRU size = 2 and 8 active experts per layer, the orchestrator triggers
  30 evictions across 32 expert loads on the toy. Stats: hits/misses/evictions
  per layer reported.

### 2.3 真实权重分片转换 ✅
- ✅ `scripts/load_real_v4_weights.py --per-expert-ir`: new streaming mode that
  loads + dequantizes ONE expert at a time, builds an Expert nn.Module, traces
  with `ov.convert_model`, writes the IR, frees memory. Backbone collected into
  a single BF16 safetensors at the end. Output layout matches 2.1's so 2.2's
  orchestrator can drive either source without changes.
- ✅ Peak RAM ~5-10 GB (backbone + one expert) — fits on this 64 GB host. The
  old `--full-bf16` mode (~500 GB peak) is preserved for documentation; only
  `--dry-run` runs on this host without real weights.
- ✅ Dry-run still passes (69187 keys, 67569 mapped, 1618 skipped, 34167 paired
  FP4/FP8 weights). Tests/test_dequant.py still passes.

## P1 — 性能优化

### 2.4 混合精度 Expert（HOBBIT 路线） ✅ 2026-05-27
- ✅ `scripts/mixed_precision_experts.py`: end-to-end pipeline in one script
  (Steps 1-4 fused for the toy validation).
  - **Step 1 — calibrate**: runs N random inputs through the split orchestrator,
    counts per-(layer, expert) selections. Writes
    `ov_ir_toy/expert_split_mixed/calibration.json` with per_layer_counts +
    tier assignment so a downstream NPU predictor / warm-cache loader can
    consume the same stats without re-running calibration.
  - **Step 2 — quantize**: applies NNCF per-expert. Toy uses INT8 hot / INT4
    cold (NNCF 3.1.0 lacks an INT2 mode, so this is the two-tier proxy on
    this stack; swapping cold → MXFP4 / NF4 / packed INT2 is a one-line
    `CompressWeightsMode` change once the lower-precision variant is wired).
  - **Step 3 — orchestrate**: reuses 2.2's pattern; the mixed-precision IRs
    live in `ov_ir_toy/expert_split_mixed/` alongside copies of the FP32
    backbone segments so a single directory drives everything.
  - **Step 4 — validate**: greedy match vs monolithic OV.
- ✅ Toy result: 16 hot (INT8) + 16 cold (INT4) per the 4-layer × 8-expert toy.
  Per-expert size FP32 ≈ 102 KB → INT8 hot ≈ 41 KB / INT4 cold ≈ 36 KB.
  Total experts: 3261 KB FP32 → 1227 KB mixed (**62.4% reduction**).
  Logits abs diff vs monolithic max 1.84e-1 mean 7.5e-3; greedy top match.
- 真实 V4-Flash 推算：calibration 在真实权重上会有非常不均匀的分布
  （toy 随机权重 8 个 expert 都 ~256 selections，所以 hot/cold 选择是任意的；
  真实模型预期 hot:cold ≈ 10:1 频次比）。
- **关键论文**：MxMoE (arXiv:2505.05799), MoPEQ (arXiv:2509.02512)

### 2.5 Speculative Expert Prefetch ✅ 2026-05-27
- ✅ `scripts/speculative_prefetch.py`: per-layer linear predictor
  (`y2_flat → next layer gate logits`), trained with MSE against real
  next-layer gate weights collected from calibration. Each predictor exports
  to its own ~4 KB OV IR under `ov_ir_toy/expert_split_predictors/`.
- ✅ Orchestrator runs predictor[i] at layer i and `ThreadPoolExecutor` kicks
  off `core.compile_model()` for the predicted top-K experts of layer i+1
  in the background while layer i's dispatch runs.
- ✅ Bit-exact output vs no-prefetch baseline — prefetch only hides compile
  latency; the dispatch still uses the REAL gate output, never the predicted
  one.
- ✅ Toy results: recall@2 = 0.77 / 0.81 / 0.83 for L1/L2/L3 predictors;
  warm-hit rate 37.5%, prefetch waste 0% (with top-4 prediction over 8-expert
  toy where every expert is active). predictor_stats.json saved for 2.6.
- 对真实 V4-Flash 推算：top-6 of 256 比 top-2 of 8 选择性强很多，预期 recall
  会下降但 waste 会上升；warm-hit 仍能显著减少 cold-start 延迟。
- **关键论文**：DALI (arXiv:2602.03495), SP-MoE (arXiv:2510.10302)

### 2.6 NPU Expert Predictor
- 把 2.5 的 predictor 部署到 NPU
- OpenVINO `compile_model(predictor, "NPU")`
- NPU 低功耗常驻推理，不抢 CPU 资源和内存带宽
- **价值**：Intel AI PC 独有能力，竞品做不到

## P2 — 进阶

### 2.7 Expert Pruning（可选）
- 分析 router activation 统计，识别冗余 expert
- 256 → 64 experts，MoE 参数量砍 75%
- 需要 calibration + perplexity 评估
- **关键论文**：MoE-Pruner (arXiv:2410.12013), SlimMoE (arXiv:2506.18349)

### 2.8 Benchmark & 文档
- perplexity 对比（full precision vs 混合精度 vs pruned）
- 推理速度 benchmark（tok/s, 首 token 延迟）
- 与 ds4 / llama.cpp 的对比
- 更新 README 和 ARTICLE.md

---

# 执行顺序

```
方向一（toy PoC）         方向二（推理引擎）
──────────────          ──────────────
1.1 KV cache ✅          2.0 iGPU 兼容性 ✅
1.2 MXFP4 验证 ✅        2.1 模型拆分 ✅
      |                 2.2 Expert offloading ✅
      |                 2.3 真实权重分片转换 ✅
1.3 upstream PR ← next   2.4 混合精度 ✅
      |                 2.5 Speculative prefetch ✅
1.4 云机器全量转换        2.6 NPU predictor ← next
```

两个方向在 toy 阶段可以共享代码（modeling、config、权重映射），
方向二的 2.1-2.2 可以复用方向一的 KV cache 成果。
