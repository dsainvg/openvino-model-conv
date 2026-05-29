# Qwen3.6-35B-A3B on OpenVINO — 独立实验计划

Created: 2026-05-27
Updated: 2026-05-28

## 背景

Qwen3.6-35B-A3B（2026.4.16 发布）是目前最新的开源 MoE 模型之一。
DeepSeek-V4-Flash 方向二已在推进中，本计划作为**独立的并行实验**。

### 为什么选这个模型

| 维度 | Qwen3.6-35B-A3B | DeepSeek-V4-Flash |
|------|-----------------|-------------------|
| 总参数 | 35B | 284B |
| 激活参数 | 3B | 13B |
| Expert 数 | 256 (top-8 + 1 shared) | 256 (top-6 + 1 shared) |
| INT4 内存 | ~18GB | ~140GB |
| 64GB 能全装？ | **Yes，余量充裕** | No |
| 架构 | **Gated DeltaNet + MoE（全新）** | Hybrid Sparse Attn + mHC |
| 多模态 | 原生图片/视频 | 纯文本 |
| 社区热度 | 高 | 低 |
| OpenVINO 支持 | Qwen3 有，**3.6 未验证** | 无 |
| 许可 | Apache 2.0 | MIT |

### 架构要点

```
每 4 层的布局（3:1 混合）：
  3 × (Gated DeltaNet → MoE)    ← 线性注意力，O(n) 复杂度
  1 × (Gated Attention → MoE)    ← 传统 GQA 注意力

Gated DeltaNet：
  - 不建 attention matrix，维护 running state（类 RNN）
  - 32 V heads + 16 QK heads, head_dim=128
  - Delta rule + 指数门控 + Causal Conv1D + L2 norm

MoE：
  - 256 routed experts, expert_intermediate_dim=512
  - top-8 routed + 1 shared = 9 active
  - 每 token 只用 9/257 ≈ 3.5% 的 expert 参数

其他：
  - 262K 原生上下文，YaRN 扩展到 1M
  - 训练用 MTP（multi-step token prediction）
  - vocab_size=248,320
```

---

## 实验目标

**在 64GB Intel AI PC 上实现 Qwen3.6-35B-A3B 的高效 MoE 推理。**

具体目标：
1. 验证 OpenVINO 能否转换/运行 Qwen3.6 的 DeltaNet + MoE 架构
2. 实现 expert 选择性计算（只算 top-9，跳过其余 247 个 expert）
3. 量化到 INT4/MXFP4，benchmark 推理速度和质量
4. 产出可展示的推理 demo（文本生成 + 多模态）

---

## Phase 0：环境 & 可行性验证 ✅ 2026-05-27

### 0.1 下载模型 ✅
```bash
# 使用第三方 GPTQ-Int4（~18GB，palmfuture 社区量化版）
huggingface-cli download palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 --local-dir ./qwen36-35b-int4
```

### 0.2 标准路径试跑 ✅
- [x] `optimum-intel` 直接 export — **失败**，DeltaNet (Gated DeltaNet) 不被 OV 追踪支持
- [x] 定位原因：DeltaNet 的 running state 更新 + causal conv1d step 需要手工移植
- [x] 结论：需要 Phase 1 手工重写
- 探测脚本：`scripts/test_config.py`, `scripts/test_load.py`,
  `scripts/dump_layer_keys.py`, `scripts/inspect_gptq.py`

### 0.3 Transformers 推理验证 ✅
- [x] 分析模型结构：识别 40 层 (3:1 linear:full pattern)，256 experts，GPTQ INT4 量化格式
- [x] 确认 GPTQ 权重格式：gptqmodel 6.0.3, sym=True, group_size=128, pack_dtype=int32

---

## Phase 1：架构移植 ✅ 2026-05-27

> Phase 0.2 标准路径失败（DeltaNet 不可追踪），执行手工移植。

### 1.1 分析不可追踪的组件 ✅
- [x] Gated DeltaNet：running state (conv_state + recurrent_state) 需要函数式 IO
- [x] MoE routing：scatter/gather 用 compute-all + mask 方案替代
- [x] 所有 in-place 操作替换为 functional update（torch.where 替代 index_copy_）

### 1.2 纯 PyTorch 等价重写 ✅
- [x] `src/modeling.py` (605 行) — 完整 OV 可追踪移植：
  - `QwenGatedDeltaNet`: causal conv1d step + delta rule 循环，所有状态作为函数 IO
  - `QwenAttention`: GQA + partial RoPE + sigmoid 输出门控 + 显式 KV cache
  - `QwenMoEBlock`: compute-all + mask（Phase 1 方案，OV 可追踪）
  - `QwenDecoderLayer`: 支持 full_attention / linear_attention 两种层类型
  - `QwenForCausalLM`: decode-only forward，所有状态显式 IO
- [x] `src/configuration.py`: 配置 dataclass + `make_toy_config()` + `from_pretrained_dir()`
- [x] 在 toy config (2 层, 4 experts) 上验证：7 个测试用例通过 (`tests/test_toy_match.py`)

### 1.3 OpenVINO IR 转换 ✅
- [x] `scripts/convert_toy.py`: toy 模型 → OV IR（FlatWrapper 展平状态列表）
- [x] IR save/load + CPU 推理验证
- [x] 数值对比 PyTorch vs OpenVINO 通过

---

## Phase 2：MoE 高效推理引擎（核心创新） 🔧 IN PROGRESS

### 2.1 模型拆分 ✅ 2026-05-27
- [x] `src/split_inference.py` (254 行)：
  - `QwenLayerBackboneFull` / `QwenLayerBackboneLinear` — 每层 backbone wrapper（router + shared expert，不含 routed experts）
  - `moe_router_step()` / `moe_combine_step()` — 路由 + 聚合分离
  - `extract_expert_state_dict()` / `build_standalone_expert()` — 单 expert 提取 + 重建
  - `monolithic_step_via_split()` — PyTorch 参考实现，验证 split path == monolithic
- [x] 验证拆分后数值 == 整体模型（diff < 1e-4）
- [x] 测试：`tests/test_split_inference.py` (4 个用例，包含 OV roundtrip)

### 2.2 选择性 Expert 计算 ✅ 2026-05-27
- [x] `scripts/split_orchestrator_toy.py`: per-layer backbone IR + per-expert IR 端到端编排
- [x] Router 先行 → 只 dispatch top-K expert → combine
- [x] 在 toy 上验证 split-OV == monolithic-PT

### 2.3 Expert 缓存策略 ✅ 2026-05-28
- [x] `src/expert_manager.py`：`ExpertManager` LRU cache（capacity + pinning + hit/miss/eviction stats）
- [x] 预热：`ExpertFrequency` 统计 calibration expert 激活频率，`hot_keys()` + `prewarm(pin=True)` 预加载 hot experts
- [x] 缓存对输出透明（capacity=2 与 unbounded 生成结果字节一致），LRU 命中率随 capacity 提升（0% → 89%）
- 注：冷 expert mmap 未做（已有 lazy loader callback，磁盘后端可后续接入）

### 2.4 真实权重加载 ✅ 2026-05-27
- [x] `src/gptq_dequant.py` (188 行)：GPTQ INT4 解量化
  - `unpack_qweight()` / `unpack_qzeros()` — int32 → int4 位解包
  - `dequantize_gptq()` — (raw - effective_zero) * scale，sym=True 时 zero=8
  - `load_gptq_linear()` — 从 safetensors 按 prefix 加载并 dequant
  - 测试：`tests/test_gptq_dequant.py` (6 个用例，含真实权重 integration test)
- [x] `src/load_weights.py` (209 行)：完整权重映射 + 流式加载
  - 支持 linear_attention / full_attention 两种层类型的权重加载
  - 自动 GPTQ dequant routed experts，plain load backbone 权重
  - `load_real_model()` 支持按子集层加载（省内存）
  - 测试：`tests/test_load_real_weights.py` (4 个用例，1-layer smoke test)

### 2.5 推理 pipeline ✅ 2026-05-28
- [x] `src/pipeline.py`：`Qwen36Pipeline.generate()` 完整多层 autoregressive decode（prefill + greedy/temperature 采样）
- [x] step() 经 split path + ExpertManager，数值匹配 monolithic forward（< 1e-4）
- [x] toy 端到端生成验证（5 个用例）；真实权重端到端 40 层生成需 ~13 min 加载（workstation run，未在 CI 跑）

---

## Phase 3：量化 & 性能优化 ✅ 2026-05-28

### 3.1 量化实验 ✅ 2026-05-28
- [x] Int4 (GPTQ) vs FP32 expert 压缩比：8x（理论）/ 3.07x（整 checkpoint 22.78GB vs ~70GB BF16）
- [x] `scripts/quant_sweep.py`：真专家 FP32 → INT8 / INT4_SYM / INT4_ASYM / NF4 / MXFP4 NNCF sweep
  - INT8 3.97x cos 0.99988；INT4 ~5.5x cos 0.994；MXFP4 5.69x cos 0.989（激活误差，perplexity 代理）
- [ ] perplexity（WikiText-2）— 本机无参考（gptqmodel 无 Windows wheel，未下 BF16），用激活误差代理

### 3.2 性能 benchmark ✅ 2026-05-28
- [x] `scripts/benchmark.py`：朴素全 expert vs 选择性计算（toy 2.85x，full-scale 理论 32x）
- [x] 首 token 延迟 (TTFT) + 生成吞吐 (tok/s)
- [x] LRU 命中率 vs capacity sweep
- [x] `scripts/device_split.py`：CPU only vs backbone=iGPU/experts=CPU
  - 关键结论：移植图在 Arc 140T 全算子被接住、CPU↔GPU 数值差 4.3e-4；toy 小图 GPU 受 dispatch 开销（0.45x），真模型 2048 宽 backbone 才受益

### 3.3 多模态验证
- [ ] scoped out：vision tower + MTP head 未移植（纯文本 port）

---

## Phase 4：Demo & 文档 ✅ 2026-05-28

### 4.1 交互式 demo ✅ 2026-05-28
- [x] `scripts/demo.py`：CLI 文本生成 demo（toy + --real 模式）
- [x] 实时显示：每层哪些 expert 被激活、cache 命中率、tok/s、hottest experts
- [ ] 图片理解 demo — scoped out（纯文本 port）

### 4.2 文档 & 文章 ✅ 2026-05-28
- [x] `src/README.md`：how to run、架构、module map、benchmark 结果、status
- [x] `ARTICLE-QWEN36.md`：技术复盘（三道墙 + split 范式 + 实测数字）
- [x] 发布到 HuggingFace：https://huggingface.co/imbob798/qwen36-35b-openvino-moe-split（24 文件，源码+脚本+测试+文档）

### 4.3 Upstream 贡献 ✅ 2026-05-29
- [x] `UPSTREAM_ISSUE-QWEN36.md`：optimum-intel（qwen3_5_moe export）+ openvino.genai（MoE split-IR 范式）双草稿
- [x] optimum-intel issue 已提交：huggingface/optimum-intel#1754
- [x] openvino.genai issue 已提交：openvinotoolkit/openvino.genai#3917

---

## 项目结构（实际）

```
openvino-model-lab/
  qwen36/
    src/                           # Qwen3.6 架构（已实现）
      __init__.py
      configuration.py             # ✅ 配置 dataclass + toy/real loader
      modeling.py                  # ✅ OV 可追踪的完整模型（605 行）
      split_inference.py           # ✅ backbone/expert 分离 + Python 编排（254 行）
      gptq_dequant.py              # ✅ GPTQ INT4 解量化（188 行）
      load_weights.py              # ✅ 真实权重加载器（209 行）
    scripts/
      test_config.py               # ✅ Phase 0 探测
      test_load.py                 # ✅ Phase 0 探测
      dump_layer_keys.py           # ✅ Phase 0 探测
      inspect_gptq.py              # ✅ Phase 0 探测
      convert_toy.py               # ✅ Toy → OV IR 转换
      split_orchestrator_toy.py    # ✅ Split-IR 端到端 demo
    tests/
      test_toy_match.py            # ✅ 7 个用例
      test_split_inference.py      # ✅ 4 个用例
      test_gptq_dequant.py         # ✅ 6 个用例
      test_load_real_weights.py    # ✅ 4 个用例（需真实权重）
  deepseek-v4/                     # V4 架构
    src/
```

---

## 风险 & 决策点

| 风险 | 应对 |
|------|------|
| DeltaNet 无法被 OV 追踪 | Phase 1 手工重写，参考 V4 项目经验 |
| optimum-intel 已原生支持 3.6 | Phase 0.2 就能发现，跳过 Phase 1，直接做 Phase 2 |
| Expert 太小（dim=512），拆分开销 > 收益 | 改为按 group 打包（每组 16 个 expert 一个 IR） |
| 内存不够跑 BF16 全量 | 用第三方 INT4 GPTQ (palmfuture/)，或本地 NNCF 量化 |

---

## 执行顺序

```
Phase 0 (1 天) ✅   环境探测，确认 DeltaNet 不可追踪
     ↓
Phase 1 (1 天) ✅   手工移植 + OV IR 转换 + toy 验证
     ↓
Phase 2 (1 天) ✅   Split-IR + 真实权重加载 + LRU cache + autoregressive pipeline
     ↓
Phase 3 ✅         量化 sweep + benchmark + CPU/iGPU split
     ↓
Phase 4 ✅         Demo + README + 技术文章 + upstream 草稿
```

实际进度远超预期：Phase 0-4 全部在两天内完成（原计划 2-3 周）。
剩余非核心项：HF IR 发布（需先登录 HF）、多模态、weight-as-input 参数化单 expert IR、perplexity 正式评测。
Upstream issues 已提交：optimum-intel#1754, openvino.genai#3917。
