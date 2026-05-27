# Qwen3.6-35B-A3B on OpenVINO — 独立实验计划

Created: 2026-05-27

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

## Phase 0：环境 & 可行性验证（1-2 天）

### 0.1 下载模型
```bash
# 官方暂无 Qwen3.6 GPTQ-Int4，直接下载 BF16 全量（~70GB）
huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir ./qwen36-35b
# 或用第三方 GPTQ-Int4（~18GB，palmfuture 社区量化版）
huggingface-cli download palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 --local-dir ./qwen36-35b-int4
```

### 0.2 标准路径试跑
- [ ] `optimum-intel` 直接 export：`optimum-cli export openvino --model Qwen/Qwen3.6-35B-A3B`
- [ ] 如果失败，定位原因：DeltaNet 不支持？MoE routing 不支持？
- [ ] 记录错误信息，评估需要多少自定义工作
- **预期**：OpenVINO 2026.0 支持 Qwen3-30B-A3B（旧架构），但 3.6 的 DeltaNet 可能不行

### 0.3 Transformers 推理验证
- [ ] `transformers` 直接加载，跑一次推理确认输出正确
- [ ] 记录内存占用、推理时间作为 baseline
- [ ] 分析模型结构：`model.named_modules()` 导出层级

---

## Phase 1：架构移植（如需要）（3-5 天）

> 如果 Phase 0.2 标准路径成功，跳过此阶段。

### 1.1 分析不可追踪的组件
- [ ] Gated DeltaNet 的 running state 更新逻辑能否被 `ov.convert_model` 追踪？
- [ ] MoE routing 的 scatter/gather 能否追踪？
- [ ] 识别需要重写为"图友好"版本的算子

### 1.2 纯 PyTorch 等价重写
- [ ] 类似 V4 项目的做法：用图友好的算子替换不可追踪的部分
- [ ] Gated DeltaNet：如果 running state 不能追踪，考虑展开为显式矩阵运算
- [ ] MoE routing：复用 V4 的 "compute-all + gate matrix" 方案（临时）
- [ ] 在小 config 上验证数值一致性

### 1.3 OpenVINO IR 转换
- [ ] `ov.convert_model` 追踪
- [ ] IR save/load + CPU 推理
- [ ] 数值对比 PyTorch vs OpenVINO

---

## Phase 2：MoE 高效推理引擎（核心创新）（5-7 天）

### 2.1 模型拆分
- [ ] 分离 backbone（DeltaNet layers + Attention layers + shared expert）和 routed experts
- [ ] backbone → 单一 IR（常驻内存，~3-5GB INT4）
- [ ] 每个 routed expert → 独立小 IR（~2MB INT4 each, expert_dim=512 很小）
- [ ] 验证拆分后数值 == 整体模型

### 2.2 选择性 Expert 计算
- [ ] Router 先行：输入 hidden state → router → top-8 expert indices
- [ ] 只加载 + 计算 8 个 routed expert + 1 shared expert
- [ ] 跳过其余 248 个 expert（vs 朴素方案计算全部 256 个）
- [ ] 理论加速：~28x expert 计算量减少

### 2.3 Expert 缓存策略
- [ ] LRU cache：最近使用的 expert 留在内存
- [ ] 预热：统计 calibration 数据上的 expert activation 频率，预加载 hot experts
- [ ] 冷 expert 走磁盘 mmap

### 2.4 推理 pipeline
```python
# 伪代码
backbone = ov.compile_model("backbone.xml", "CPU")
expert_cache = LRUCache(max_size=64)  # 64 个 expert 常驻

for token in input_tokens:
    # 1. backbone forward（DeltaNet/Attention + router）
    hidden, router_logits = backbone(token, past_state)
    
    # 2. top-8 expert selection
    top_indices = router_logits.topk(8)
    
    # 3. load + run selected experts
    expert_outputs = []
    for idx in top_indices:
        expert = expert_cache.get_or_load(idx)
        expert_outputs.append(expert(hidden))
    
    # 4. weighted sum + shared expert
    output = gate_combine(expert_outputs) + shared_expert(hidden)
```

---

## Phase 3：量化 & 性能优化（3-5 天）

### 3.1 量化实验
- [ ] FP32 → INT8 → INT4 → MXFP4（复用 V4 项目的 NNCF 流程）
- [ ] 每种精度的 perplexity（WikiText-2）
- [ ] 每种精度的 tok/s

### 3.2 性能 benchmark
- [ ] 对比：朴素全 expert vs 选择性计算
- [ ] 对比：OpenVINO vs transformers (PyTorch)
- [ ] 对比：CPU only vs CPU + iGPU（backbone on iGPU, experts on CPU）
- [ ] 首 token 延迟 (TTFT) + 生成吞吐 (tok/s)

### 3.3 多模态验证
- [ ] 图片理解推理（Qwen3.6 原生支持）
- [ ] 记录 vision encoder 的开销

---

## Phase 4：Demo & 文档（2-3 天）

### 4.1 交互式 demo
- [ ] CLI 文本生成 demo
- [ ] 图片理解 demo（输入图片 + 问题）
- [ ] 实时显示：哪些 expert 被激活、cache 命中率、tok/s

### 4.2 文档 & 文章
- [ ] README：how to run、benchmark 结果、架构图
- [ ] 技术文章：类似 V4 的 ARTICLE.md，讲解 MoE 高效推理的工程思路
- [ ] 发布到 HuggingFace（预转换的 IR）

### 4.3 Upstream 贡献
- [ ] 如果 DeltaNet 需要自定义移植 → 向 optimum-intel 提 issue/PR
- [ ] 如果 MoE 选择性计算有通用性 → 向 openvino.genai 提建议

---

## 项目结构（在现有 repo 内）

```
deepseek-v4-openvino/
  src/
    deepseek_v4/               # 已有，V4 架构
    qwen36/                    # 新增，Qwen3.6 架构（如需自定义移植）
      modeling_qwen36.py
      configuration_qwen36.py
    engine/                    # 新增，通用 MoE 推理引擎（V4 + Qwen3.6 共用）
      expert_manager.py        # Expert 拆分 + 加载 + LRU cache
      router.py                # Router 计算 + top-k 选择
      pipeline.py              # 推理 pipeline 编排
  scripts/
    qwen36_convert.py          # Qwen3.6 OV 转换
    qwen36_benchmark.py        # 性能测试
    qwen36_demo.py             # 交互式 demo
  tests/
    test_qwen36_smoke.py
    test_engine.py             # engine 通用测试
```

**共享模块**：
- `src/engine/` — V4 方向二和 Qwen3.6 共用同一套 MoE 推理引擎
- NNCF 量化流程直接复用
- V4 先验证架构移植，Qwen3.6 先验证 engine，两边互补

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
Phase 0 (1-2天)    确认可行性，找到阻塞点
     ↓
Phase 1 (3-5天)    架构移植（仅在标准路径失败时）
     ↓
Phase 2 (5-7天)    MoE 高效推理（核心价值）
     ↓
Phase 3 (3-5天)    量化 + benchmark
     ↓
Phase 4 (2-3天)    Demo + 文档 + 发布
```

总工期估计：2-3 周（Phase 1 可能跳过）
