# TODO — DeepSeek-V4 on OpenVINO

Updated: 2026-05-26

Hardware: Intel Core Ultra 9 285H, 64GB RAM, Arc 140T iGPU

Upstream issues filed:
- huggingface/optimum-intel#1748
- openvinotoolkit/openvino#36071

---

## P0 — 64GB 本机可做，价值最高

### 1. Autoregressive decode（KV cache）
- 实现 `past_key_values` 输入/输出，支持逐 token 生成
- 难点：V4 的 KV cache 结构**逐层不同**（sliding-window / ratio-4 compressed / ratio-128 compressed）
- 改造 `modeling_deepseek_v4.py`：每个 attention 层返回各自形状的 KV cache
- 改造 `convert_to_openvino.py`：追踪时带 `past_key_values` 动态输入
- 移除 `use_cache=False` 限制，让 `OVModelForCausalLM` 支持真正的生成
- toy 模型上做 + 验证，不依赖真实权重
- **价值**：从 "能跑 prefill" 到 "能生成文本"，是项目实用性的分水岭

### 2. MXFP4 量化路径验证（toy 上）
- OpenVINO 2026.0 已支持 `CompressWeightsMode.E2M1`（MXFP4 + E8M0 microscale, CPU only）
- 改造 `scripts/quantize_with_nncf.py`：增加 MXFP4 量化路径
- 在 toy IR 上验证 MXFP4 压缩 + 推理正确性
- **价值**：为未来大机器上用原生 FP4（~140GB 而非 ~500GB）铺路

### 3. GPU 后端验证（Intel Arc 140T iGPU）
- 本机自带 Arc 140T，测试 `compile_model(model, "GPU")`
- 记录 iGPU 上的兼容性和性能差异
- **价值**：低成本验证，Intel AI PC 是 OpenVINO 的主推场景

---

## P1 — 等外部条件

### 4. optimum-intel 上游 PR
- 等 issue #1748 维护者回复 per-layer KV cache 的方案
- 按回复指引编写 `DeepSeekV4OnnxConfig` + `DeepSeekV4OpenVINOConfig` + patcher + cache generator
- 目标：`pip install optimum-intel` 后直接 `from_pretrained("deepseek-ai/DeepSeek-V4-Flash")`

### 5. 真实 V4-Flash 权重全量加载
- **阻塞条件**：需要 >=512GB RAM 机器（BF16 dequant），或 >=256GB + MXFP4 路径
- 方案 A：租云实例（AWS r7i.16xlarge 512GB ~$4/hr，一次转换 ~$10-20）
- 方案 B：找 Intel DevCloud / 社区合作者提供大内存环境
- 方案 C：实现分片逐层加载（改 `load_real_v4_weights.py`，逐层 dequant → 逐层写入 IR，峰值降到 ~60-80GB），可能在 64GB 上勉强可行但需要大量工程
- 转换完的 INT4 IR (~70-80GB) 可下载回本机，推理时 64GB 可能勉强够

### 6. 真实权重的 OpenVINO 推理验证
- 依赖 #5 完成
- `ov.convert_model` 追踪真实规模模型（43 层、4096 hidden、256 expert）
- 对比 PyTorch 和 OpenVINO 的 greedy output
- 记录追踪时间、IR 文件大小、推理延迟

---

## P2 — 锦上添花

### 7. 基准测试 & 文档
- 在真实权重上跑 perplexity benchmark（WikiText-2 或类似）
- 对比 FP32 / INT8 / INT4 / MXFP4 的 perplexity 和推理速度
- 更新 README 和 ARTICLE.md

### 8. MTP (Multi-Token Prediction)
- 可提升推理吞吐（speculative decoding 思路）
- 没有 MTP 也是完整的 autoregressive LM

### 9. Hash routing（前 3 层）
- 真实 V4-Flash 前 3 层用 hash routing 而非标准 MoE routing
- 用标准 routing 替代不影响正确性，只影响效率

---

## 执行顺序（64GB 本机）

```
1. KV cache / autoregressive decode（P0.1）← 现在开始
   - 改 modeling → 改 convert 脚本 → toy 上验证生成
2. MXFP4 量化验证（P0.2）
   - 升级 OpenVINO → 改 quantize 脚本 → toy 上验证
3. iGPU 验证（P0.3）
   - compile_model("GPU") → 记录结果
4. 等 upstream 回复 → PR（P1.4）
5. 租云机器跑真实权重（P1.5-6）
```
