# dpv4-openvino

将 **DeepSeek-V4 (V4-Flash)** 架构移植到 **OpenVINO** 的概念验证 (PoC)。
使用纯 PyTorch 端到端地构建一个随机初始化的小尺寸 toy 模型,通过
`openvino.convert_model` 完成图追踪,导出为 IR,然后重新加载并在 CPU 上推理。
同一套 Python 模块在显存/内存足够的环境下可以直接加载真正的 V4-Flash 权重。

[English README](README.md) — [Hugging Face: `imbob798/deepseek-v4-toy-int4-ov`](https://huggingface.co/imbob798/deepseek-v4-toy-int4-ov)

## 当前状态

| 项目 | 状态 |
| --- | --- |
| Toy V4 PyTorch 前向 | 通过,logits 全部为有限值 |
| `ov.convert_model` 图追踪 | 通过 |
| OpenVINO IR 在 CPU 上保存 / 加载 | 通过 |
| PyTorch 与 OpenVINO 的贪心采样一致性 (B=1) | 一致 |
| 动态 shape (`(1,64) (1,128) (1,256) (2,128)`) | 全部可运行 |
| B=2 数值一致性 | 有 1 个 token 漂移(FP 舍入顺序差异,非拓扑问题) |
| 通过 NNCF 做 INT8 / INT4 权重压缩 | 在 toy IR 上通过,贪心 token 与 FP32 一致 |
| `optimum-intel` 的 `OVModelForCausalLM.from_pretrained` | 通过(toy,`use_cache=False`) |
| 真实 V4-Flash 权重加载器 (FP4 + FP8 反量化) | 代码已写;反量化逻辑用合成张量验证;名字映射 100% 覆盖真实 V4 的 key |
| 真实 V4-Flash 完整加载 | 未运行 — 需要约 500 GB 内存(本机 64 GB) |

约定的 PoC 完成标准是 **"OpenVINO IR 能够加载并运行而不崩溃"** —— 该标准已达成,
另外还做了与 PyTorch 的数值一致性自检。

## 移植覆盖了哪些 V4 特性

参考实现位于 `v4_flash_meta/inference/model.py`(从
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
下载),依赖 TileLang JIT 内核以及 OpenVINO 无法直接追踪的 FP4/FP8 微缩放数据类型。
本次移植将这些替换成纯 PyTorch 等价实现,同时保持架构形状完全一致:

- **混合稀疏注意力**: 滑动窗口 (size=128) 加上索引器驱动的、针对压缩 KV 的 top-k
  (`compress_ratio=4`),以及对 `compress_ratio=128` 压缩 KV 的稠密注意力,通过配置
  `compress_ratios` 字段逐层选择。稀疏注意力以"稠密 gather + softmax"实现,以保证
  图结构对编译器友好。
- **多潜在注意力 (MLA)**: Q/O LoRA 分解,MQA(单 KV 头),attention sink 被合并到
  softmax 分母中。
- **KV 压缩器 (Compressor)**: 对 `compress_ratio` 个连续 token 做带门控的池化
  (ratio==4 时窗口重叠),仅 prefill,不跨调用保留状态。
- **索引器 (Indexer)**: 一个独立的、类似注意力的子网络,从压缩 KV 序列中挑选 top-k 位置。
- **流形约束的 Hyper-Connections (mHC)**: 隐状态在每一层都携带 `hc_mult=4` 份并行副本,
  通过图内 Sinkhorn 循环重新组合 (toy 用 `hc_sinkhorn_iters=4`,真实 V4-Flash 用 20)。
- **MoE**: 真实 V4-Flash 是 256 个路由专家 + 1 个共享专家,top-6 选择,采用
  `sqrtsoftplus` 打分与 `noaux_tc` topk。实现上采用"全部专家一起算"的策略,通过 scatter
  构建一个稠密的 `[N, E]` 门控矩阵,从而图中没有 Python 端调度。
- **YaRN RoPE 缩放** (真实 V4-Flash factor=16,从 65k → 1M 上下文;toy factor=1)。

**未移植**的部分:TileLang 内核、FP4 / FP8 / E8M0 微缩放 GEMM、MTP next-next 预测块、
前 3 层的 hash routing(toy 中可配置为 0)。

## 仓库结构

```
src/deepseek_v4/
  configuration_deepseek_v4.py   # 包含全部 V4 字段的 PretrainedConfig
  modeling_deepseek_v4.py        # ~720 行,inference/model.py 的纯 PyTorch 移植
  __init__.py
tests/
  test_modeling_smoke.py         # toy 配置 + PyTorch 前向冒烟测试
  test_ov_dynamic_shapes.py      # 在与追踪 shape 不同的输入上运行 IR
  test_dequant.py                # FP4/FP8 反量化与名字映射的单元测试
scripts/
  convert_to_openvino.py         # PyTorch -> ov.convert_model -> 保存 IR -> CPU 运行 + 对比
  quantize_with_nncf.py          # 用 nncf.compress_weights 把 FP32 IR 压到 INT8 / INT4
  export_to_optimum_intel.py     # 保存 HF 目录并打包 IR -> OVModelForCausalLM.from_pretrained
  load_real_v4_weights.py        # 真实 V4 -> 我们的命名:FP4/FP8 反量化 + 名字映射 (--dry-run)
  fetch_v4_meta.py               # 下载 HF V4-Flash 仓库的元数据
  probe_v4_repos.py              # 快速探测 HF 仓库的小工具
v4_flash_meta/                   # 镜像的 HF 元数据 + 参考实现 (deepseek MIT 许可)
ov_ir_toy/
  deepseek_v4_toy.xml/.bin       # 为 toy 模型生成的 OpenVINO IR
```

## 环境准备

测试环境: Windows 11, Intel Core Ultra 9 285H + Arc 140T iGPU, 64 GB 内存, Python 3.12。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch==2.11.0 transformers==4.57.6 openvino==2026.1.0 optimum-intel==1.27.0 numpy
```

## 运行 PoC

```powershell
# 1. 在 toy 模型 (~134 万参数) 上跑 PyTorch 冒烟测试
python tests\test_modeling_smoke.py

# 2. 追踪并转换为 OpenVINO IR,再加载、在 CPU 上推理,并与 PyTorch 结果对比
python scripts\convert_to_openvino.py

# 3. 验证保存的 IR 可以在与追踪时不同的 shape 上运行
python tests\test_ov_dynamic_shapes.py

# 4. 用 NNCF 把 IR 压到 INT8 / INT4,并对比与 FP32 的数值与体积
python scripts\quantize_with_nncf.py

# 5. 以 HF 格式保存模型并打包 IR,通过 optimum-intel 加载
python scripts\export_to_optimum_intel.py

# 6. (面向真实 V4) 验证 FP4/FP8 反量化与名字映射
python tests\test_dequant.py
python scripts\load_real_v4_weights.py --dry-run
```

第 2 步会同时打印 PyTorch 和 OpenVINO 的 logits,以及贪心采样的下一个 token 是否一致。
IR 会被写到 `ov_ir_toy/deepseek_v4_toy.xml`(以及对应的 `.bin`)。

## Toy 配置

为了让 PoC 在 CPU 上几秒钟内就能跑完,toy 模型刻意做得很小:

| 字段 | Toy | 真实 V4-Flash |
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
| 参数量 | ~1.34 M | ~284 B(约 13 B 激活) |

4 层组合 `[0, 0, 4, 128]` 可以覆盖到所有的注意力路径: 纯滑动窗口、窗口 + 索引器驱动的稀疏
压缩、以及窗口 + 稠密压缩。

## 真实 V4 权重加载器

`scripts/load_real_v4_weights.py` 读取真实 V4-Flash 检查点中的
`model.safetensors.index.json`,把每个 key 映射到我们的参数名,并按 shard 把
FP4 专家权重(e2m1fn,32 列一份的 E8M0 微缩放)与 FP8 主干权重(e4m3fn,
128×128 块缩放)反量化为 BF16。

- 反量化逻辑由 `tests/test_dequant.py` 用合成张量做单元测试覆盖。
- `--dry-run` 仅读取 index 文件、检查命名覆盖。在真实 V4-Flash 的 index 上当前给出:
  67,569 个 key 映射到我们的参数,1,618 个落在显式跳过名单(MTP 块、hash routing
  表、路由门偏置),0 个未映射。
- **完整加载在本机不会跑**: BF16 反量化后大约需要 500 GB 内存峰值,而本机 64 GB。

## 已知限制

- **本机只能用 toy 权重**。加载器代码已写好,但完整加载真实 V4 大约需要 500 GB 内存峰值。
  在足够大的机器上的入口是
  `python scripts/load_real_v4_weights.py --weights-dir <V4-Flash 目录>`。
- **B=2 数值漂移**。追踪时使用的是 B=1,以 B=2 运行 IR 时会出现 1 个贪心 token 漂移,
  原因是浮点累加顺序差异,不是拓扑问题。
- **仅支持 prefill**。Compressor 与 KV cache 不会跨调用保留;尚未为自回归解码接入
  `past_key_values`,因此 `OVModelForCausalLM` 需要以 `use_cache=False` 加载。
- **未实现 MTP 与 hash routing**。Multi-Token Prediction 块以及前 3 层的 hash-routing
  表在加载真实 V4 时被跳过(配置中 `num_nextn_predict_layers=0`、`num_hash_layers=0`)。

## 致谢

`v4_flash_meta/` 镜像了
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
中的文件,该仓库以 MIT 许可证发布(见 `v4_flash_meta/LICENSE`)。
`src/deepseek_v4/` 中的 PyTorch 移植代码为本仓库原创,遵循参考架构。
