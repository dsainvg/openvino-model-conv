# 把 Qwen3.6-35B-A3B 搬到 OpenVINO 上：一台 64GB 笔记本的复盘

V4-Flash 那篇复盘里我抱怨过：284B 的模型,这台 64GB 的笔记本(Intel Core Ultra 9 285H + Arc 140T iGPU)装不下,只能搭 toy 验证转换路径。Qwen3.6-35B-A3B 不一样——它总参 35B、激活才 3B,INT4 量化版整包 22.78GB,**这台机器装得下**。所以这次的目标不是"工作量证明",而是真的把它跑起来。

但"装得下"不等于"跑得动"。从权重到第一个 token,中间隔着三道墙。

## 一、第一道墙:工具链根本不认识它

第一反应当然是走标准路径:`optimum-cli export openvino --model ...`。直接报错——`KeyError: 'qwen3_5_moe'`。注意架构名是 `qwen3_5_moe`,不是 `qwen3_6`:Qwen 把 3.5/3.6 这一代都注册在 `3_5` 下。

往下查,三层都缺:
- `optimum-intel` 1.27.0 的 OpenVINO 导出器只注册到 `qwen3_moe`(上一代),没有 `qwen3_5_moe`。
- `transformers` 4.57(主 venv 的版本)根本没有这个模型类,得升到 5.5+。升到 5.9 之后 `AutoConfig` 能读了,但 `optimum-intel` 又被 `transformers<4.58` 钉死,两者不可能共存。
- GPTQ 权重要 `gptqmodel` 才能加载,而它在 Windows 上没有预编译 wheel,源码编译要 MSVC + cmake(`pypcre` 那一坨)。`auto-gptq` 对 Python 3.12 早就停更了。

结论很干脆:标准路径死透了,Phase 1 手工移植不可跳过。`gptqmodel` 装不上其实也不致命——GPTQ 的 int4 打包格式是公开的,我自己写了 40 行解量化(`gptq_dequant.py`),直接从 safetensors 出浮点权重。验证时发现 sym 量化的 zero 全是 7(gptqmodel 的 -1 偏移约定,有效 zero=8),scales 是 fp16,desc_act=False 时 g_idx 就是 `arange//128`。对着真权重一跑,数值合理,绕过了整个 GPTQ 运行时。

## 二、第二道墙:Gated DeltaNet 追踪不了

Qwen3.6 是 3:1 混合:每 4 层是 3 个 Gated DeltaNet(线性注意力)+ 1 个 GQA 注意力。注意力那半边好办,真正的硬骨头是 DeltaNet。

它本质是个类 Mamba 的 SSM:维护 running state,用 delta rule + 指数门控更新,前面还挂一个 causal conv1d。`transformers` 里它走的是 `flash-linear-attention` 和 `causal-conv1d` 两个 CUDA 内核,OpenVINO 追踪不了。好在源码里带了纯 torch 的 fallback(`torch_recurrent_gated_delta_rule`),这就是我移植的蓝本。

两个改写要点:
- **状态显式化**。`transformers` 把 conv_state / recurrent_state 塞在一个 `Cache` 对象里原地更新。图里没有"对象",所以我把每层的 `(conv_state, recurrent_state)` 全部提成显式的输入/输出张量——进去一份、出来一份,跟 V4 处理 KV cache 是同一招。
- **去掉所有 in-place**。KV cache 的写入不能用 `index_copy_`,改成 `torch.where` + one-hot 位置掩码。prefill / decode 双路径只保留 decode(seq_len=1)那条。

写完在 toy 上转 IR,`ov.convert_model` 0.6 秒过,跟 PyTorch 数值差 4.4e-4——float32 图融合的噪声地板。线性注意力这道墙,过了。

## 三、第三道墙:256 个专家塞不进一张图

MoE 才是这个项目的核心矛盾。256 个 routed expert,如果像 Phase 1 那样"全算 + 门控掩码"塞进单张 IR,图会被展开成 256 份专家子图——toy 的 4 专家没问题,真模型直接爆炸。

但每个 token 其实只激活 top-8。于是做了拆分:
- **backbone IR**(每层一张):注意力 + router + shared expert,**不含** routed expert。输出 router 选出的 top-k 索引、权重、shared 输出。
- **per-expert IR**:每个专家单独一张小图。
- **Python 编排**:backbone 跑完拿到 top-k → 只 dispatch 这 8 个专家 → 加权合并。

担心拆开会掉精度,结果反而更准:端到端对比单体 PyTorch,**最大误差 3.3e-7**——比单体 OV(4.4e-4)还小一个数量级,因为拆开后没有大图融合的浮点重排。10 张 IR(2 backbone + 8 expert)0.8 秒转完,一步 decode 10 毫秒。

专家计算量,理论上从 256 降到 8,full-scale **32x**;toy(32 专家 top-8)实测 2.85x(差距是注意力/router 的固定开销摊不掉)。这就是"35B 总参、3B 激活"在工程上真正兑现的地方。

## 四、缓存、量化、iGPU:把数字坐实

光有拆分还不够,得让它在真硬件上省下来。

**Expert LRU**。`ExpertManager` 懒加载 + LRU 淘汰 + pin 住热专家。toy 是随机初始化、路由近似均匀,所以命中率随容量爬:容量=top_k 时 0%,容量=单步工作集时 15%,全装下 89%。真模型路由是重度偏斜的,配合 calibration 预热(`ExpertFrequency.hot_keys → prewarm`)命中率会高得多——这点 V4 那边也印证过。

**量化 sweep**。把一个真专家解到 FP32,再用 NNCF 压到各精度,拿激活误差当 perplexity 代理(本机没法跑 WikiText-2):INT8 近乎无损(3.97x,cos 0.99988);INT4 几个变体挤在 5.5x / cos 0.994;MXFP4 最狠 5.69x,cos 0.989。

**iGPU**。最让我意外的是:backbone 整张图丢到 Arc 140T 上,**所有算子都被 GPU plugin 接住了**——DeltaNet 的递推、partial RoPE、`repeat_interleave`、`torch.where`、MoE router,没有一个不支持的。CPU↔GPU 数值差 4.3e-4。toy 上 GPU 反而慢(0.45x,小图被 dispatch 开销吃掉),但"能编译、数值对得上"本身就是 Phase 3.2 最想要的结论:真模型那条 2048 宽的稠密 backbone 放 iGPU、稀疏专家留 CPU,这条路是通的。

## 五、回头看

这次和 V4 最大的不同:V4 是"代码先于权重",这次是权重真的 plug 进来了——单层真权重(256 专家全解量化到 bf16)19 秒加载、forward 出来的 logits 分布合理,两种层类型都验证过。全模型 40 层约 13 分钟、占满 64GB,是 workstation 级别的一次性跑,没进 CI。

三道墙复盘下来,真正的产品不是"又一个能跑的模型",而是那套**backbone IR + per-expert IR + Python 编排**的拆分范式:它让"总参很大、激活很小"的 MoE 第一次能在消费级内存里按需展开。而且这套东西不是 Qwen 专属——`qwen3_next`、`olmo_hybrid` 用的是同一套线性注意力,V4 方向二用的是同一套专家拆分。一次移植,几处复用。

剩下的坑也记一笔:MXFP4 那条要不要做成真正的端侧 kernel、per-expert IR 能不能改成"权重当输入"的参数化单图(把 10,240 张专家图压成 1 张 + 缓存)、还有多模态那半边(vision tower + MTP)我这次直接没碰——纯文本优先。
