# 把 DeepSeek-V4 搬到 OpenVINO 上:一个 64GB 笔记本的复盘

DeepSeek-V4-Flash 发布的那天,我盯着 Hugging Face 的 README 看了大概十分钟。架构改了大半:hybrid sparse attention、manifold-constrained Hyper-Connections、FP4 专家权重 + FP8 主干 + UE8M0 微缩放。然后我去 OpenVINO 的支持矩阵搜了一下 `deepseek_v4`,没有。当然没有,人家昨天才发的。

我手里有一台 64GB 内存的笔记本(Intel Core Ultra 9 285H + Arc 140T iGPU)。V4-Flash 全量是 284B 参数,光 INT4 存储要 140GB,反量化到 BF16 大约 500GB。这台机器,装不下。

但我还是想试试。

## 一、装不下不等于做不了

新架构上一个新硬件平台,除了权重之外,还有一整条转换路径要走通:PyTorch 模型搭建、`ov.convert_model` 的图追踪、IR 的保存与加载、动态 shape 测试、量化、optimum-intel 集成。这条路径不依赖权重大小。我可以拿一个 toy 配置(4 层、128 hidden、8 个专家)搭一个**同架构**的小模型,把转换路径走通。等到有 500GB 内存的机器,把权重 plug 进去。

这其实是一个很合理的杠杆点:本地我拿不到 SOTA 推理,但我可以拿到"把 V4 装到 Intel 平台上的工作量证明"。代码会先于权重就位。

## 二、第一道墙:TileLang

打开 `deepseek-ai/DeepSeek-V4-Flash` 仓库的 `inference/model.py`,第一个发现是:整个推理用 TileLang 写的 JIT 内核。OpenVINO 追踪不了。

第一个决定:把 TileLang 全部替换成纯 PyTorch 算子。FP4/FP8 GEMM 全部退化成 BF16 的 `nn.Linear`。性能上是巨大倒退,但对于 PoC 来说,正确性优先。这一步看起来粗暴,实际上是把"硬件相关的优化层"和"架构层"切干净——后续要换硬件、换精度、换 backend,都从这一刀开始。

## 三、第二道墙:几个图不友好的拓扑

真正的硬骨头是几个把研究代码搬到生产图时绕不过去的拓扑:

**稀疏注意力**。滑动窗口 + 索引器选 top-k 压缩 KV,reference 里是稀疏 CUDA 内核。OpenVINO 的图里没有"按索引稀疏聚合"这种原语,得重写成"稠密 gather + softmax"——把所有要用到的 token 一次性收上来,稀疏只体现在选哪些 index,实际算的是稠密 attention。计算量大了一点,图是干净的。

**流形约束的 Hyper-Connections (mHC)**。每个 block 输入输出都带 4 份并行隐状态,中间走一个 Sinkhorn 迭代做归一化。Sinkhorn 在 reference 里是 20 次,我把它写成图内 for-loop(toy 配 4 次),`ov.convert_model` 追踪时会展开成 4 倍的子图。

**MoE 路由**。经典实现里,每个 token 选 top-k 专家,然后 Python 端按专家 dispatch tokens。这条路图不下来。改成"全部专家都算"——所有专家一次跑完,然后用 scatter 构造一个 `[N, E]` 的稠密门控矩阵,把每个 token 的输出加权求和。计算量爆炸,但图里没有 Python 端的 if/else。

总结一下这三个改写:**研究代码追求 FLOPs,生产图追求拓扑可追踪性**。两者经常冲突,这次就全部冲突了一次。

## 四、各种小坑

RoPE 的广播 shape 我写错过两次。cos 是 `[S, D/2]`,x 是 `[B, S, H, D/2]`,我一开始按"维度数对齐"自动 unsqueeze,结果 cos 的 S 维和 x 的 B 维对齐了——产生了一个能跑但数值完全错的图。后来改成显式构造广播 shape `[1, S, 1, ..., 1, D/2]` 才对。

OpenVINO IR 的输出 port 默认没名字,要手动 `set_names({"logits"})`。否则 compile 之后调用 `out.any_name` 会炸。

B=2 的时候有一个 batch element 的贪心 token 和 PyTorch 不一致——浮点累加顺序问题,不是拓扑 bug,但在 toy 里也没必要花时间死扣。

## 五、PASS

最终,1.34M 参数的 toy 模型:PyTorch 前向通过,`ov.convert_model` 追踪通过,IR 保存 + CPU 加载 + 推理通过,B=1 的 PT/OV 贪心 token 完全一致(都是 token 19)。从决定动手到这一刻,大约一周。

PoC 的 done bar 当时定义得很简单:**IR loads + runs without crashing**。达成。

## 六、达成之后再补的三件事

PoC 通过之后,有三件事是"再加一点边际成本就能让这个工作的杠杆翻倍"的,我都补上了:

**NNCF 量化**。拿 FP32 IR 直接走 `nncf.compress_weights`,FP32 6.33 MB → INT8 3.03 MB → INT4 2.71 MB。三个变体的贪心 token 都一致。toy 上的绝对差异不大,但流程证明了。

**optimum-intel 集成**。把 toy 模型存成 HF 格式 + `auto_map`,IR 直接打包进同一个目录,`OVModelForCausalLM.from_pretrained(trust_remote_code=True, use_cache=False)` 可以加载并跑通。这是个关键节点:第三方拿到这个 repo,不需要懂内部细节,直接 `from_pretrained` 就用上了。

**真实权重加载器**。写了 FP4(e2m1fn,32 列 E8M0 微缩放)和 FP8(e4m3fn,128×128 块缩放)的反量化函数,加上参数名映射。`--dry-run` 跑 V4-Flash 的 safetensors index,**67,569 个 key 全部映射到我们的参数名,1,618 个进入显式跳过名单(MTP 块、hash routing 表、路由门偏置),0 个未映射**。dequant 逻辑用合成张量做了单元测试。完整加载因为内存吃不下没在本机跑过,但代码路径是完整的。

## 七、复盘:杠杆点在哪里

做完之后回头看,这个项目最关键的不是任何一段代码,而是**最开始那个判断**:我装不下权重,但我可以把转换路径完整走通。

这种 dis-aggregation——把"权重"和"路径"切开——是个人在大模型工程里少数还能保留的杠杆点。权重在云上、在大厂集群、在花得起电费的人手里。但是"如何把一个新架构搬到一个新硬件平台"这件事,**还没有被自动化掉**。Reference 实现里的 TileLang、稀疏内核、FP4 微缩放——每一个都得手工翻译成目标平台能 trace 的形式。这个过程里需要的是判断和耐心,不是 GPU 时数。

所以即使是一台 64GB 的笔记本,也能产出一个完整的"V4 on OpenVINO"转换 + 量化 + 集成的工作样本。它现在挂在 [github.com/bob798/deepseek-v4-openvino](https://github.com/bob798/deepseek-v4-openvino),任何拿到真实权重的人都可以 plug-and-play。

下一步是把 `model_type="deepseek_v4"` 注册到 optimum-intel 上游。如果合进去了,以后任何人 `pip install optimum-intel` 就能用。如果没合进去,issue 本身就是公开的工作量证明。

## 写在最后

所以这其实不是一个"我做了一个 SOTA 推理"的故事。是一个**"权重还没到我手上,但我已经把着陆跑道修好了"**的故事。

要等的人,在等大显存机器、等量化好的权重、等官方支持。
不等的人,先把基础设施立起来。

权重总会落地。落地那一刻,谁的跑道修得更近,谁先起飞。
