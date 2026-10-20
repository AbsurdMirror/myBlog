---
title: Layer 1：从 Engram 写入到局部注意力与 MoE
status: published
updated: 2026-10-20
---

# Layer 1：从 Engram 写入到局部注意力与 MoE

## 1. 相对 Layer 0，只看两处差异

本文分析 **DeepSeek-V4.1-Flash 的 Layer 1**。Layer 0 已经完整讲过 mHC、RMSNorm、局部 Attention / SWA 和 MoE；Layer 1 不再重复这些推导，而是回答两个新问题：

1. Engram 怎样在普通 Transformer Block 之前，根据当前 token 及其局部历史检索条件记忆，并写回四路残差？
2. Layer 1 的第一次 `hc_pre` 为什么不再使用固定的 $[1,0,0,0]$，而是使用 Layer 0 MoE 侧传来的动态 `pre_next`？

记

$$
N=BS,\qquad D=5120,\qquad C=4,\qquad W=128.
$$

Layer 1 从 Layer 0 接收：

$$
X^{(1)}_{\mathrm{in}}:[B,S,4,5120]\quad\text{(BF16)},
\qquad
p^{(0)}_{\mathrm{next}}:[B,S,4]\quad\text{(FP32)}.
$$

其中 $X^{(1)}_{\mathrm{in}}$ 是四路残差，$p^{(0)}_{\mathrm{next}}$ 是 Layer 0 的 MoE 侧 mHC 生成的下一次汇聚系数。完整数据流为：

$$
\boxed{
(X^{(1)}_{\mathrm{in}},p^{(0)}_{\mathrm{next}})
\rightarrow \mathrm{Engram}
\rightarrow \mathrm{Layer\ 1\ Block}
\rightarrow (X^{(1)}_{\mathrm{out}},p^{(1)}_{\mathrm{next}})
}
$$

逐步对照如下。“结构相同”只表示算子、shape 和执行顺序与 Layer 0 相同；Layer 1 拥有自己的参数，输入数值也已经被 Engram 改变，并不与 Layer 0 共享权重或计算结果。

| 步骤 | 相对 Layer 0 | Layer 1 的行为 |
|---|---|---|
| 1. Engram Hash、查表、门控与四路写回 | **新增** | 在普通 Block 前修改四路残差；不修改传入的 `pre_next` |
| 2. Attention 侧 mHC 系数生成 | 结构相同 | 从 Engram 更新后的四路残差预测本层 `attn_pre/post/comb` |
| 3. Attention 入口 `hc_pre` | **输入不同** | 使用 Layer 0 MoE 侧生成的动态 `pre_next`，不再使用 identity-pre |
| 4. Attention 入口 RMSNorm | 结构相同 | 独立参数，输入来自步骤 3 |
| 5. Local Attention / SWA | 结构相同 | 仍是 128-token 局部窗口，没有 compressed/global KV 或 Indexer |
| 6. Attention 侧 `hc_post` | 结构相同 | 融合 Attention 输出和 Engram 更新后的四路残差 |
| 7. MoE 侧 mHC 系数生成 | 结构相同 | 生成本层 `ffn_pre/post/comb` |
| 8. MoE 入口 `hc_pre` | 结构相同 | 使用本层 Attention 侧生成的 `attn_pre` |
| 9. MoE 入口 RMSNorm | 结构相同 | 独立参数 |
| 10. MoE FFN | 结构相同 | 384 个 routed experts 选 6 个，并执行 1 个 shared expert |
| 11. MoE 侧 `hc_post` | 结构相同 | 输出四路残差，并把 `ffn_pre` 作为下一层的 `pre_next` |

固定配置中 `compress_ratios[1]=0`，而 `kv_source_layers` 与 `index_source_layers` 都从 Layer 2 开始。因此 Layer 1 与 Layer 0 一样只执行 SWA；本页新增的注意力外路径只有 Engram。[配置核查](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/config.json)

## 2. Engram

### 2.1 从 Token 到 24 个查表结果

![图 1：Layer 1 的 Token 压缩、n-gram Hash 与 Engram 查表](assets/13_layer1_engram_lookup.png)

图 1 从左到右依次经过 Token 压缩、历史补齐、$N$-gram Hash、分区寻址和参数表查找。图中的蓝色块表示随 batch 和序列变化的张量，黄色块表示固定 buffer 或模型参数，绿色块表示计算步骤。

主要对象如下：

| 对象 | Shape | 含义 |
|---|---|---|
| $T$ | $[B,S]$ | 原始 Token ID |
| $T_{\mathrm{map}}$ | $[129280]$ | 原始词表到压缩词表的固定映射 |
| $G_0,G_1,G_2,G_3$ | 各为 $[B,S]$ | 当前 token 与最近三个历史 token |
| $H_{\mathrm{off}},H_{\mathrm{mod}}$ | 各为 $[3,8]$ | 三种 $N$-gram、八个 Hash head 的 offset 与模数 |
| $I_E$ | $[B,S,24]$ | 每个 token 的 24 个查表地址 |
| $E_{\mathrm{table}}$ | $[384006168,256]$ | Layer 1 的 Engram 参数表 |
| $E_{\mathrm{flat}}$ | $[B,S,6144]$ | 24 行查表结果展平后的输出 |

计算分为五步。

#### Token 压缩

原始 Token ID 的取值范围来自大小为 129,280 的词表。Hash 之前先通过固定映射 $T_{\mathrm{map}}$ 得到压缩 ID：

$$
T_{c,b,t}=T_{\mathrm{map}}\!\left[T_{b,t}\right],
\qquad
T_c:[B,S].
$$

固定配置把原始 Token 映射到 99,092 个压缩 ID。映射构建时会对能够完整解码的 token 文本执行 Unicode 规范化、去音调、转小写、空白折叠和首尾空白处理；规范化结果相同的 token 共用压缩 ID。$T_{\mathrm{map}}$ 是固定运行时 buffer，不是可学习参数。

#### 组织当前 token 与历史 token

图中的 $G_0$ 表示当前位置的压缩 Token ID，$G_1,G_2,G_3$ 依次向历史回看一步、两步和三步。Prefill 与 Decode 可能分属不同次前向，因此实现还维护压缩 Token 历史 cache，使当前前向能够读取此前已经写入的历史。

记 $G_{b,t,r}$ 为位置 $t$ 向前回看 $r$ 步得到的输入，$r=0,1,2,3$：

$$
G_{b,t,r}=
\begin{cases}
T_{c,b,t-r},&
t-r\ge 0\ \land\ [t-r,t]\text{ 内全部有效},\\
t_{\mathrm{pad}},&
\text{其他}.
\end{cases}
$$

序列开头缺少的历史使用固定 $t_{\mathrm{pad}}$。多模态输入中的图像 span 会在压缩缓存中写入无效标记；任何跨过无效位置的 $N$-gram 都会被阻断并改用 $t_{\mathrm{pad}}$，避免把文本与图像位置拼成错误的局部短语。

#### 构造 2、3、4-gram Hash

Layer 1 为四个回看位置准备四个固定奇数乘数 $\mu_0,\mu_1,\mu_2,\mu_3$。先逐位置相乘，再通过 XOR 累积，分别得到 2、3、4-gram Hash：

$$
H_{b,t,n}
=
\bigoplus_{r=0}^{n-1}
\left(G_{b,t,r}\mu_r\right),
\qquad n\in\{2,3,4\}.
$$

对应关系是：

$$
\begin{aligned}
H_{b,t,2}
&=(G_{b,t,0}\mu_0)\oplus(G_{b,t,1}\mu_1),\\
H_{b,t,3}
&=H_{b,t,2}\oplus(G_{b,t,2}\mu_2),\\
H_{b,t,4}
&=H_{b,t,3}\oplus(G_{b,t,3}\mu_3).
\end{aligned}
$$

因此，同一个当前 token 会同时保留二元、三元和四元局部上下文，而不是只使用最长的 4-gram。

#### 生成 24 个查表地址

每种 $N$-gram 都进入八个 Hash head。对阶数 $n\in\{2,3,4\}$ 和 head $q\in\{0,\ldots,7\}$，地址为：

$$
I_{E,b,t,n,q}
=
H_{\mathrm{off},n,q}
+
\left(
H_{b,t,n}\bmod H_{\mathrm{mod},n,q}
\right).
$$

$H_{\mathrm{mod},n,q}$ 是该分区使用的质数模数，$H_{\mathrm{off},n,q}$ 是它在大表中的起始位置。不同 $(n,q)$ 使用互不重叠的地址区间，因此逻辑上可以理解为 24 张小表，参数存储上则拼接为一张连续大表。

把三种阶数与八个 head 展平：

$$
I_E:[B,S,3,8]\rightarrow[B,S,24].
$$

#### 查表并展平

Layer 1 的 Engram 参数表为：

$$
E_{\mathrm{table}}:[384006168,256].
$$

每个地址读取一行 256 维向量：

$$
E_{b,t,n,q,:}
=
E_{\mathrm{table}}
\left[
I_{E,b,t,n,q},:
\right],
$$

因此查表结果为：

$$
E:[B,S,3,8,256]
\rightarrow
[B,S,24,256].
$$

最后把 24 行沿特征维展平：

$$
\boxed{
E_{\mathrm{flat}}:[B,S,6144]
},
\qquad
6144=3\times8\times256.
$$

至此，图 1 的输出已经包含当前 token 对应的 24 份局部短语记忆。这里的“24 次查表”是逻辑行读取口径；固定 revision 的官方推理实现会分别读取 FP8 数据与对应 scale，不能直接把逻辑行数等同于底层访存指令数。

在官方推理实现中，模型会在进入各层循环之前，同时生成 Layer 1 和 Layer 14 所需的 Hash 地址：

$$
I_{\mathrm{all}}:[B,S,2,24].
$$

执行 Layer 1 时只取第一个切片 $I_{\mathrm{all}}[:,:,0,:]$。图 1 为了说明 Layer 1 的算子数据流，只画出了这个 $[B,S,24]$ 切片。

### 2.2 从查表结果到四路残差写回

![图 2：Layer 1 Engram 的 KV 投影、四路 Gate 与残差写回](assets/14_layer1_engram_computation.png)

图 2 接续图 1 的输出 $E_{\mathrm{flat}}$。查表结果先投影为四路 Key 和一份共享 Value，再与四路残差分别计算 Gate，最后把同一份 Value 按不同幅度写回四路。

主要输入和参数如下：

| 对象 | Shape | 含义 |
|---|---|---|
| $E_{\mathrm{flat}}$ | $[B,S,6144]$ | 图 1 得到的 24 行查表结果 |
| $X$ | $[B,S,4,5120]$ | Layer 0 输出的四路残差 |
| $W_E$ | $[25600,6144]$ | 生成 Key 和 Value 的投影权重，无 bias |
| $W_Q,W_K$ | 各为 $[4,5120]$ | 四路 Gate 使用的逐维缩放参数 |
| $T_{\mathrm{mask}}$ | $[B,S]$ | 可选的文本位置 mask；计算时广播到四路 |

计算分为四步。

官方推理实现将 Engram 表的 FP8 数据按 scale 还原为 BF16。进入 Gate 分支后，$X,K_E,V_E,W_Q,W_K$ 显式转换为 FP32 参与计算；最终输出 $Y$ 再转换回 $X$ 的 dtype，本配置中为 BF16。

#### 生成四路 Key 和共享 Value

查表结果与 $W_E$ 相乘：

$$
F=E_{\mathrm{flat}}W_E^{\mathsf T},
\qquad
F:[B,S,25600].
$$

沿最后一维按 $20480+5120$ 切分：

$$
[K_E,V_E]
=\operatorname{split}(F,[20480,5120]),
$$

$$
K_E:[B,S,4,5120],
\qquad
V_E:[B,S,5120].
$$

$K_E$ 包含四路不同的 Key，$V_E$ 只有一份共享 Value。Engram 不会为四路分别生成四份 Value；四路写入差异来自后面各自计算的 Gate。

#### 计算四路 RMS 缩放因子

两组可学习参数先逐元素相乘：

$$
W=W_Q\odot W_K,
\qquad
W:[4,5120].
$$

对四路残差 $X$ 和四路 Key $K_E$ 分别沿 5120 维计算 RMS 倒数。对残差流 $c\in\{0,1,2,3\}$：

$$
r^X_{b,t,c}
=
\left(
\frac{1}{5120}
\sum_{d=0}^{5119}X_{b,t,c,d}^{2}
+\epsilon
\right)^{-1/2},
$$

$$
r^K_{b,t,c}
=
\left(
\frac{1}{5120}
\sum_{d=0}^{5119}K_{E,b,t,c,d}^{2}
+\epsilon
\right)^{-1/2},
\qquad
\epsilon=10^{-20}.
$$

$r^X$ 与 $r^K$ 的 Shape 都是 $[B,S,4]$。它们只是当前输入动态生成的缩放因子，没有额外的可学习 RMSNorm 权重。

#### 由加权点积生成 Gate

每一路分别计算归一化加权点积：

$$
D_{b,t,c}
=
\frac{r^X_{b,t,c}r^K_{b,t,c}}{\sqrt{5120}}
\sum_{d=0}^{5119}
X_{b,t,c,d}
W_{c,d}
K_{E,b,t,c,d},
$$

$$
D:[B,S,4].
$$

官方推理实现不会直接把 $D$ 输入 Sigmoid，而是先执行 Signed Sqrt：

$$
\widetilde D_{b,t,c}
=
\operatorname{copysign}
\left(
\sqrt{\max\left(|D_{b,t,c}|,10^{-6}\right)},
D_{b,t,c}
\right).
$$

这里使用 $\operatorname{copysign}$，因此 $D=0$ 时也按照源码复制正号，而不是把结果写成零。随后得到四路 Gate：

$$
g_{b,t,c}=\sigma\!\left(\widetilde D_{b,t,c}\right),
\qquad
g:[B,S,4].
$$

纯文本输入不需要额外处理。多模态输入中，图像位置的 Gate 在 Sigmoid 之后被 mask 置零：

$$
g_{b,t,c}\leftarrow
\begin{cases}
g_{b,t,c},&T_{\mathrm{mask},b,t}=1,\\
0,&T_{\mathrm{mask},b,t}=0.
\end{cases}
$$

图 2 为了压缩画面，把 $T_{\mathrm{mask}}$ 的箭头画在 Sigmoid 附近；源码执行顺序是先计算 Sigmoid，再应用 mask。

#### 把共享 Value 写回四路残差

同一个 $V_E$ 广播到四路，每一路使用自己的 Gate：

$$
\boxed{
Y_{b,t,c,d}
=
X_{b,t,c,d}
+g_{b,t,c}V_{E,b,t,d}
},
$$

$$
Y:[B,S,4,5120].
$$

因此，Engram 写回具有三个特征：

1. 四路共享同一份 $V_E$。
2. 四路分别使用自己的 $g_{b,t,c}$，写入幅度可以不同。
3. Engram 只更新四路残差；Layer 0 传来的动态 `pre_next` 原样保留，继续交给 Layer 1 的第一次 `hc_pre`。

### 2.3 参数量、计算量与 Required Global Memory 理论下界

本节只统计 Layer 1 的一个 Engram 算子。Layer 1 的 mHC、RMSNorm、局部 Attention 和 MoE 留到整层汇总，不在这里重复计算。数字后缀 B 表示 $10^9$；存储量使用 Byte、GB 和 GiB。

#### 参数量

只统计可学习模型标量。FP8 scale 属于量化存储元数据，计入后面的权重驻留字节，不重复算作逻辑模型参数。

| 参数 | Shape | 参数量 |
|---|---|---:|
| $E_{\mathrm{table}}$ | $[384006168,256]$ | 98,305,579,008 |
| $W_E$ | $[25600,6144]$ | 157,286,400 |
| $W_Q$ | $[4,5120]$ | 20,480 |
| $W_K$ | $[4,5120]$ | 20,480 |
| **合计** | — | **98,462,906,368** |

$$
\boxed{
\begin{aligned}
P_{\mathrm{Engram1}}
={}&384006168\times256\\
&+25600\times6144\\
&+2\times4\times5120\\
={}&98462906368.
\end{aligned}
}
$$

即 Layer 1 的单个 Engram 包含约 **98.463B** 个逻辑参数。技术报告中的约 196B 是 Layer 1 与 Layer 14 两个 Engram 的模型级合计近似。

#### 计算量

统计口径与 Layer 0 相同：标量加法、减法和乘法各计 1 次；XOR、整数取模、逻辑 Lookup、rsqrt、sqrt、exp、除法和比较单列。忽略量化与反量化、类型转换、通信、数据搬运和具体 kernel 融合。

下表为 Layer 1 Engram 处理一个 token 的计算量。Hash 部分只归因 Layer 1 的一组地址。

| 阶段 | 加 / 减 / 乘 | 其他运算 |
|---|---:|---|
| 压缩 Token ID | 0 | 逻辑 Lookup：1 |
| 当前及三个历史位置乘 Hash 乘数 | 4 | XOR：3 |
| 24 个分区取模并加 offset | 24 | 整数取模：24 |
| 读取 24 行 Engram 表 | 0 | 逻辑行 Lookup：24 |
| $6144\rightarrow25600$ 投影 | 314,547,200 | — |
| $X,K_E$ 的八个 RMS 因子 | 81,928 | rsqrt：8 |
| 四路加权点积与归一化缩放 | 61,448 | — |
| Signed Sqrt 与 Sigmoid | 8 | sqrt：4；exp：4；除法：4；比较：4 |
| 四路 $gV_E$ 写回 | 40,960 | — |
| **每 token 合计** | **314,731,572** | 见上 |

其中最大项是 $6144\rightarrow25600$ 投影。每个输出元素执行 6144 次乘法和 6143 次加法：

$$
25600\times(6144+6143)
=314547200.
$$

此外，官方推理实现每次调用 Engram 时还会计算一次：

$$
W=W_Q\odot W_K,
$$

产生 20,480 次与 token 数量无关的逐元素乘法。因此处理 $N=BS$ 个 token 时：

$$
\boxed{
C_{\mathrm{Engram1,mul/add/sub}}
=314731572N+20480
}.
$$

同时还有：

$$
3N\ \mathrm{XOR},
\qquad
24N\ \mathrm{整数取模},
\qquad
25N\ \mathrm{逻辑Lookup},
$$

$$
8N\ \mathrm{rsqrt},
\quad
4N\ \mathrm{sqrt},
\quad
4N\ \mathrm{exp},
\quad
4N\ \mathrm{除法},
\quad
4N\ \mathrm{比较}.
$$

官方推理实现会在一次 Hash 计算中同时生成 Layer 1 和 Layer 14 的地址，所以完整调用实际包含每 token 8 次整数乘法、6 次 XOR、48 次取模和 48 次 offset 加法。上面的算子表只把其中一半 Hash 主体归因给 Layer 1；共享 Token 压缩和历史 cache 不能在 Layer 14 再重复统计。

#### Required Global Memory 理论下界（理想融合口径）

与 Layer 0 相同，这里把 Engram 视为一个逻辑分析单元，并采用以下假设：

- **Weight**：参数和量化 scale 长期保存在 Global Memory。
- **I/O**：完整输入与输出保存在 Global Memory，并分别计数。
- **Internal Activation**：$E_{\mathrm{flat}},F,K_E,V_E,D,g$ 等中间量允许通过融合、分块或工作区复用，不要求全部完整写回 Global Memory。

按 `world_size=1` 核算，固定 revision 的官方推理实现中权重存储为：

| 权重 | 数据与 scale | GPU 驻留字节 |
|---|---|---:|
| Engram 表 | $384006168\times(256+8)$ | 101,377,628,352 |
| $W_E$ | $25600\times6144+800\times192$ | 157,440,000 |
| $W_Q,W_K$ | $2\times4\times5120\times2$ | 81,920 |
| **合计** | — | **101,535,150,272** |

Engram 表使用 FP8 E4M3 数据，每 32 个元素配一个 E8M0 scale；$W_E$ 使用 FP8 E4M3 数据和 $32\times32$ block scale；$W_Q,W_K$ 使用 BF16。

Engram 主体的逻辑边界为：

| Tensor | 分类 | Shape / dtype | Logical Size |
|---|---|---|---:|
| $X$ | Input | $[B,S,4,5120]$ BF16 | $40960N$ B |
| $I_E$ | Input | $[B,S,24]$ INT64 | $192N$ B |
| $Y$ | Output | $[B,S,4,5120]$ BF16 | $40960N$ B |

如果把 Layer 1 的 24 个地址视为独立输入，则：

$$
\boxed{
M_{\mathrm{Engram1,body}}
=101535150272+82112N
\quad\mathrm{Byte}
}.
$$

在官方推理实现中，$I_E$ 实际是完整 $I_{\mathrm{all}}:[B,S,2,24]$ 的视图，不会单独分配 $192N$ Byte。共享 Hash 前置组件还持有：

| 对象 | Shape / dtype | 字节数 |
|---|---|---:|
| Token 压缩映射 | $[129280]$ INT64 | 1,034,240 |
| 两层 Hash 乘数 | $[2,4]$ INT64 | 64 |
| 两层质数模数 | $[2,3,8]$ INT64 | 384 |
| 两层 offset | $[2,24]$ INT64 | 384 |
| 压缩 Token 历史 cache | $[B_{\max},S_{\max}]$ INT64 | $8B_{\max}S_{\max}$ |
| 完整 Hash 输出 | $[B,S,2,24]$ INT64 | $384N$ |

前四项固定 buffer 合计 1,035,072 Byte。按实际共享分配、不重复计算 Layer 1 地址视图时：

$$
\boxed{
\begin{aligned}
M_{\mathrm{Engram1+shared\ hash,min}}
={}&101536185344\\
&+82304N\\
&+8B_{\max}S_{\max}
\quad\mathrm{Byte}.
\end{aligned}
}
$$

多模态路径若把新生成的 BOOL `engram_mask:[B,S]` 独立计入，再加 $N$ Byte。原始 Token ID $T$ 是模型级共享输入；若把 Engram 当作完全独立的端到端算子并把 $T$ 也计入边界，再加 $8N$ Byte。

这些公式是 Required Global Memory 的理论下界，不是进程峰值显存，也不是显存读写量。FP8 解量化临时值、GEMM workspace、分片通信 buffer、allocator 保留区和 host-memory offload 都会改变实际测量结果。

形状、数据类型与执行顺序依据固定 revision 的[模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/config.json)、[Hash 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/engram.py)和[模型前向实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py)。

## 3. Layer 1 第一次 `hc_pre` 的真正变化

模型只在最初把 embedding 扩成四路时创建一次 identity-pre：

$$
p^{\mathrm{identity}}=[1,0,0,0].
$$

Layer 0 的 Attention 入口使用它；Layer 0 执行结束后返回 MoE 侧生成的 `ffn_pre`，即 $p^{(0)}_{\mathrm{next}}$。Layer 1 不会重新创建 identity-pre，而是直接用这个动态状态汇聚 Engram 更新后的四路残差：

$$
\boxed{
Z^{(1)}_{b,t,d}
=\sum_{c=0}^{3}
p^{(0)}_{\mathrm{next},b,t,c}
X'_{b,t,c,d}
},
\qquad Z^{(1)}:[B,S,5120].
$$

与此同时，Layer 1 Attention 侧的 `hc_mixes` 已经从 $X'$ 预测出新的 `attn_pre/post/comb`。但这里的第一次 `hc_pre` 仍使用**上一层**传来的 $p^{(0)}_{\mathrm{next}}$；本层新生成的 `attn_pre` 留给紧随其后的 MoE 入口。顺序是：

$$
\underbrace{p^{(0)}_{\mathrm{next}}}_{\text{Layer 0 MoE 产生}}
\xrightarrow{\mathrm{Layer\ 1\ Attention\ hc\_pre}}
\underbrace{p^{(1)}_{A}}_{\text{Layer 1 Attention 产生}}
\xrightarrow{\mathrm{Layer\ 1\ MoE\ hc\_pre}}
\underbrace{p^{(1)}_{\mathrm{next}}}_{\text{Layer 1 MoE 产生}}.
$$

这正是 Single-Pass mHC 的跨子层、跨层系数传递。与 Layer 0 相比，Layer 1 的 `hc_pre` 算法没有变化，变化的是第一次调用的系数来源。[Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L909-L997)

## 4. 继承自 Layer 0 的模块

Engram 写回之后，Layer 1 进入与 Layer 0 同构的 Block。下面只记录复用关系；公式、shape 和内部步骤见 [Layer 0：从四路残差到局部注意力与 MoE](02-layer0.md)。

| 模块 | Layer 1 相对 Layer 0 | 需要记住的差别 |
|---|---|---|
| Attention 侧 mHC | 算法与 shape 相同，参数独立 | 系数从 Engram 更新后的 $X'$ 预测；首次汇聚用动态 `pre_next` |
| 两个入口 RMSNorm | 相同结构，参数独立 | 输入数值不同 |
| Local Attention / SWA | 相同结构，参数独立 | 同为 128-token 窗口；没有 global KV、Compressor 或 Indexer |
| Attention 侧 `hc_post` | 相同 | residual 是 Engram 更新后的四路张量 |
| MoE 侧 mHC | 相同结构，参数独立 | 其 `ffn_pre` 传给 Layer 2 |
| MoE FFN | 相同结构，参数独立 | 仍是 Top-6 routed + 1 shared expert |
| Local KV cache 与 RoPE 表 | 相同 shape，各层独立 | 不与 Layer 0 共用缓存或权重 |

所以 Layer 1 可以概括为：

$$
\boxed{
\mathrm{Layer\ 1}
=\mathrm{Layer\ 0\ 基准\ Block}
+\mathrm{前置\ Engram}
+\mathrm{动态跨层\ pre}
}
$$

这里的加号表示结构差异，不表示参数共享。

## 5. Layer 1 整层汇总

Engram、动态 `pre_next` 和继承模块已经分别说明。下面按照 Layer 0 的写法，只汇总整层参数、计算、显存和跨层输出，不再把 Engram 算子拆开重复讲述。

### 5.1 整层参数量与 GPU 驻留字节

Layer 1 的基准 Block 参数结构与 Layer 0 相同，再加本层 Engram：

| 部分 | 逻辑参数量 | GPU 驻留字节 |
|---|---:|---:|
| 基准 Block：两套 mHC、Norm、SWA、MoE | 13,754,511,990 | 7,423,022,552 B |
| Layer 1 Engram | 98,462,906,368 | 101,535,150,272 B |
| **Layer 1 合计** | **112,217,418,358** | **108,958,172,824 B** |

$$
\boxed{P_{\mathrm{Layer1}}=112217418358\approx112.217\ \mathrm{B}}
$$

$$
\boxed{
M_{\mathrm{Layer1,param}}
=108958172824\ \mathrm{Byte}
\approx108.958\ \mathrm{GB}
\approx101.475\ \mathrm{GiB}
}
$$

Engram 占 Layer 1 逻辑参数的约 **87.74%**，占本文核算权重驻留字节的约 **93.19%**。这不表示生产部署必须把 101.5 GB Engram 全放在单张 GPU：官方报告明确采用确定性地址进行预取，官方推理实现也支持按行跨 `world_size` 分片，并在查表后合并结果。本节的单卡完整驻留值是便于核算的 `world_size=1` 基准，不是部署建议。

### 5.2 整层前向计算量

Layer 0 的官方通用路径为：

$$
C_{\mathrm{Layer0}}
=N\left(755138906+131200\overline L\right),
$$

其中 $\overline L$ 是每个查询实际参与 SWA 评分的平均局部历史长度，$1\le\overline L\le128$。Layer 1 后半段使用同一通用路径；它的首次 `hc_pre` 是动态系数，不能做 Layer 0 identity-pre 的理论常量折叠。

加上 Engram：

$$
\boxed{
C_{\mathrm{Layer1,mul/add/sub}}
=N\left(1069870478+131200\overline L\right)+20480
}
$$

在暂不考虑 $\overline L$ 的固定部分时，Engram 使本层每 token 的加/减/乘主项比 Layer 0 官方通用路径增加约 **41.68%**。这个比例不是实际延迟比例：Engram 表可预取和分片，FP8 GEMM、访存、通信、并发重叠与 kernel 融合都会改变硬件时间。[官方报告的 Engram 预取说明](https://arxiv.org/abs/2609.19969)

### 5.3 整层 Required Global Memory 理论下界与状态

这一节给出固定 revision 官方推理实现的**共享状态视角**：参数和 scale、层边界 I/O、本层 local Attention 状态、模型级 Hash 固定 buffer、Token 历史 cache，以及同时服务 Layer 1 和 Layer 14 的完整 `engram_hashes`。内部 $E_{\mathrm{flat}},F,K_E,V_E,D,g$ 假设可通过融合、分块或工作区复用，不要求全部同时写入 Global Memory。

#### 边界 I/O 与持久状态

Layer 1 输入和输出均包含四路 BF16 残差与一份 FP32 pre：

$$
M_{\mathrm{boundary}}
=2\times N(4\times5120\times2+4\times4)
=81952N\ \mathrm{Byte}.
$$

本层 Attention 状态与 Layer 0 相同：

$$
\mathrm{window\_kv\_cache}:[B_{\max},128,512]\ \mathrm{BF16}
=131072B_{\max}\ \mathrm{Byte},
$$

$$
\mathrm{freqs\_cis}:[S_{\max},32]\ \mathrm{Complex64}
=256S_{\max}\ \mathrm{Byte}.
$$

模型级 Hash 前置组件持有：

| 对象 | Shape / dtype | 字节数 |
|---|---|---:|
| Token 压缩映射 | `[129280]` INT64 | 1,034,240 |
| 两层 Hash 乘数 | `[2,4]` INT64 | 64 |
| 两层质数模数 | `[2,3,8]` INT64 | 384 |
| 两层 offset | `[2,24]` INT64 | 384 |
| 压缩 Token 历史 cache | `[B_max,S_max]` INT64 | $8B_{\max}S_{\max}$ |
| 本次完整 Hash 输出 | `[B,S,2,24]` INT64 | $384N$ |

前四项合计 1,035,072 Byte。`engram_hashes` 在进入 Layer 0 前生成，并至少保留到 Layer 14 消费完成；Layer 1 使用的 `[B,S,24]` 只是视图，不再分配 192N Byte。

#### 汇总公式

纯文本路径没有 `engram_mask`，于是：

$$
\boxed{
\begin{aligned}
M_{\mathrm{Layer1,min}}={}&108958172824
+81952N
+384N\\
&+131072B_{\max}
+256S_{\max}\\
&+1035072
+8B_{\max}S_{\max}
\end{aligned}
\quad\mathrm{Byte}
}
$$

整理为：

$$
\boxed{
M_{\mathrm{Layer1,min}}
=108959207896
+82336N
+131072B_{\max}
+256S_{\max}
+8B_{\max}S_{\max}
\quad\mathrm{Byte}
}
$$

多模态路径若把新生成的 BOOL `engram_mask:[B,S]` 独立计入，再加 $N$ Byte。上式把整份 Hash 固定 buffer、Token cache 和 `[B,S,2,24]` Hash 输出计了一次；它们是 Layer 1 与 Layer 14 共享的模型级资源，做整模型汇总时不能再次相加。

这仍不是进程峰值显存，也不是显存读写量：未融合激活、FP8 解量化临时值、GEMM workspace、allocator 保留区、分片通信 buffer 以及实际推理引擎的 Engram CPU / host-memory offload 都会改变测量结果。

### 5.4 跨层输出与结论

Layer 1 最终输出：

| 对象 | Shape | dtype | 去向 / 生命周期 |
|---|---|---|---|
| 四路残差 $X^{(1)}_{\mathrm{out}}$ | `[B,S,4,5120]` | BF16 | Layer 2 Attention 侧 mHC |
| $p^{(1)}_{\mathrm{next}}$ | `[B,S,4]` | FP32 | Layer 2 第一次 `hc_pre` |
| `window_kv_cache` | `[B_max,128,512]` | BF16 | Layer 1 后续 Decode 的 128-token 环形窗口 |
| Token Hash cache | `[B_max,S_max]` | INT64 | Layer 1、14 共用的跨 Prefill / Decode 压缩 Token 历史 |
| `engram_hashes` | `[B,S,2,24]` | INT64 | 本次前向中供 Layer 1、14 依次消费 |

核心结论只有三条：

1. **Layer 1 是 Layer 0 基准 Block 的第一次结构扩展。** 新增 Engram，但 Attention 仍是纯 128-token SWA，MoE 与 mHC 主体不变。
2. **Engram 向四路写入共享 Value、独立 Gate。** 它更新四路残差，却不改动 Layer 0 传来的动态 `pre_next`。
3. **本层容量主要由条件记忆决定。** 单个 Engram 带来 98.463B 逻辑参数和约 94.562 GiB 的 `world_size=1` 权重驻留；实际部署依赖行分片、预取或 host-memory offload，而不是把它理解成必须单卡常驻的普通线性层。

## 6. 资料来源与版本说明

- [DeepSeek-V4.1-Flash 技术报告](https://arxiv.org/abs/2609.19969)：模型总体结构、两处 Engram、196B 总容量、Hash head 与部署预取。
- [Engram 原始论文](https://arxiv.org/abs/2601.07372)：条件记忆、Tokenizer Compression、Multi-Head Hashing 与 Context-aware Gating。
- [固定 revision 的配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/config.json)：Layer 1、Engram、SWA、MoE 的精确尺寸。
- [固定 revision 的 Hash 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/engram.py)：Token 压缩、历史 cache、Hash 乘数、质数桶与 offset。
- [固定 revision 的模型前向](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py)：Engram 表、KV 投影、Gate、Block 与跨层 `pre_mix`。

全文于 2026-10-20 对照固定 revision [`5254448103b6254fc646dacbca1cbbf20e9216c1`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/5254448103b6254fc646dacbca1cbbf20e9216c1) 重新核算。参数量、计算量和 Required Global Memory 使用本文声明的逻辑口径；官方报告中的模型级近似、官方推理实现的张量分配和具体部署引擎的测量值不混为同一数字。
