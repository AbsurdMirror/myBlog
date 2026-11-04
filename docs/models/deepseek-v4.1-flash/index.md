---
title: DeepSeek-V4.1-Flash 图解系列
status: published
updated: 2026-11-04
---

# DeepSeek-V4.1-Flash 图解系列

本系列研究 **DeepSeek-V4.1-Flash** 的结构、数学计算与推理状态。以 Layer 0 为基准层，逐步理解其他层的差异和特殊模块，最后串起完整推理与成本分析。其他变体的配置和层分类不直接沿用。

目前 **Layer 0、Layer 1 和 Layer 2 三篇正文已发布**，其余篇目仍是阅读规划，不表示内容已经完成。篇目可随后续讨论调整。

## 阅读顺序与完成状态

| 篇目 | 解决的问题与范围 | 前置阅读 | 当前状态 |
|---|---|---|---|
| ① 模型全貌 | 输入输出、整体主干、层分类、层间数据及缓存概览；建立模型地图，不逐算子推导 | 无 | 规划中，正文尚未开始 |
| [② Layer 0：基准层详解](02-layer0.md) | 八模块、四路残差、mHC、RMSNorm、SWA、MoE；公式、shape、参数与状态 | 先了解本文符号；①完成后建议先读① | 已发布，含十二张配图；持续修订 |
| [③ Layer 1：从 Engram 写入到局部注意力与 MoE](03-layer1.md) | 以 Layer 0 为基准，只展开前置 Engram、动态跨层 `pre_next`、增量参数、计算与状态 | ② | 已发布，含两张配图 |
| [④ Layer 2：从局部注意力到压缩稀疏注意力与 MoE](04-layer2.md) | Compressor、Indexer、Local / Global KV 与索引合并、稀疏 Attention，以及 Attention 与整层资源分析 | ②、③ | 已发布，含七张配图 |
| ⑤ 特殊模块详解 | 候选主题为 Engram、视觉路径、DSpark 等；展开作用、接入位置、内部计算与状态 | ①、②及相关差异层 | 规划中，逐项讨论后展开 |
| ⑥ 完整推理过程 | Prefill、Decode、多轮对话的依赖、时间顺序与状态生命周期 | ①—⑤中相关内容 | 规划中，正文尚未开始 |
| ⑦ 存储与计算分析 | 权重、持久缓存、临时激活、计算量、访存量及实现与硬件映射 | 前述结构与推理篇 | 规划中，正文尚未开始 |

目前可直接从第②篇开始阅读，再用第③篇观察 Layer 1 如何增加 Engram，用第④篇观察 Layer 2 如何引入压缩稀疏 Attention。差异层复用基准层已经讲清的模块。尚未写成正文的篇目不提供空链接；整模型级的存储、性能和硬件执行集中到第⑦篇。

## 版本与资料依据

官方资料入口：[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。Layer 0 全文于 2026-10-14、Layer 1 全文于 2026-10-20、Layer 2 全文于 2026-11-04 对照固定 revision [`5254448103b6254fc646dacbca1cbbf20e9216c1`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/5254448103b6254fc646dacbca1cbbf20e9216c1) 复核；各图的标注勘误和限定仍以相邻正文及 `figures.json` 为准。

## 共用符号

| 记号 | 含义 |
|---|---|
| `B` | batch，独立会话数 |
| `S` | 本次处理的 token 数 |
| `S_his` | 参与评分的 KV 总长度，包含本次已加入的 token |
| `B*S` | 逐 token 独立运算时展平的行数，不表示跨会话混合 |
| `hidden` / `D` | 隐藏维度；Layer 0 正文取 `D=5120` |

完整序列示例明确 `S_his=S`；带历史缓存的场景另行声明。可学习权重、运行时系数和固定常数分别说明。

## 已发布文章的配图进度

Layer 0 使用十二张图：整层八模块一张、mHC 系数生成两张、`hc_pre` 汇聚一张、`hc_post` 融合一张、RMSNorm 一张、SWA 三张，以及 MoE 的 Router、专家参数逻辑索引、专家计算三张。Layer 1 新增两张 Engram 图，分别说明 Hash 查表和四路门控写回；图中简写与 mask 位置的限定以相邻正文为准。

Layer 2 使用七张图，依次展示整层总览、Attention 总图、Q / Local KV、Compressor、Indexer、KV 与索引合并，以及复用的 Layer 0 注意力计算图。
