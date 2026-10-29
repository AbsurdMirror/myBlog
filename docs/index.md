---
title: 图解技术笔记
status: published
updated: 2026-10-29
---

# 图解技术笔记

从一张结构图开始，逐层理解模型的数据流、数学公式与张量形状。

## GPU 架构与编程

### NVIDIA Tensor Memory Accelerator

[阅读 NVIDIA TMA 调研系列](hardware/nvidia-tma/index.md)

前四章已经发布：从产品演进到 Hopper、Blackwell 与 Rubin 的 PTX 接口，再结合专利与逆向资料分析 TMA 内部模块、数据通路、Multicast 和 Group 完成机制。

## 模型解析

### DeepSeek V4.1 · Flash

[阅读 Layer 0：四路残差、局部注意力与 MoE](models/deepseek-v4.1-flash/02-layer0.md)

当前篇章含十二张手绘配图：整层结构、mHC 三步计算、RMSNorm、SWA 的窗口 KV / sink softmax / 两级输出投影，以及 MoE 的 Router / 专家参数索引 / 专家计算。正文已补齐 Layer 0 整层参数、计算、显存和状态汇总；复杂图可以点开放大，公式和 shape 同时保留为文本。

[系列目录、六篇阅读规划与资料说明](models/deepseek-v4.1-flash/index.md)

系列规划为约六篇正文＋一个目录页，目前 Layer 0 已发布，其他篇目待后续讨论逐步展开。

## 旧文归档

[旧博客内容与保留地址](legacy/index.md)
