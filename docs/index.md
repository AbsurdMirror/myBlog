---
title: 图解技术笔记
status: published
updated: 2026-10-10
---

# 图解技术笔记

从一张结构图开始，逐层理解模型的数据流、数学公式与张量形状。

## 模型解析

### DeepSeek V4.1 · Flash

[阅读 Layer 0：四路残差、局部注意力与 MoE](models/deepseek-v4.1-flash/02-layer0.md)

当前篇章含九张手绘原图：整层结构、mHC 三步计算、RMSNorm，以及 SWA 的窗口 KV、sink softmax 和两级输出投影。复杂图可以点开放大，公式和 shape 同时保留为文本。

[系列目录、六篇阅读规划与资料说明](models/deepseek-v4.1-flash/index.md)

系列规划为约六篇正文＋一个目录页，目前 Layer 0 已发布，其他篇目待后续讨论逐步展开。

## 旧文归档

[旧博客内容与保留地址](legacy/index.md)
