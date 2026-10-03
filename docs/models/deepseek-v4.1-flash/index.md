---
title: DeepSeek-V4.1-Flash 图解系列
status: draft
---

# DeepSeek-V4.1-Flash 图解系列

从模型的外部状态开始，选取基准层建立数学理解，再讨论其他层差异、推理过程与成本。当前只展开已讨论的 Layer 0，不把未确认部分填成结论。

| 篇目 | 内容与状态 |
|---|---|
| 01 模型全貌 | 输入输出、层分类、状态与缓存；计划中 |
| [02 Layer 0 基准层](02-layer0.md) | 八模块、mHC、RMSNorm、SWA、MoE；已有五张配图的审阅稿 |
| 03 其他层的差异 | 生成／复用 KV、重新索引、解码器入口；计划中 |
| 04 特殊模块 | Engram、视觉与其他独立模块；计划中 |
| 05 完整推理过程 | Prefill／Decode 状态流转；计划中 |
| 06 存储与计算 | 参数、缓存、激活、运算和实现；计划中 |

模型上游：[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。本轮在 2026-10-03 核对 main 配置与参考前向。候选 revision 已记录，但固定版本文件访问尚未核实，因此本文不是已完成版本锁定的复现报告。
