---
title: NVIDIA Tensor Memory Accelerator 调研
status: published
updated: 2026-10-20
---

# NVIDIA Tensor Memory Accelerator 调研

本系列从产品演进、编程模型与 PTX 指令三个层次梳理 NVIDIA Tensor Memory Accelerator（TMA）。前两章先建立代际地图，再深入 Hopper 第一代接口；Blackwell 与 Rubin 的 PTX 演进留待后续独立文章。

## 阅读顺序

| 篇目 | 内容 | 状态 |
|---|---|---|
| [第一章：NVIDIA 产品迭代与 TMA 概述](01-overview.md) | 从 Ampere 异步拷贝前身到 Hopper、Blackwell 与 Rubin，区分产品扩展、公共接口能力与尚待验证的硬件实现 | 已发布 |
| [第二章：Hopper 第一代 TMA 的 PTX 设计](02-hopper-ptx.md) | Tensor Map 字段、bulk/tensor copy、reduction、prefetch、完成机制与 descriptor proxy 更新 | 已发布 |
| Blackwell PTX 8.6–8.8 演进 | CTA-local destination、mask、gather/scatter、target 分化 | 后续独立文章 |
| Rubin PTX 9.4 演进 | instruction-time descriptor override 与 eviction priority | 后续独立文章 |

## 阅读提示

- 第一章覆盖全局路线图，适合先建立产品、Compute Capability、PTX 版本和 target 之间的关系。
- 第二章只讨论 Hopper/PTX 8.0–8.3，不把 Blackwell 或 Rubin 的后续接口混入首代设计。
- 正文保留稳定的 `[Sxxx]` 引用编号；对应的一手资料入口集中在[参考资料索引](references.md)。
- “第一、第二、第三功能阶段”是本调研为了组织公开能力变化采用的分析框架，并非 NVIDIA 官方硬件代际名称。

## 当前范围

当前发布包含 2 篇正文、16 张采用版配图与一份精简参考资料索引。文章保留源调研的技术结论与证据边界，只对博客导航、图片路径和章节路线图做了发布所需的最小调整。

