---
title: NVIDIA Tensor Memory Accelerator 调研
status: published
updated: 2026-10-30
---

# NVIDIA Tensor Memory Accelerator 调研

本系列从产品演进、PTX 接口与硬件设计三个层次梳理 NVIDIA Tensor Memory Accelerator（TMA）。前三章先建立代际地图，再依次深入 Hopper 第一代接口以及 Blackwell、Rubin 的后续 PTX 演进；第四章结合 NVIDIA 专利、官方说明和逆向资料，讨论 TMA 的模块、数据通路与完成机制。

## 阅读顺序

| 篇目 | 内容 | 状态 |
|---|---|---|
| [第一章：NVIDIA 产品迭代与 TMA 概述](01-overview.md) | 从 Ampere 异步拷贝前身到 Hopper、Blackwell 与 Rubin，区分产品扩展、公共接口能力与尚待验证的硬件实现 | 已发布 |
| [第二章：Hopper 第一代 TMA 的 PTX 设计](02-hopper-ptx.md) | Tensor Map 字段、bulk/tensor copy、reduction、prefetch、完成机制与 descriptor proxy 更新 | 已发布 |
| [第三章：Blackwell 与 Rubin 的 TMA PTX 演进](03-blackwell-rubin-ptx.md) | PTX 8.6–9.4 的 CTA-local destination、mask、gather/scatter、target 分化、instruction-time descriptor override 与 cache priority | 已发布 |
| [第四章：TMA 硬件设计](04-hardware-design.md) | per-SM 服务模型、descriptor cache、请求与响应、mbarrier、Multicast、Group、Reduce 和 descriptor 更新 | 已发布 |

## 独立文章

| 篇目 | 内容 | 状态 |
|---|---|---|
| [TMA 地址生成：Tile 与 Im2col](address-generation.md) | 从逻辑 Tensor、Pixel Box 和坐标转换到 OOB 判断与 Global Memory 地址 | 已发布 |
| NV Swizzle 分析 | 数据重排与地址变换 | 草稿，尚未发布 |

## 阅读提示

- 第一章覆盖全局路线图，适合先建立产品、Compute Capability、PTX 版本和 target 之间的关系。
- 第二章只讨论 Hopper/PTX 8.0–8.3，不把 Blackwell 或 Rubin 的后续接口混入首代设计。
- 第三章以第二章为基线，只记录 PTX 8.6–9.4 中新增或发生变化的描述符字段、指令修饰符、操作数和 target 约束。
- 第四章以 Hopper 为主要背景，区分专利实施例、逆向观察和据此形成的实现推理；地址生成与 Swizzle 的数学分析另作独立文章。
- 正文保留稳定的 `[Sxxx]` 引用编号；对应的一手资料入口集中在[参考资料索引](references.md)。
- “第一、第二、第三功能阶段”是本调研为了组织公开能力变化采用的分析框架，并非 NVIDIA 官方硬件代际名称。

## 当前范围

当前发布包含 4 篇章节正文、1 篇独立文章、33 张配图与一份精简参考资料索引。第四章和地址生成文章在正文中链接原始资料，便于对照。

