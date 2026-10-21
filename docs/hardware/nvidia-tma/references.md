---
title: NVIDIA TMA 调研参考资料索引
status: published
updated: 2026-10-21
---

# NVIDIA TMA 调研参考资料索引

本页只保留前两章实际引用的一手资料入口与稳定编号，供正文中的 `[Sxxx]` 回指；调研笔记、摘录和中间判断不在博客发布。

## 第一章引用

- **S001** — [CUDA Programming Guide：Using the Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
- **S002** — [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- **S003** — [Hopper Tuning Guide 12.1](https://docs.nvidia.com/cuda/archive/12.1.0/pdf/Hopper_Tuning_Guide.pdf)
- **S004** — [CUDA Toolkit 12.0 Release Notes](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/index.html)
- **S005** — [CUDA Driver API 12.0：Tensor Memory Object Management](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- **S006** — [Ampere Tuning Guide 11.8](https://docs.nvidia.com/cuda/archive/11.8.0/ampere-tuning-guide/index.html)
- **S007** — [CUDA C++ Programming Guide 11.8：Asynchronous Data Copies](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html)
- **S008** — [CUDA C++ Programming Guide 12.5：Tensor Memory Access](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html)
- **S009** — [CuTe TMA Tensors](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/cute/0z_tma_tensors.html)
- **S011** — [CUDA GPU Compute Capability 产品表](https://developer.nvidia.com/cuda/gpus)
- **S012** — [PTX ISA 8.0 / CUDA 12.0](https://docs.nvidia.com/cuda/archive/12.0.1/pdf/ptx_isa_8.0.pdf)
- **S013** — [CUDA 11.8 Release Notes 与 PTX ISA 7.8](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/)
- **S014** — [H100/Hopper 发布](https://nvidianews.nvidia.com/news/nvidia-announces-hopper-architecture-the-next-generation-of-accelerated-computing)
- **S015** — [H200 产品发布](https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform)
- **S016** — [Blackwell/B200/GB200 发布](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing)
- **S017** — [Blackwell Ultra/B300/GB300 发布](https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning)
- **S018** — [PTX ISA release notes：Blackwell-era TMA 扩展](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes)
- **S020** — [Rubin 平台发布](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)
- **S021** — [Inside NVIDIA Rubin GPU Architecture](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)
- **S022** — [PTX ISA 9.4：Rubin-era TMA 表达](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- **S023** — [CUDA Programming Guide：Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- **S024** — [CUDA Toolkit 12.0 Released for General Availability](https://developer.nvidia.com/blog/cuda-toolkit-12-0-released-for-general-availability/)
- **S025** — [Blackwell MLPerf Training v4.1](https://developer.nvidia.com/blog/nvidia-blackwell-doubles-llm-training-performance-in-mlperf-training-v4-1/)
- **S026** — [NVIDIA Grace Hopper Superchip Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/)

## 第二章引用

- **S027** — [PTX ISA 8.0：Tensor、Tensor Map 与 Bulk/Tensor 指令](https://docs.nvidia.com/cuda/archive/12.0.0/pdf/ptx_isa_8.0.pdf)
- **S028** — [CUDA Driver API 12.0：Hopper Tensor Map 编码字段](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- **S031** — [PTX ISA 8.8：Hopper `tensormap.replace` 与 Proxy 操作](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)
- **S032** — [PTX ISA 8.3：Memory Hierarchy with Clusters](https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#memory-hierarchy)

## 第三章引用

- **S034** — [PTX ISA 8.7：Blackwell TMA 指令与 tensor mode 语义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)
- **S035** — [PTX ISA 8.8：family-specific target、sub-byte type 与 96 B swizzle](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/index.html)
- **S036** — [PTX ISA 9.0–9.3：target 更名、`.ignore_oob` 与 memory semantics](https://docs.nvidia.com/cuda/archive/13.3.0/parallel-thread-execution/index.html)
- **S037** — [PTX ISA 9.4：Tensor Property Override](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy)
- **S038** — [PTX ISA 9.4：Report、32-bit Multicast 与 Cache Priority](https://docs.nvidia.com/cuda/parallel-thread-execution/#completion-mechanisms-for-asynchronous-copy-operations)
- **S039** — [CUDA Driver API 12.8：Blackwell Tensor Map 编码接口](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- **S040** — [CUDA Core Compute Libraries：`cp.async.bulk.tensor`](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html)、[LLVM NVPTX Backend User Guide](https://llvm.org/docs/NVPTXUsage.html)

---

[返回 TMA 系列目录](index.md) · [第一章](01-overview.md) · [第二章](02-hopper-ptx.md) · [第三章](03-blackwell-rubin-ptx.md)
