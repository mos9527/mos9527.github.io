---
author: mos9527
lastmod: 2025-12-01T11:01:11.623433
title: Foundation 施工笔记 【3】- 原生 Profiling 及优化
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

[上一篇](https://mos9527.com/posts/foundation/pt-2-gpu-driven-pipeline-with-culling/)后，我们*几乎*拥有了一个非常高效且“所见即所得”的 GPU 渲染管线。不过，特别地，对于**遮蔽剔除**，可以注意到非平凡的额外开销。在我的RDNA3集显本，2.5K分辨率下，frametime在开启后反而增了~3ms...

很显然，原实现是亟须优化的。当然，GPU侧，NV/AMD 也早有了非常成熟的工具链，即[NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics)，[Radeon™ Developer Tool Suite](https://gpuopen.com/news/introducing-radeon-developer-tool-suite/)。CPU侧自己也很早集成了[tracy profiler](https://github.com/wolfpld/tracy)。但这些都离不开第三方工具的加持 - 接下来将介绍一些“引擎”内的profile手段，和利用它们及官方工具降低culling带来的overhead。

## Timestamp Queries

Query 类 API 覆盖率相当广泛 - [Vulkan (1.0+)](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#queries)，[DX12](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing) 等等皆有第一方支持 - 为我们的 RHI 及 [Renderer](https://mos9527.com/Foundation/classFoundation_1_1RenderCore_1_1Renderer.html) 每个pass添加也并非难事。

在这里我们需要的主要是timing/时序信息，故仅需采样[Timestamp Queries](https://docs.vulkan.org/samples/latest/samples/api/hpp_timestamp_queries/README.html)。在每个pass真正录制前后引入`TopOfPipe`, `BottomOfPipe`timestamp引入即可（保守地）估计执行该pass需要**至少**用时，如图。

![image-20251201110059390](/image-foundation/image-20251201110059390.png)