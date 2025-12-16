---
author: mos9527
lastmod: 2025-12-16T23:48:32.260161
title: Foundation 施工笔记 【6】- 路径追踪
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

Foundation现在(2025/12/16)也有了能用的RT相关API，Editor的GPUScene也有了BLAS上传/压缩(compact)与逐帧TLAS更新支持。到目前为止用rt做的只有inline query实现硬阴影——在做实时GI相关内容之前，不妨复习下采样/PBR/降噪相关知识——那就写个GPU Path Tracer吧？

PBRT/[Physically Based Rendering:From Theory To Implementation](https://pbr-book.org/)/[Kanition大佬v3翻译版](https://github.com/kanition/pbrtbook) 和手头的 RTR4/[Real-Time Rendering 4th Edition](https://www.realtimerendering.com/) （尤其是第九章）将是我们这里主要的信息来源。

### SBT (Shader Binding Table)

之前用过了非常方便的Inline Ray Query - 从fragment/pixel，compute可以直接产生光线进行trace：从这里出发进行PT是可行的，这也是[nvpro-samples/vk_mini_path_tracer](https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html#moresamples) 的教学式做法。

不过完全利用硬件的RT管线会离不开[SBT/Shader Binding Table](https://docs.vulkan.org/spec/latest/chapters/raytracing.html#shader-binding-table)，即Shader绑定表。除了shader单体更小更快之外，调度也有由驱动优化的可能。此外RHI目前还没有SBT相关设施，借此一并处理。参考 [nvpro-samples/vk_raytracing_tutorial_KHR](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/#step-43-create-basic-ray-tracing-pipeline-structure)

<h1 style="color:red">--- 施工中 ---</h1>