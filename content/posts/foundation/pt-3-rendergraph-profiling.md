---
author: mos9527
lastmod: 2025-12-02T17:08:59.175871
title: Foundation 施工笔记 【3】- 原生 Profiling 及早期优化
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

[上一篇](https://mos9527.com/posts/foundation/pt-2-gpu-driven-pipeline-with-culling/)后，我们*几乎*拥有了一个非常高效且“所见即所得”的 GPU 渲染管线。不过，特别地，对于**遮蔽剔除**，可以注意到非平凡的额外开销。

很显然，原实现是亟须优化的。当然，GPU侧，NV/AMD 也早有了非常成熟的工具链，即[NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics)，[Radeon™ Developer Tool Suite](https://gpuopen.com/news/introducing-radeon-developer-tool-suite/)。CPU侧自己也很早集成了[tracy profiler](https://github.com/wolfpld/tracy)。但这些都离不开第三方工具的加持 - 接下来将介绍一些“引擎”内的profile手段，和利用它们及官方工具降低culling带来的overhead。

## Timestamp Queries

Query 类 API 覆盖率相当广泛 - [Vulkan (1.0+)](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#queries)，[DX12](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing) 等等皆有第一方支持 - 为我们的 RHI 及 [Renderer](https://mos9527.com/Foundation/classFoundation_1_1RenderCore_1_1Renderer.html) 每个pass添加也并非难事。以下为 Renderer 中节选：

```c++
    void Execute(size_t thread_id) noexcept override
    {
        ZoneScoped;
        ZoneNameF("<%s>", pass->name.c_str());
        *cmd = r->ExecuteAllocateCommandList(pass->queue, thread_id);
        (*cmd)->Begin();
        (*cmd)->WriteTimestamp(queryPool, RHIPipelineStageBits::TopOfPipe, pass->handle * 2);
        (*cmd)->BeginTransition();
        for (auto& [res, desc] : (*barriers))
        {
            res.Visit([&](RHIBuffer* p) { (*cmd)->SetBufferTransition(p, desc); },
                      [&](RHITexture* p) { (*cmd)->SetImageTransition(p, desc); });
        }
        (*cmd)->EndTransition();
        (*cmd)->DebugBegin(pass->name.c_str());
        pass->frameExec = r->mFrameSwapped;
        pass->pass->Record(pass->handle, r, *cmd);
        (*cmd)->DebugEnd();
        (*cmd)->WriteTimestamp(queryPool, RHIPipelineStageBits::BottomOfPipe, pass->handle * 2 + 1);
        (*cmd)->End();
    }
```

在这里我们需要的主要是timing/时序信息，故仅需采样[Timestamp Queries](https://docs.vulkan.org/samples/latest/samples/api/hpp_timestamp_queries/README.html)。在每个pass真正录制前后引入`TopOfPipe`, `BottomOfPipe`timestamp引入即可（保守地）估计执行该pass需要**至少**用时，如图。

![image-20251201110059390](/image-foundation/image-20251201110059390.png)

## Profiler UI

值得注意的是，Pass的执行**开始**顺序一定，但**可能**不按该顺序完毕。也就是说，**即使**是在同一个queue上submit，收集到pass的时序是有可能“重叠”的——Queue上的执行并非线性！参见 [Ensure Correct Vulkan Synchronization by Using Synchronization Validation](https://www.lunarg.com/wp-content/uploads/2021/08/Vulkan-Synchronization-SIGGRAPH-2021.pdf)（下图）。

![image-20251202161527523](/image-foundation/image-20251202161527523.png)

同时，**Foundation 支持Async Compute**。两个Queue只会增大“可能并行”的可能性，若存在则在一行内绘制profile时序一定会重叠 - 我们需要分行。

### 区间分行

事实上，这个问题很*典*：理解成启止时间已知进行一种**调度**，是否感觉更加熟悉？参见[区间调度问题](https://en.wikipedia.org/wiki/Interval_scheduling) —— OS课上学过的[最早截止时间调度](https://zh.wikipedia.org/wiki/%E6%9C%80%E6%97%A9%E6%88%AA%E6%AD%A2%E6%97%B6%E9%97%B4%E4%BC%98%E5%85%88%E8%B0%83%E5%BA%A6)，这里就可以用到。$O(n logn)$贪心解法如下：

```c++
int ImProfilerAssignLanes(Span<ImProfilerSample> samples)
{
    std::sort(samples.begin(), samples.end());
    // Partition into lanes - work has chance to overlap on the GPU.
    int top = 0; PriorityQueue<Pair<size_t, int>, std::greater<>> Q(GLOBAL_ALLOC);
    for (auto& sample : samples)
    {
        if (!Q.empty() && Q.top().first <= sample.startTick)
        {
            sample.lane = Q.top().second; Q.pop();
            Q.emplace(sample.endTick, sample.lane);
        }
        else
        {
            sample.lane = top++;
            Q.emplace(sample.endTick, sample.lane);
        }
    }
    return top;
}
```

不过——算法有一个特性会带来一些麻烦：由于“抢占”的是结束时间最早的列，当区间是**连续**的时候，会出现区间在列之间**交替**的现象。对负载均衡而言可能是好事，但很看起来会很麻烦：Overlap情况并不清楚。

![image-20251202162600175](/image-foundation/image-20251202162600175.png)

但是**注意**。GPU内能够Overlap的工作是不会*太多*的：barrier等等都将限制能够并行的pass数量，也就是**行数**。如上图，也只有`Clear Overdraw..`和async compute部分的工作重在了一起。我们不妨考虑一下的贪心做法。这里，我们将尽可能向更早期的行加入区间：

```c++
int ImProfilerAssignLanes(Span<ImProfilerSample> samples)
{
    std::sort(samples.begin(), samples.end());
    // Partition into lanes - work has chance to overlap on the GPU.
    Map<int, size_t> Q(GLOBAL_ALLOC);
    for (auto& sample : samples)
    {
        for (int lane = 0;; lane++)
        {
            if (Q[lane] <= sample.startTick)
            {
                sample.lane = lane, Q[lane] = sample.endTick;
                break;
            }
        }
    }
    return Q.size();
}

```

对行数$K$，时间复杂度是$O(Knlogn)$。最坏情况下$K=n$会退化成一个$O(n^2)$的解法 - 但是事实告诉我们这是不可能的：如前文所述，$K$的值一般很小，基本可以认为是常数。这里的区间效果如下：

![image-20251202163945387](/image-foundation/image-20251202163945387.png)

最后，暂时的Profiler设计如图。注意：

- `CPU to Present`是一帧数据从CPU提交到被swapchain显示的**延迟** - swap越多越高。
- `Present to Present` 则为帧时间，也用于计算 FPS 和 CPU/GPUΔ
- GPU为完成所有command buffer的总用时

### 效果

效果如下。以下为开启Async Compute与关闭的时序表现：

![image-20251202165816737](/image-foundation/image-20251202165816737.png)

![image-20251202170449485](/image-foundation/image-20251202170449485.png)

当然，async compute带来的额外同步开销是不可避免的（注意图1中的“空白”部分）。同时在此工作*大多*串行，overlap效果被同步overhead抵消而反减——后期在pass更复杂，ALU/带宽工作分离程度更高时，优势将更加“显然”。
