---
author: mos9527
lastmod: 2025-12-03T21:20:52.771743
title: Foundation 施工笔记 【3】- 原生 Profiling 及早期优化
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

上一篇的实现仅仅是POC的水平，亟须优化。工具上，NV/AMD 也早有了非常成熟的toolchain，即[NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics)，[Radeon™ Developer Tool Suite](https://gpuopen.com/news/introducing-radeon-developer-tool-suite/)。CPU侧自己也很早集成了[tracy profiler](https://github.com/wolfpld/tracy)：但这些都离不开第三方工具的加持 - 接下来将介绍一些“引擎”内的profile手段，和利用它们及官方工具降低culling带来的overhead。

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

但是**注意**。GPU内能够Overlap的工作是不会*太多*的：barrier等等都将限制能够并行的pass数量，也就是**行数**。如上图，也只有`Clear Overdraw..`和async compute部分的工作重在了一起。我们不妨考虑一下*另一种*贪心做法。这里，我们将尽可能向更早期的行加入区间：

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

额外的，还有标配直方图等统计量。实现上很粗暴：环形缓冲+排序+binning。注意时间轴使用了$log_{10}$缩放：这里和[`tracy`](https://github.com/wolfpld/tracy) 一致。

![image-20251203212038626](/image-foundation/image-20251203212038626.png)

## Cull Indices

SM6.0加入的[Wave Intrinsics](https://github.com/Microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics)允许我们直接且*全平台*地利用GPU的SIMD潜力来构建高并发，低抢占的现代compute shader。（还有PS/fragment可用的 quad wave - 这里暂时不提）

回忆Task/Mesh shader本身也是compute模型的——这里也可以使用wave/subgroup intrinsics来减少同步上的一些开销。接下来着重Task Shader的剔除整理部分进行分析。

### LDS 实现


回顾当前整理可见meshlet id时 - 我们的实现依赖于LDS/groupshared内存，[work group barrier](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/groupmemorybarrierwithgroupsync)与[原子操作 InterlockedAdd](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/interlockedadd)

```glsl
if (visible) {
    uint sharedCount;
    InterlockedAdd(shared.meshletCount,1,sharedCount);
    shared.meshletIndices[sharedCount] = meshletID;
}
GroupMemoryBarrierWithGroupSync();
DispatchMesh(shared.meshletCount,1,1,shared);
```

- 对于前者，workgroup大小大于wave大小时会阻碍wave之间并行（相反的，小于或等于时会变得不必要——[驱动会优化掉这些barrier](https://github.com/microsoft/DirectX-Graphics-Samples/issues/140#issuecomment-211549126)）
- 对于后者，并发访问量大时会产生抢占而有退化成串行执行风险——不过，这在本例中并不显然。

可见，在这里利用wave函数的优化动机主要来自*后者*——对原子级同时发生的**写**而言，其开销是最好避免的。前者正如上文所述，*很可能*是不存在的开销。

**注意**：‘可能’取决于GPU真正跑这些shader使用的**Wave Size**。GPU上可以**任何 $2^n$ wave size**执行CS。特别地，N卡一般在一个wave跑$32$个线程，AMD*从前只会*跑$64$个——RDNA开始也是wave32了（图源：[RDNA Architecture - GPUOpen](https://gpuopen.com/download/RDNA_Architecture_public.pdf))。一个显然，但是消极的策略即为调整workgroup大小为$32$ - 这里暂不作考虑。

![image-20251203103648827](/image-foundation/image-20251203103648827.png)

### Wave 实现

Wave Instrinsic的出现允许我们不*完全*依赖与驱动的优化，而构造硬件wave大小无关的“零开销”同步。同时，一系列wave指令也可以加速原来本需要原子访问等的一些操作，这将在接下来的实现中继续解释。

来自 [【技术精讲】AMD RDNA™ 显卡上的Mesh Shaders（二）：优化和最佳实践](https://zhuanlan.zhihu.com/p/691937933) 也有提及：

> 在 amplification shader 执行每个 meshlet 剔除操作时，即检查预先确定的一组 meshlet 的每个元素的可见性时，我们建议每个 amplification shader 线程组至少处理 32 或 64 个元素（如 meshlet）。通过选择对应的 amplification shader 线程组的大小，可以在 RDNA™ 显卡上使用 `WavePrefixCountBits` 和 `WaveActiveCountBits` wave 内置函数来完成剔除。

需要注意的是，原文样例实现做了workgroup在同一个wave上运行的假设：

```glsl
if (visible)
{
    const uint idx = WavePrefixCountBits(visible);
    payload.meshletIds[idx] = meshletId;
}
const uint meshletCount = WaveActiveCountBits(visible);
DispatchMesh(meshletCount, 1, 1, payload);
```

`DispatchMesh`是以workgroup为同一单位的 - 意味着若workgroup大于wave大小（如WG=64, wave=32），payload访问将出现错误的重叠。wave大小无关的实现需要一些额外的工作：

```glsl
uint waveVisibleCount = WaveActiveCountBits(visible);
if (waveVisibleCount > 0){
    uint waveMeshletOffset = 0;
    if (WaveIsFirstLane())
        InterlockedAdd(shared.meshletCount, waveVisibleCount, waveMeshletOffset);
    waveMeshletOffset = WaveActiveMax(waveMeshletOffset);
    if (visible) {
        // Output the visible meshlet
        uint sum = WavePrefixCountBits(visible);
        shared.meshletIndices[waveMeshletOffset + sum] = meshletID;
    }
}
shared.instanceID = work.instanceID;
DispatchMesh(shared.meshletCount,1,1,shared);
```

- 我们假设workgroup大小比wave大小更小，或是正整数倍——因为wave和我们给定的workgroup大小均为$2^n$，因此假设是一定的。
- `WaveIsFirstLane()`保证一组，**不同**wave各自bump组内meshlet的数目**一次**
- `WaveActiveMax()`向整组广播属于**wave**的第一个meshlet - 接下来的`waveVisibleCount`个数的meshlet也将都属于这个wave
- `WavePrefixCountBits()` 相当于wave中lane/线程对于上述meshlet的位置。

操作数上（仅关注上述部分），假设$N$个可见meshlet，我们将LDS实现每组$N$次的`InterlockedAdd`降低到了$\frac{WorkGroupSize}{WaveSize}$的常数。最后同样`DispatchMesh`：注意（来自[DirectX Spec](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html#dispatchmesh-intrinsic)）

> This function, called from the amplification shader, launches the threadgroups for the mesh shader. This function must be called exactly once per amplification shader, must not be called from non-uniform flow control. The DispatchMesh call implies a GroupMemoryBarrierWithGroupSync(), and ends the amplification shader group’s execution.

`DispatchMesh`本身也包含一次barrier；注意到LDS实现中前一句也因此多余，上文wave实现中也因此没有额外同步。

### 结果

LDS和Wave实现的时序图如下，前后顺序对应。在AMD硬件和WorkGroup大小为64时，提升并不明显。

![image-20251203211743470](/image-foundation/image-20251203211743470.png)

![image-20251203211120058](/image-foundation/image-20251203211120058.png)
