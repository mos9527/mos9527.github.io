---
author: mos9527
lastmod: 2025-11-20T21:42:38.239000+08:00
title: Foundation 施工笔记 - 实现类 Nanite 虚拟几何体（1）
tags: ["CG","Vulkan","Foundation","meshoptimizer"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static
---

## Preface

预谋多时而虽迟但到的 [Foundation](https://github.com/mos9527/Foundation/) （名称暂定）博文系列。

特别中二的名字以外，项（天）目（坑）并非最近开始的工作（迄今 331 Commit）。期间诸多内容，如各类无锁 (Lock-free) 数据结构、RHI、RenderGraph细节、Shader 反射等，也都是实现期间*相当*想留一笔的工作...

嘛、反正也是后期也不得不记的事，不如梭哈开篇为好（）那么就开始吧？

**注：** Foundation 文档：https://mos9527.com/Foundation/

## Mesh Shader?

**注：** **参考性内容 - 酌情跳过。**深入了解，还请参阅以下文档：

- [Introduction to Turing Mesh Shaders - NVIDIA](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/)
- [【技术精讲】AMD RDNA™ 显卡上的Mesh Shaders（一）： 从 vertex shader 到 mesh shader](https://zhuanlan.zhihu.com/p/691467498)

脱离 Fixed Function 的整套`Input Assembler/Vertex+Geometry`管线，繁而就简：`Fragment/Pixel`之前，Compute模式的`Mes` 足矣且充分地代替这些功能。

![img](/image-foundation/mesh-shader-comparison-nv.jpg)

额外的，前置还可以有`Task/Amplification`环节生成 Mesh Shader WorkGroup（同时，Task Shader 也支持 Indirect Dispatch）。很显然，这样的架构是相当适合当代 GPU-Driven 渲染器的实现的。

![img](/image-foundation/meshlets-pipeline2-nv.png)

最小单元（Primitive）仍然还是三角形 - 为饱和CS单元利用率，Mesh Shader同时引入了Meshlet - 依据一定指标对Mesh进行**分区** - 直接地减小overhead， 间接的提供**压缩**（index buffer压到N个micro buffer/uint8)，**剔除**机会，和...

### Enter Nanite

作为 UE5 的招牌特性，Nanite 利用新管线的高粒度与可控性实现了**消除 LOD 过渡**的任务。

<img src="/image-foundation/nanite-spline-mesh-example-1.png" alt="Nanite Virtualized Geometry in Unreal Engine | Unreal Engine 5.7  Documentation | Epic Developer Community" style="zoom:25%;" />

除此之外，其实现对**流送/Streaming**的支持也实现了**虚拟几何体**而无视显存限制等等的优良特性，免费**镶嵌/Tesselation**，和对极高面数的网格支持，**免去Batching/Instancing**...

同时，业内，包括 [Unity / 团结引擎 - 虚拟几何体](https://docs.unity.cn/cn/tuanjiemanual/Manual/VirtualGeometry.html)， [RE Engine - Is Rendering Still Evolving?](https://www.capcom-games.com/coc/2023/en/session/15/) ，[Remedy Northlight -Alan Wake 2: A Deep Dive into Path Tracing Technology](https://www.nvidia.com/en-us/on-demand/session/gdc24-gdc1003/?playlistId=playList-821861a9-571a-4073-abab-d60ece4d1e49)，及业余空间中的 [bevy](https://jms55.github.io/posts/2024-06-09-virtual-geometry-bevy-0-14/#future-work)  等等也已在自己的管线实现了类似的技术。

![image-20251120215946457](/image-foundation/tuanjie-virtual-geometry-doc.png)

又炫又高效...那给自己的玩具渲染器实现的理由，岂不是已经**超级**充分？

## Hello Meshlets

现在，后退几步。即使不做 LOD，你的**Meshlet**怎么而来？

回顾前文 - Meshlet的本质是网格顶点的**分区**。从无向，联通（假设凸包）的几何体面上拆下面来...事实上，正是一个**最小割/图分区**问题。

![metis-graph-partitioning](/image-foundation/metis-graph-partitioning.png)

以*某种指标*出发，将整张由网格顶点构成的图分割成N份，优化这个指标...

当然，这里可没有源点汇点，不能归纳为费用流一类问题。事实上，这类问题属于 NP 难范畴。多项式时间内可解决的方案也均为启发式。如 Nanite paper 中提及的 [METIS](https://github.com/KarypisLab/METIS) - 用于后文阐述 Meshlet 建图，但同时也适用 Meshlet 自身生成。

实现上，自己选择了 - [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer)。实不相瞒，这个库的存在可以说是是整篇文章存在的理由（拜谢 Arseny！）

*（还记得使用初期发现的一个[文档方面小 Issue](https://github.com/zeux/meshoptimizer/issues/962)...有被大佬响应的及时程度震撼到）*

### 实现

利用`meshoptimizer`划分 Meshlet 本身相当容易。以下为实现部分节选。

注意以下关系成立：`outMeshletVertices[outMeshletTriangles[u]] = indices[v]`, 即每个Meshlet都含一个"微型"Index Buffer。同时注意到，存储`outMeshletTriangles`仅需对应每个Meshlet有限的顶点（<256个），存储时即使用`uint_8`。

```c++
...
size_t maxMeshlets = meshopt_buildMeshletsBound(indices.size(), kMeshletMaxVertices, kMeshletMaxTriangles);
// Worst bounds
outMeshletVertices.resize(maxMeshlets * kMeshletMaxVertices), outMeshletTriangles.resize(maxMeshlets * kMeshletMaxTriangles);
Vector<meshopt_Meshlet> meshoptMeshlets(maxMeshlets, outMeshlet.get_allocator());
size_t meshlets =
    meshopt_buildMeshlets(meshoptMeshlets.data(), outMeshletVertices.data(), outMeshletTriangles.data(),
                          indices.data(), indices.size(), reinterpret_cast<const float*>(&vertices[0]),
                          vertices.size(), sizeof(Vertex), kMeshletMaxVertices, kMeshletMaxTriangles,
                          0.25f // As recommended by the docs
    );
meshoptMeshlets.resize(meshlets);
{
    const auto& [vertexOffset, triangleOffset, vertexCount, triangleCount] = meshoptMeshlets.back();
    outMeshletVertices.resize(vertexOffset + vertexCount);
    outMeshletTriangles.resize(triangleOffset + triangleCount * 3);
}
...
```

渲染部分，简单起见 - 直接使用单个 Mesh Shader 与 `DrawMeshTasks` (对应 `vkDrawMeshTasksEXT`) 命令足矣。Dispatch数目即为Meshlet数目，即一个 WorkGroup 对应一个 Meshlet

Foundation 内的实现如下。详见 https://github.com/mos9527/Foundation/blob/vulkan/Examples/MeshShaderBasic.cpp

```c++
...
renderer->CreatePass(
    "Mesh Shader", RHIDeviceQueueType::Graphics, 0u,
    [=](PassHandle self, Renderer* r)
    {
        r->BindBackbufferRTV(self);
        r->BindTextureDSV(self, zbufferHandle,
                              {.format = RHIResourceFormat::D32SignedFloat,
                              .range = RHITextureSubresourceRange::Create(RHITextureAspectFlagBits::Depth)});
        r->BindShader(self, RHIShaderStageBits::Mesh, "meshMain", "data/shaders/MeshShaderBasicMesh.spv");
        r->BindShader(self, RHIShaderStageBits::Fragment, "fragMain", "data/shaders/MeshShaderBasicFrag.spv");
        // NOTE: globalParams is introduced by slang compiler and is currently not customizable
        //       for uniform storage members
        r->BindBufferUniform(self, uboHandle, RHIPipelineStageBits::MeshShader, "globalParams");
        r->BindBufferStorageRead(self, meshHandle,
                                 RHIPipelineStageBits::MeshShader | RHIPipelineStageBits::FragmentShader, "mesh");
    },
    [&](PassHandle self, Renderer* r, RHICommandList* cmd)
    {
        auto const& img_wh = r->GetSwapchainExtent();
        auto* uboData = r->DerefResource(uboHandle).Get<RHIBuffer*>();
        cmd->UpdateBuffer(uboData, 0, AsBytes(ubo));
        r->CmdBeginGraphics(self, cmd, img_wh);
        r->CmdSetPipeline(self, cmd);
        // Simplest dispatch - spawn meshlets one by one to each Mesh Shader WG
        // We don't need a task shader - if unbound, DrawMeshTasks dispatches
        // Mesh Shader workgroups effectively directly.
        cmd->SetViewport(0, 0, img_wh.x, img_wh.y)
            .SetScissor(0, 0, img_wh.x, img_wh.y)
            .DrawMeshTasks(ubo.mesh.lod[0].meshletCount, 1, 1)
            .EndGraphics();
    });
```

Shader 部分也将省略剔除等步骤  - 注意 Slang Shader 中对应 HLSL 的一些语义：

![illustration of the relationship between dispatch, thread groups, and threads](/image-foundation/threadgroupids.png)

注意 Work Group 和 Meshlet 在此有一对一的关系。但同时这不意味着一个 Thread 仅对应一个顶点/三角形：参见 `uint i = tid; i < vtxCount; i += kWGSize` 部分。

```glsl
static const size_t kWGSize = 64;
[outputtopology("triangle")]
[numthreads(kWGSize, 1, 1)]
void meshMain(
    // See graphics reference available at:
    // * https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sv-groupindex
    in uint gid: SV_GroupID,
    in uint tid: SV_GroupIndex,
    OutputVertices<V2F, 96> verts,
    OutputIndices<uint3, 96> triangles,
) {    
    GSMeshLOD lod = globalParams.mesh.lod[0];
    // Each WG of mesh shader processes *exactly* one meshlet per dispatch
    FMeshlet meshlet = mesh.Load<FMeshlet>(lod.meshletOffset + gid * sizeof(FMeshlet));
    uint32_t vtxBase = lod.meshletVtxOffset + meshlet.vtxOffset * sizeof(uint32_t);
    uint32_t vtxCount = meshlet.vtxCount;
    uint32_t triBase = lod.meshletTriOffset + meshlet.triOffset * sizeof(uint8_t);
    uint32_t triCount = meshlet.triCount;
    SetMeshOutputCounts(vtxCount, triCount);
    // Each thread in the WG processes multiple vertices/triangles
    for (uint i = tid; i < vtxCount; i += kWGSize) {
        uint ind = mesh.Load<uint32_t>(vtxBase + i * sizeof(uint32_t));
        FVertex vtx = mesh.Load<FVertex>(globalParams.mesh.vtxOffset + ind * sizeof(FVertex));
        V2F v2f;
        v2f.pos = mul(globalParams.mvp, float4(vtx.position, 1.0));
        v2f.meshlet = gid;
        verts[i] = v2f;
    }
    for (uint i = tid; i < triCount; i += kWGSize) {
        MIndex tri = mesh.Load<MIndex>(triBase + i * sizeof(MIndex));
        triangles[i] = uint3(tri.v0, tri.v1, tri.v2);
    }
}
```

Fragment部分直接显示各Meshlet编号。哈希函数来自 https://github.com/zeux/niagara/

```glsl
struct V2F
{
    float4 pos : SV_POSITION;
    uint32_t meshlet : ID;
};
uint hash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}
float3 hash3(uint a){
    a = hash(a);
    return float3(float(a & 255), float((a >> 8) & 255), float((a >> 16) & 255)) / 255.0;
}
struct Fragment
{
    float4 color : SV_Target;
};
[shader("fragment")]
Fragment fragMain(V2F input)
{
    Fragment output;
    output.color = float4(hash3(input.meshlet), 1.0);
    return output;
}

```

载入[斯坦福小兔兔](https://faculty.cc.gatech.edu/~turk/bunny/bunny.html)。效果如图。

![Screenshot_20251120_223343](/../../Pictures/Screenshots/Screenshot_20251120_223343.png)

## LOD Group

接下来，就 [Nanite - A Deep Dive](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf) LOD 建图部分进行实现。

![image-20251120221435924](/image-foundation/nanite-paper-46.png)

借论文原图 - 这里的任务即为

- **选择**某些区块（可以是三角面本身，也可以是已经划分的cluster）
- **合并**并**简化**这些区块。简化任务即与传统 LOD 生成一致，同样也是相当邪门的话题...
- **锁定**分界部分并**分割**。锁定为避免不同LOD边界过渡不平滑；分割如选4分2，即可构造下一级LOD
- 区分下来的这些区块均为区分原cluster的子节点

最后，重复到只剩**一级**LOD为止，即到达根节点。

对**分组**任务，是否很熟悉？没错 - 这里同样也是一个**图划分**任务。只不过多了需要**迭代**+**锁边**的需求。

![image-20251120230036279](/image-foundation/nanite-paper-48.png)

### 实现

出于机缘巧合，自己研究初期，Arseny (`meshoptimizer` 作者！) 正好发布了这篇博文：  [Billions of triangles in minutes - zeux.io](https://zeux.io/2025/09/30/billions-of-triangles-in-minutes/)

除了极大程度地避免自己走弯路以外，作者同期也将自己的 LOD 建图实现分离并给予 API 食用 - [clusterlod.h - meshoptimizer](https://github.com/zeux/meshoptimizer/blob/master/demo/clusterlod.h)

官方 Demo 并未给出实时渲染实现。不过，LOD 建图部分可谓**相当**完善且好用 - 集成可谓轻而易举。

#### Cluster 构造

**注：** 个人实现 - 非官方。有误还有烦请指正！

首先，我们给Meshlet/Cluster分组。加入`group`表示其隶属的组编号，`refined`表示**组**的父组编号

```c++
struct FMeshlet // @ref meshopt_Meshlet
{
    /* meshlet group */
    /* ID of the @ref FLODGroup this meshlet belongs to in a hierarchy */
    uint32_t group;
    /* ID of the @ref FLODGroup (with more triangles) that produced this meshlet during simplification (parent). ~0u if original geometry */
    uint32_t refined;

    /* meshlet */
    /* offsets within meshletVtx and meshletTri arrays with meshlet data */
    uint32_t vtxOffset; // in vertices
    uint32_t triOffset; // in indices (3*triangles)
    /* number of vertices and triangles used in the meshlet; data is stored in consecutive range defined by offset and
     * count */
    uint32_t vtxCount;
    uint32_t triCount;

    /* bounds */
    float4 centerRadius; // (x,y,z,r)
    float4 coneAxisAngle; // (x,y,z,cos(half solid angle))
    float3 coneApex; // (x,y,z)
};
```

分组的目的即为$$O(1)$$决定组别内所有Meshlet是否值得渲染（满足某种错误指标，等等）。每组数据如下：

```c++
struct FLODGroup // @ref clodGroup
{
    // DAG level the group was generated at
    int depth;

    // sphere bounds, in mesh coordinate space
    float3 center;
    float radius;

    // combined simplification error, in mesh coordinate space
    float error;
};
static_assert(sizeof(FLODGroup) == 24);
```

#### 选择节点

值得注意的是，我们并没有直接地表示**节点间的边**。事实上这是不需要的。

![image-20251120231516372](/image-foundation/nanite-paper-66.png)

考虑渲染中我们需要做的任务：

- **选择**是否渲染是检查**组**的错误因子是否达标。注意组越深越简化，假设（实际如此）错误系数**单调递增**...
- 组**PASS**，意味着其隶属直接子节点（一层）也被渲染，无须渲染后续组别
- 组**REJECT**，意味着错误系数太高，需要选择更细致组别

还记得前文我们记录了Meshlet/Cluster自己所属的组与父组吗？利用单调性，我们**任意地**选择一个Meshlet单元：

- 记录当前视角，**当前组**(`group`)的错误系数为$$u$$,**父亲组**(`refined`)的错误系数为$$v$$。满足 $$u \ge v$$（**注：**不同于原论文，这里当前，父亲（$$u,v$$）的关系倒置）
- 选定一个阈值$$t$$，错误低于者**PASS**
- 当且仅当$$u > t, v <= t$$，渲染**当前组**

可以发现，这样可以做到选择**【且仅选择】满足阈值的【下界】的终端节点**，正为我们想要的。而且很显然，这个**任意**操作本质**并行**，在Compute/Task Shader实现也将十分容易。

#### 正式建图

DAG结构很简单，如下。

```c++
struct DAG
{
    struct Cluster
    {
        uint32_t group{~0u}; // ID of the FLODGroup this cluster belongs to
        uint32_t refined{~0u}; // ID of the FLODGroup (with more triangles) that produced this cluster during simplification (parent). ~0u if original geometry
        Vector<uint32_t> indices;
        Cluster(Allocator* alloc) : indices(alloc) {}
    };
    Vector<Cluster> clusters; // Note: scratch buffer
    // -- final DAG data
    Vector<FLODGroup> groups; // group error bounds
    Vector<FMeshlet> meshlets; // meshlets built from all clusters
    Vector<uint32_t> meshletVtx;
    Vector<uint8_t> meshletTri;
    DAG(Allocator* alloc) : clusters(alloc), groups(alloc), meshlets(alloc), meshletVtx(alloc), meshletTri(alloc) {}
} dag;
```

注意到`Cluster`内容是不需要上传的，因为接下来我们会利用结果直接生成Meshlet本身。建图过程如下：

```c++
clodBuild(config, mesh,
              [&](clodGroup group, const clodCluster* clusters, size_t cluster_count) -> int
              {
                  dag.groups.push_back(FLODGroup{
                      .depth = group.depth,
                      .center = {group.simplified.center[0], group.simplified.center[1], group.simplified.center[2]},
                      .radius = group.simplified.radius,
                      .error = group.simplified.error});
                  for (size_t i = 0; i < cluster_count; i++)
                  {
                      auto& cluster = clusters[i];
                      auto& lvl = dag.clusters.emplace_back(vertices.get_allocator().mResource);
                      lvl.group = dag.groups.size() - 1u, lvl.refined = cluster.refined;
                      auto& ind = lvl.indices;
                      ind.insert(ind.end(), cluster.indices, cluster.indices + cluster.index_count);
                  }
                  return 0;
              });
```

注意到`clodBuild`的callback在group顺序上有单调递增的保证。最后回顾前文Meshlet构造过程，我们也就此index buffer构造micro index buffer。过程如下：

```c++
// Done - build meshlets for each cluster
size_t numIndices = 0;
for (auto& cluster : dag.clusters)
    numIndices += cluster.indices.size();
// Worst bounds
dag.meshletVtx.resize(numIndices), dag.meshletTri.resize(numIndices);
uint32_t* vtx = dag.meshletVtx.data();
uint8_t* tri = dag.meshletTri.data();
dag.meshlets.reserve(dag.clusters.size());
for (auto& cluster : dag.clusters)
{
    FMeshlet meshlet{
        .group = cluster.group,
        .refined = cluster.refined,
        .vtxOffset = static_cast<uint32_t>(vtx - dag.meshletVtx.data()),
        .triOffset = static_cast<uint32_t>(tri - dag.meshletTri.data()),
    };
    size_t unique = clodLocalIndices(vtx, tri, cluster.indices.data(), cluster.indices.size());
    vtx += unique, tri += cluster.indices.size();
    meshlet.vtxCount = unique, meshlet.triCount = cluster.indices.size() / 3;
    // Compute bounds
    meshopt_Bounds bounds = meshopt_computeMeshletBounds(
        &dag.meshletVtx[meshlet.vtxOffset], &dag.meshletTri[meshlet.triOffset], meshlet.triCount,
        reinterpret_cast<const float*>(&vertices[0]), vertices.size(), sizeof(FVertex));
    meshlet.centerRadius = float4(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius);
    meshlet.coneAxisAngle =
        float4(bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2], bounds.cone_cutoff);
    meshlet.coneApex = float3(bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2]);
    dag.meshlets.push_back(meshlet);
}
```

最后上传至 GPU - 详见 https://github.com/mos9527/Foundation/blob/vulkan/Examples/MeshShaderHierarchicalLOD.cpp

## Discrete LOD

在实现view-dependent自动LOD之前，不妨尝试直接利用group对应的图内深度，实现传统的 LOD 过渡。

Shader 选择仅需一句话：

```glsl
...
uint32_t groupBase = globalParams.mesh.groupOffset + meshlet.group * sizeof(FLODGroup);
FLODGroup lodGroup = mesh.Load<FLODGroup>(groupBase);
if (lodGroup.depth != globalParams.cutDepth) { // <- Cull depth    
    SetMeshOutputCounts(0, 0);
    return;
}
...
```

效果如下，左至右上至下 LOD 层次递增。（注：帧率差距在于笔记本没插电+debug build；如未说明性能指标将实际相近）

这一部分的完整实现见： https://github.com/mos9527/Foundation/commit/c15200bbf32c8a46cb0982f5da0a7615a7c02581

| ![image-20251121085148908](/image-foundation/image-20251121085148908.png) | ![image-20251121085159730](/image-foundation/image-20251121085159730.png) | ![image-20251121085208195](/image-foundation/image-20251121085208195.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20251121085219914](/image-foundation/image-20251121085219914.png) | ![image-20251121085232140](/image-foundation/image-20251121085232140.png) | ![image-20251121085243437](/image-foundation/image-20251121085243437.png) |
| ![image-20251121085252867](/image-foundation/image-20251121085252867.png) | ![image-20251121085300264](/image-foundation/image-20251121085300264.png) | ![image-20251121085306427](/image-foundation/image-20251121085306427.png) |

## View-Dependent LOD

等我修好hugo inline latex 再写- -
