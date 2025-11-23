---
author: mos9527
lastmod: 2025-11-22T21:42:38.239000+08:00
title: Foundation 施工笔记 【2】- Editor 场景加载与 GPU-Driven 渲染
tags: ["CG","Vulkan","Foundation","meshoptimizer"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface
迄今为止，Editor渲染方面实现仅处理了单个 Mesh 的最简单情况。接下来我们将正式引入**场景**加载。

在 CPU 上表达物体间关系的方案诸多——本质上也都是为实现某种Scene Graph。同时，本篇体裁内的最终目标仅仅是为了渲染 [GLTF](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) 场景。此后内容也将围绕其结构进行展开。

### 坐标系细节

我们约定使用**右手系**坐标系统。即在相机视角，$+X$为右，$+Y$为上，$-Z$为前。这也是Blender等的默认坐标系（$+Z$指向相机“内”）

![image-20251122102948474](/image-foundation/image-20251122102948474.png)

### 透视及视角矩阵

- 注：$f$与$a$

$$
f = \frac{1}{\tan(fov_y / 2)}，a = \text{宽高比}
$$

- 我们想要让$z$轴上，**相机处**  $ z= z_{n}$ 的NDC为 $z_{ndc} = 1$； **无穷远**，或一定$z_{f}$ 处 $z_{ndc} = 0$
- 理由是充分的。简要地，$[1,0]$映射可大幅改善near plane附件深度精度。详细解释及动机还请参考：
  - https://mynameismjp.wordpress.com/2010/03/22/attack-of-the-depth-buffer/
  - https://developer.nvidia.com/content/depth-precision-visualized

- 下面直接给出改配置对应透视矩阵，可由代入计算$z=z_n,z_{NDC}=1$, $z = z_f, z_{NDC} = 0$易得

$$
P = \begin{bmatrix}
\frac{f}{a} & 0 & 0 & 0 \newline
0 & f & 0 & 0 \newline
0 & 0 & \frac{z_n}{z_f - z_n} & \frac{z_f z_n}{z_f - z_n} \newline
0 & 0 & -1 & 0 
\end{bmatrix}
$$

- 代入$z_f = +\inf$即可得无穷远版本

$$
P = \begin{bmatrix}
\frac{f}{a} & 0 & 0 & 0 \newline
0 & f & 0 & 0 \newline
0 & 0 & 0 & z_n \newline
0 & 0 & -1 & 0 
\end{bmatrix}
$$

- 对Vulkan，Viewport的Y轴需要翻转

  ![img](/image-shading-reverse/viewports_gl_vk.png)

  可以考虑直接翻转投影$Y$轴（翻转$P[1][1]$) 。但这样做在其他 RHI 平台又需要二次调整 - 其次，未经过透视变换的量（如法线）与其次空间的position量会产生坐标系不一致——可以理解为什么[Unity 会用postprocess](https://mos9527.com/posts/pjsk/shading-reverse-part-1/#present%E7%BF%BB%E8%BD%AC)仅仅善后Framebuffer

  在Vulkan（目前我们唯一支持的 RHI），翻转Viewport/Framebuffer很容易；仅需在`vkSetViewport`做翻转并启用`shaderDrawParameters` extension即可。我们RHI中的实现如下：

```c++
RHICommandList& VulkanCommandList::SetViewport(float x, float y, float width, float height, float depth_min, float depth_max, bool flipY) {
  CHECK(mAllocator && "Invalid command list states.");
  if (flipY)
  {
      y = height - y;
      height = -height;
  }
  vk::Viewport viewport{ x, y, width, height, depth_min, depth_max };
  mCommandBuffer.setViewport(0, viewport);
  return *this;
}
```

​	这样做，我们的坐标系和Blender，`OpenGL`与`GLTF`将完全一致。对视角矩阵的构造如下：$pos=(0,0,0),quat(xyzw)=(0,0,0,1)$的原始变换下，相机将朝$-Z$方向看。

```c++
// Forward = -Z
inline mat4 viewMatrixRHReverseZ(vec3 pos, quat rot)
{
    mat4 view = mat4_cast(rot);
    view[3] = vec4(pos.x,pos.y,pos.z,1.0f);
    return inverse(view);
}
```

## GPU Scene 

传统地，渲染管线需要在 CPU 组织DrawCalls - 变换，剔除等操作在上传前完成。自 DX11 世代起出现的 Compute Shader模型及`ExecuteIndirect`使得GPU上生成DC成为可能。

遍历场景的任务移交 GPU，很显然，在图上去跑一遍 DFS 的利用率会非常低下：callstack很深且串行，最多只能利用根节点个 GPU 线程。**扁平化**，**基于数据**的场景表达方式更适合 GPU 计算。这点包括 [Unreal Engine](https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Renderer/Private/GPUScene.h) 在内的大多数现代引擎都有采用。

当然，CPU上的场景遍历也是不可避免的：节点间的Transform变换需要更新子节点的global transform，任何编辑操作也将在CPU上完成后上传至GPU。不过，粒度和频率上都将大幅减少。后期也将讨论利用空间数据结构加速CPU侧场景交互的方案。

### Buffer 设计细节
场景仅申请`PrimitiveBuffer`和`InstanceBuffer`两部分缓冲区，区分依据为且仅为**更新频率**。
- `PrimitiveBuffer`：存储 Mesh 几何信息及相关元数据。仅在场景加载或 Mesh 变更时更新
  - **真正的**数据流送（传入**且**传出）暂不在本篇考虑 - 实现上这将需要一个靠谱的 GPA (General Purpose Allocator) 系统。暂时我们使用简单的线性/Bump Allocator。
  
- `InstanceBuffer`：存储可能逐帧更新的实例信息。细节上值得注意的有：
  - 数据结构大小均一致。注意Shader中没有union的对应概念（[Slang支持指针](https://shader-slang.org/slang/user-guide/convenience-features.html#:~:text=the%20source%20type.-,Pointers%20(limited),-Slang%20supports%20pointers) - 但这并非全平台且限制良多，如并不支持对局部变量取pointer），故不同类型实现为 "tag + user data"结构。
  
  - 实现上为**Ring Buffer**。Swapchain会有$N$个Backbuffer - 这意味着最坏情况下，在`Acquire()`一帧后会有$N-1$帧的**历史**场景数据仍在被GPU使用。
    避免[ROW（Read On Write）风险](https://www.lunarg.com/wp-content/uploads/2021/08/Vulkan-Synchronization-SIGGRAPH-2021.pdf)是必须的 - 方案之一*可以*是为每个Backbuffer分配一份InstanceBuffer。很显然这很浪费：场景数据量将仅能以上限大小$M$以$O(MN)$的代价存储。环形缓冲则允许‘有多少用多少’ - 代价则为$O(kN)$，其中$k$为场景中实例数。
  - 值得注意的是，CPU写入overrun情况将较难调试。但避免的充分必要条件即为预留$MN$空间，假设上界已知。
  - 绑定时，`vkCmdBindDescriptorSets`也允许传入[Descriptor Dynamic Offset](https://docs.vulkan.org/guide/latest/descriptor_dynamic_offset.html)。Shader上可直接就通常的Uniform Buffer方式访问数据，无需额外计算偏移。或者，手动传入offset读取也可取——鉴于该特性的'Vulkanism' - 出于对未来 RHI 兼容性考虑选择后者。
  
  

## "Draw Scene" GPU-command

我们将不考虑传统Vertex管线而直接实现Meshlet整套 GPU Driven 管线。子标题名来自 [GPU-Driven Rendering Pipelines - Sebastian Aaltonen SIGGRAPH 2015](https://www.advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf)

回忆Mesh Shader管线可选的前置Task/Amplifcation Shader Stage——生成Meshlet "Drawcall" 本身可以来自这个几乎是Compute Shader（除仅能在Graphics Queue上跑）的环节进行。命令对应 [DrawMeshTasks](https://mos9527.com/Foundation/classFoundation_1_1RHI_1_1RHICommandList.html#ae4fde8bf43a426dfacc15012a122e272) （Vulkan中的`VkCmdDrawMeshTasksEXT`)

鲜为人知的还有 [DrawMeshTasksIndirect](https://mos9527.com/Foundation/classFoundation_1_1RHI_1_1RHICommandList.html#ab24bede0c6c26faa021dc0638fc40a25) （Vulkan中的`VkCmdDrawMeshTasksIndirectEXT`) —— 这里可以从（或许是）Compute Shader 生成的 command buffer 从 GPU（驱动）dispatch Task Shaders。两层'amplification'的自由度可谓相当大；同时，若使用Compute生成命令，还将允许Async Compute的实现（不需要 Graphics Queue！），和真正的 Graphics Queue 进行一定并行overlap。

### 完整 Dispatch Chain

综上，我们完整的Dispatch链如下，处理对象粒度递增。(CS: Comptue Shader)

| CS `Dispatch`                                                | CS `Submit`                                                | Task `DrawMeshTasksIndirect`                          | Task->Mesh `DispatchMesh`                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| **实例**剔除；产生连续**存储非空** Task 命令及计数     | 产生 Indirect Task Dispatch 命令    | **Meshlet**的**自适应 LOD 选择**+**剔除**，并产生`DispatchMesh`在同一Pipeline进行 | **三角形**剔除，继续到Fragment/Pixel Stage（略） |

需要注意的是，Task-Mesh属于同一管线。故Task中的`DispatchMesh`**仅能为0或1个**。为此在`CS Submit`时可进行分组，对实例$N$ 个 Meshlet产生$\lceil \frac{N}{WorkGroupSize}\rceil $个 Task Shader Indirect。

最后——在 CPU 上，准备好前置 Buffer 之后的Dispatch仅需一句[DrawMeshTasksIndirect](https://mos9527.com/Foundation/classFoundation_1_1RHI_1_1RHICommandList.html#ab24bede0c6c26faa021dc0638fc40a25)

### 实例化效果

在**不进行任何剔除**的情况下，性能指标及效果如图。共$10^3$的斯坦福小兔子实例，模型均复用上述PrimitiveBuffer同一指针。

![image-20251123101132828](/image-foundation/image-20251123101132828.png)
