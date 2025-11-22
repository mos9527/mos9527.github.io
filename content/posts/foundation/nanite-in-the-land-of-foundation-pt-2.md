---
author: mos9527
lastmod: 2025-11-20T21:42:38.239000+08:00
title: Foundation 施工笔记 - 复现 Nanite 虚拟几何体【2】
tags: ["CG","Vulkan","Foundation","meshoptimizer"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface
迄今为止，渲染方面实现仅处理了单个 Mesh 的最简单情况。本篇将着重于**场景渲染**与**剔除操作**。

## For Instance

在 CPU 上表达物体间关系的方案诸多——本质上也都是为实现某种Scene Graph。同时，本篇体裁内的最终目标仅仅是为了渲染 [GLTF](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) 场景。此后内容也将围绕其结构进行展开。

### 坐标系

我们约定使用**右手系**坐标系统。即在相机视角，$+X$为右，$+Y$为上，$-Z$为前。这也是Blender等的默认坐标系（$+Z$指向相机“内”）

![image-20251122102948474](/image-foundation/image-20251122102948474.png)

#### 透视及视角矩阵

- 注：$f$与$a$

$$
f = \frac{1}{\tan(fov_y / 2)}，a = \text{宽高比}
$$

- 我们想要让$z$轴上，**相机处**  $ z= z_{n}$ 的NDC为 $z_{ndc} = 1$； **无穷远**，或一定$$z_{f}$$处 $z_{ndc} = 0$
- 理由是充分的。简要地，$[1,0]$映射可大幅改善near plane附件深度精度。详细解释及动机还请参考：
  - https://mynameismjp.wordpress.com/2010/03/22/attack-of-the-depth-buffer/
  - https://developer.nvidia.com/content/depth-precision-visualized

- 下面直接给出改配置对应透视矩阵，可由代入计算$z=z_f,z_{NDC}=1$, $z = z_f, z_{NDC} = 0$易得

$$
P = \begin{bmatrix}
\frac{f}{a} & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & \frac{z_n}{z_f - z_n} & \frac{z_f z_n}{z_f - z_n} \\
0 & 0 & -1 & 0 
\end{bmatrix}
$$

- 代入$z_f = +\inf$即可得无穷远版本

$$
P = \begin{bmatrix}
\frac{f}{a} & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 0 & z_n \\
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

