---
author: mos9527
lastmod: 2024-01-05T08:00:54.681672+08:00
title: Project SEKAI 逆向（7）：3D 模型提取
tags: ["逆向","unity","pjsk","api","project sekai","miku","unity","3d","cg","blender"]
categories: ["Project SEKAI 逆向", "逆向","Tech Art"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Project SEKAI 逆向（7）：3D 模型

### 1. 文件结构

- 目前发现的 3D 模型基本都在 `[ab cache]/live_pv/model/` 下

![image-20240105080707360](/assets/image-20240105080707360.png)

![image-20240105080841622](/assets/image-20240105080841622.png)

初步观察：

- (1) Body 即目标模型；当然，作为skinned mesh，而且带有blend shapes，处理细节会很多；后面继续讲
- (2) 处的 `MonoBehavior` 就其名字猜测是碰撞盒
- (3) 处的几个 `Texture2D` 则作为texture map

### 2. 模型收集

利用`sssekai`取得数据的流程在（5）中已有描述，这里不再多说

首先整理下根据mesh发现的**数据需求**

- (1) **Static Mesh**

pjsk发现的所有mesh在相应assetbundle中会有1个或更多`GameObject`的ref；对于这些ref，static mesh会出现在`m_MeshRenderer`之中

其他细节暂且不说；因为做 Skinned Mesh 导入时都是我们要处理的东西

- (2) **Skinned Mesh**

不同于static mesh,这些ref会出现在`m_SkinnedMeshRenderer`之中

同时，我们也会需要**骨骼结构的信息**；bone  weight以外，也需要bone path（后面会用来反向hash）和transform

- (3) **Blend Shapes**

  这些可以出现在static/skinned mesh之中；如果存在，我们也会需要blend shape名字的hash，理由和bone path一致

  加之，Unity存在aseetbundle中动画path也都是crc，blendshape不是例外

**总结:**

- (1) 所以对于static mesh,搜集对应`GameObject`即可

- (2) 对于skinned mesh，同时也需要构造bone hierarchy（就是个单根有向无环图啦），并且整理vertex权重；

  则需要收集的，反而只是bone的transform而已；transform有子/父节点信息，也有拥有transform的`GameObject`的ref

- (3) 的数据，在(1)(2)中都会有

### 3. 模型导入

当然，这里就不考虑将模型转化为中间格式了（i.e. FBX,GLTF）

利用Blender Python，可以直接给这些素材写个importer

实现细节上，有几个值得注意的地方：

- Unity读到的mesh是triangle list

- Blender使用右手系，Unity/Direct3D使用左手系

|坐标系|前|上|左|
|-|-|-|-|
|Unity|   Z     |   Y  |   X|
|Blender|  -Y     |   Z  |  -X|

  - 意味着对向量需要如下转化

    $\vec{V_{blender}}(X,Y,Z) = \vec(-V_{unity}.X,-V_{unity}.Z,V_{unity}.Y)$

  - 对四元数XYZ部分

    $\vec{Q_{blender}}(W,X,Y,Z) = \overline{\vec(V_{unity}.W,-V_{unity}.X,-V_{unity}.Z,V_{unity}.Y)}$
    
- Unity存储vector类型数据可能以2,3,4或其他个数浮点数读取，而vector不会额外封包，需要从flat float array中读取

  意味着需要这样的处理

  ```python
         vtxFloats = int(len(data.m_Vertices) / data.m_VertexCount)
         vert = bm.verts.new(swizzle_vector3(
              data.m_Vertices[vtx * vtxFloats], # x,y,z
              data.m_Vertices[vtx * vtxFloats + 1],
              data.m_Vertices[vtx * vtxFloats + 2]            
          ))
  ```

  嗯。这里的`vtxFloats`就有可能是$4$. 虽然$w$项并用不到

- 对于BlendShape, blender并不支持用他们修改法线或uv;这些信息只能丢掉

- **Blender的BlendShape名字不能超过64字，否则名称会被截取**

- 对于bone,他们会以`Transform`的方式呈现；但在模型（和动画文件）中，他们只会以`Scene`中这些**transform的完整路径的hash存储**

- 然后，**Blender的Vertex Group(bone weight group)同样也不能有64+长名字**

- 对于vertex color，blender的`vertex_colors`layer在4.0已被弃用；不过可以放在**Color Atrributes**

**注：**Blender中对写脚本帮助很大的一个小功能

![image-20240104202236513](/assets/image-20240104202236513.png)

![image-20240105085540376](/assets/image-20240105085540376.png)

### 4. Shaders!

`Texture2D`和其他meta信息导入后，接下来就是做shader了

当然，真正搞NPR的话<u>*值得也需要*再开一个blog系列描述</u>；暂时搓一个基础复现吧

- 手头有的纹理资源如下：

1.  `tex_[...]_C`

   Base **C**olor Map，没什么好说的

![image-20240105081336340](/assets/image-20240105081336340.png)

2. `tex_[...]_S`

   **S**hadowed Color Map（乱猜

   - NPR渲染中常用的阈值Map；为节省性能（和细节质量），引擎也许并不会绘制真正的**Shadow Map**

   - 在很多 NPR Shader中，你会见到这样的逻辑：

   ```glsl
   if (dot(N, L) > threshold) {
   	diffuse = Sample(ColorMap, uv);
   } else {
   	diffuse = Sample(ShadowColorMap, uv);
   }
   ```

即：对NdotL作阈值处理，光线亮（NdotL更大）采用原map，光线暗/无法照明（NdotL更小或为负）采用阴影map

![image-20240105081322556](/assets/image-20240105081322556.png)

3. `tex_[...]_H`

   **H**ightlight Map

   - 注意到`Format`出于某种原因竟然是未压缩的`RGB565`;同时,$R$通道恒为$0$,$B$通道恒为$132$，只有G通道有带意义的信息

   - 这些区域标记对应材质发光部分；虽然没有专门提供Emissive，不过如此直接利用base color也不非一种选择

![image-20240105081327608](/assets/image-20240105081327608.png)

4. Vertex Color

   - [虽然不是]()texture map，但是放这里讲会合适不少

![image-20240105180210479](/assets/image-20240105180210479.png)

- 这里只有RG通道有信息,猜测：

  - $R$通道决定是否接受描边

  - $G$通道决定高光强度


### 5. Shader 实现

1. 阴影 / Diffuse

![image-20240105180740265](/assets/image-20240105180740265.png)

**注：** BSDF应为Diffuse BSDF,截图暂未更新

这里实现的即为上文所述的阈值阴影，不多说了

2. Specular 

直接利用Specular BSDF的输出和前文所提到的weight，mix到输出即可

3. Emissive

用`_H`材质的$G$通道叠加，node如图

![image-20240105183024248](/assets/image-20240105183024248.png)

至此Shader部分介绍完毕，效果如图

![image-20240105183301304](/assets/image-20240105183301304.png)

![image-20240105183407294](/assets/image-20240105183407294.png)

### 6.描边

`ShadeKai`使用将mesh沿法线偏移的tech渲染边界；在 Blender 中，有对应的 Solidify Modifier

- 不过，重现如图效果的话... (PV: [愛して愛して愛して](https://www.bilibili.com/video/BV1cP4y1P7TM/)))

![image-20240105183827963](/assets/image-20240105183827963.png)

- 可见$1$区域带明显描边而$2$区域没有，观察vertex color：

![image-20240105183943570](/assets/image-20240105183943570.png)

![image-20240105184014817](/assets/image-20240105184014817.png)

这和之前对描边做的描述是一致的; $R$​值决定是否描边

貌似Solidify不能根据顶点定制操作；完全复现需要其他路子

---

以上；貌似这篇有点太长了，抱歉

下一次回归PJSK应该还是会讨论NPR Shader的相关实现

同时图形学知识会更多，也许就不发在"*逆向*"专题了吧

那么..

***SEE YOU SPACE COWBOY...***

### References

https://github.com/mos9527/sssekai_blender_io 👈 插件在这

https://github.com/KH40-khoast40/Shadekai

https://github.com/KhronosGroup/glTF-Blender-IO

https://github.com/theturboturnip/yk_gmd_io

https://github.com/SutandoTsukai181/yakuza-gmt-blender

https://github.com/UuuNyaa/blender_mmd_tools