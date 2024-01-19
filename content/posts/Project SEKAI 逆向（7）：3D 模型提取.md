---
author: mos9527
title: Project SEKAI 逆向（7）：3D 模型提取
tags: ["逆向","unity","pjsk","api","project sekai","miku","unity","3d","cg","blender"]
categories: ["Project SEKAI 逆向", "逆向","Tech Art"]
ShowToc: true
TocOpen: true
typora-root-url: ./..\..\static

---

# Project SEKAI 逆向（7）：3D 模型

### 0. 前言

见（4），利用AssetStudio将sekai的模型导入到DCC软件（i.e. blender）并不成功

加之，任何关于sekai模型的其他信息（i.e. 物理效果bounding box）也会全部被丢失

嗯，看起来造轮子的motivation很充足！

那就开始吧？

**Code：** https://github.com/mos9527/sssekai_blender_io

**注：** 仅测试于 Blender 4.0

### 1. 文件结构

目前发现的 3D 模型基本都在 `[ab cache]/live_pv/model/` 下；看看miku的原味模：

![image-20240105080707360](/assets/image-20240105080707360.png)

![image-20240105080841622](/assets/image-20240105080841622.png)

初步观察：

- (1) Body 即目标模型；当然，作为skinned mesh，而且带有blend shapes，处理细节会很多；后面继续讲
- (2) 处的 `MonoBehavior` 就其名字猜测是碰撞盒
- (3) 处的几个 `Texture2D` 则作为texture map

### 2. 模型收集

利用`sssekai`取得数据的流程在（5）中已有描述，这里不再多说

也许各家用Unity渲染可能会走不同的封包途径，不过这里就pjsk的方法进行分析

(术语比较多，还请自行百度

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

总结一下

- (1) 所以对于static mesh,搜集对应`GameObject`即可

- (2) 对于skinned mesh，同时也需要构造bone hierarchy（就是个单根有向无环图啦），并且整理vertex权重；

  则需要收集的，反而只是bone的transform而已；transform有子/父节点信息，也有拥有transform的`GameObject`的ref

- (3) 的数据，在(1)(2)中都会有

具体实现，请看文首代码链接
### 3. 模型导入

当然，这里就不考虑将模型转化为中间格式了（i.e. FBX,GLTF）

利用blender python，可以直接给这些素材写个importer

实现细节非常，非常多😓这里只提几个坑点

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
    
    不知道为什么还要取共轭...

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

- blender的BlendShape名字不能超过64字，否则被截取

- 对于bone,他们会以`Transform`的方式呈现；但在模型中，他们只会以`Scene`中这些**transform的完整路径的hash存储**

- 然后，blender的Vertex Group(bone weight group)同样也不能有64+长名字

- 对于vertex color，blender的`vertex_colors`layer在4.0已被弃用；不过可以放在Color Atrributes

..同样；具体实现，请看文首代码链接

对了，写blender python脚本，你应该会想开这个

![image-20240104202236513](/assets/image-20240104202236513.png)

![image-20240105085540376](/assets/image-20240105085540376.png)

脚本中就可以直接写这个路径了

### 4. TextureMaps to what?

`Texture2D`和其他meta信息导入后，接下来就是做shader了

参考 https://github.com/KH40-khoast40/Shadekai，检查手头有的texture map

1.  `tex_[...]_C`

   Base **C**olor Map，没什么好说的

![image-20240105081336340](/assets/image-20240105081336340.png)

2. `tex_[...]_S`

   **S**hadowed Color Map（乱猜

   pjsk的确使用shadow mapping；当然，此shadow map非彼[shadow map](https://en.wikipedia.org/wiki/Shadow_mapping)

   这里用于阴影部分像素的替代base color map，选择性地调整区域颜色

   具体细节未知，不过貌似不是简单调节明度的结果；也许是为什么会多一个map的原因

![image-20240105081322556](/assets/image-20240105081322556.png)

3. `tex_[...]_H`

   **H**ightlight Map （同样瞎猜，不过应该差不多

   注意到`Format`出于某种原因竟然是未压缩的`RGB565`;同时,$R$通道恒为$0$,$B$通道恒为$132$，只有G通道有带意义的信息

   这些区域标记对应材质发光部分；虽然没有专门提供Emissive，不过如此直接利用base color也不非一种选择

![image-20240105081327608](/assets/image-20240105081327608.png)

4. Vertex Color

   虽然不是texture map，但是放这里讲会合适不少（

   很少见有人会用到这个信息；不过，pjsk的模型都是有Vertex Color的

![image-20240105180210479](/assets/image-20240105180210479.png)

这里只有RG通道有信息；如果`ShadeKai`正确，那么：

- $R$通道决定是否接受描边
- $G$通道决定高光强度

### 5. Shader 实现

用blender的shader node可以很轻松地实现前文所述的NPR效果

1. 阴影

![image-20240105180740265](/assets/image-20240105180740265.png)

首先，加入SpecularBSDF pass + color ramp，二值化后可得到hard shadow

ShadeKai中还会利用$N \cdot -L$决定顶点-光源是否在同一半空间中来调整阴影。不过blender中使用light vector比较麻烦；暂且用BSDF pass得到的shadow map妥协

得到shadow数据，只需mix basecolor map即可

2. 高光

直接利用Specular BSDF的输出和前文所提到的weight，mix到输出即可

3. 自发光

用`_H`材质的$G$通道叠加，node如图

![image-20240105183024248](/assets/image-20240105183024248.png)

至此Shader部分介绍完毕，效果如图

![image-20240105183301304](/assets/image-20240105183301304.png)

![image-20240105183407294](/assets/image-20240105183407294.png)

### 6.描边 (NOTES ONLY)

**注：** 暂时没找到blender里选择性描边的思路（悲）；以下为日后在[Foundation](https://github.com/mos9527/Foundation)中复现所做笔记

`ShadeKai`使用将mesh沿法线偏移的tech渲染边界

目标即达成图示效果 (PV: [愛して愛して愛して](https://www.bilibili.com/video/BV1cP4y1P7TM/)))

![image-20240105183827963](/assets/image-20240105183827963.png)

可见$1$区域带明显描边而$2$区域没有，观察vertex color：

![image-20240105183943570](/assets/image-20240105183943570.png)

![image-20240105184014817](/assets/image-20240105184014817.png)

这和之前对描边做的描述是一致的; $R$值决定是否描边

---

以上；貌似这篇有点太长了（

还有描边效果可以通过blender freestyle实现，不过选择精细不到edge

嗯，看看以后能不能解决