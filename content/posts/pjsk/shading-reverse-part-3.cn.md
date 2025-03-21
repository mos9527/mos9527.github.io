---
author: mos9527
lastmod: 2025-03-20T16:24:19.886000+08:00
title: PSJK Blender卡通渲染管线重现【3】- SDF 面部渲染实现
tags: ["逆向","Unity","PJSK","Project SEKAI","Blender","CG","3D","NPR","Python"]
categories: ["PJSK", "逆向", "合集", "CG"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static
typora-copy-images-to: ../../../static/image-shading-reverse
---

# Preface

在v2面部模型更新后游戏终于引入了SDF面部阴影；实现上和市面上已有的大部分方案大同小异

本文分享自己在Blender中**正确**实现该效果的一种思路

**注意：** 由于个人水平有限错误难免，**强烈推荐**阅读以下文本作为预备知识：

- [二次元角色卡通渲染—面部篇 by MIZI](https://zhuanlan.zhihu.com/p/411188212)
- [卡通渲染——360度脸部SDF光照方案 by Yu-ki016](https://zhuanlan.zhihu.com/p/670837192)
- [Signed Distance Field by 欧克欧克](https://zhuanlan.zhihu.com/p/337944099)

## 1. SDF 概述

SDF面部阴影在解决朴素法线/N dot L阈值阴影在极端光照角度表现上的不足以外提供了表现上相当大的自由

具体实现上也相当简洁，以下是HIFI Rush中的应用例

![image-20250118195404350](/image-shading-reverse/image-20250118195404350.png)

图源：[Tango Gameworksのチャレンジが詰まったカートゥーン調リズムアクションゲーム『Hi-Fi RUSH（ハイファイラッシュ）』（1）キャラクター・モー*ション・エフェクト編](https://cgworld.jp/article/202306-hifirush01.html)

美术在（凭不同光照角度于原几何上的）渲染输出为蓝本，手绘出基于**角度**的阈值图后经*某种合成算法*后生成一张以亮度映射到光照角度的SDF图像

- 着色上，SDF可以将**点/Fragment**渲染的问题转换为**物体/Object**整体渲染的问题 

​	最后的着色与且仅与**光照角度**有关（敲黑板），某种程度上来说等效于对法线一致的*平面*作阴影

- 合成上，有 Valve 在 [SIGGRAPH 2007 上做的技术分享](https://steamcdn-a.akamaihd.net/apps/valve/2007/SIGGRAPH2007_AlphaTestedMagnification.pdf) 和相当多教程会介绍的[8-points Signed Sequential Euclidean Distance Transform/8SSEDT](https://github.com/Lisapple/8SSEDT)

  在之后的文章中将会对这两者进行更深入的实现和剖析

当然，本篇还是以复现游戏效果为主

## 2. 游戏内实现

抽出游戏所用的SDF贴图如下

![faceSdf](/image-shading-reverse/faceSdf.png)

在 Photoshop 中查看阈值，很显然是上文所介绍过的类似模型

  <video autoplay style="width:100%" controls src="https://github.com/user-attachments/assets/c653ee88-0d85-4724-bdad-1c62f1742d7d
"/>

回忆我们所需的**阈值**对应**光照角度**；接下来马上就会用到（再次敲黑板）

## 3. Blender 实现

### 光照角度

从角度出发，考虑单个光向量$l$ —— 如图（来自 RTR4 *...当然这里和BRDF并没有直接联系*



![image-20250118194315626](/image-shading-reverse/image-20250118194315626.png)

由于对称性很显然只凭$ \mathbf{\hat{n}} \cdot \mathbf{\hat{l}}$ 我们只能获得 $\theta \in [0,90\degree]$的光照角（$<0$ 部分背光故省去）

但引入切向量可以解决该问题！如此将允许我们将光向量放在**球面坐标**中观察

只看法向量$n$和切向量$t$，不考虑视角$v$

- 显然有$cos\theta_i = n \cdot l, cos\phi_i = t \cdot l$

- 回顾之前$\theta$范围，可以注意到$\phi \leq 180\degree$ 对应$b = n \times t$一侧半空间，反之亦然

- 如次，结合对称性，根据$\phi$就可以轻松得到$[-90\degree,90\degree]$的光照角

- 公式如下，记$\omega$为最后光照角度
  $$
  \omega = \theta_i * sgn(sin \phi_i)
  $$

### 切空间角度

> 注：其实这块完全不用写 = = $\theta$角和$N$的关系理应很显然为$cos\theta= \mathbf{L \cdot N}$


接下来介绍使用切空间/$\mathbf{TBN}$表示的方法

首先由上文已知我们已经有了正交基$\mathbf{t,b,n}$($b$为$t,n$叉积），$\mathbf{TBN}$矩阵如下
$$
\mathbf{TBN} =
\begin{bmatrix}
T_0 & B_0 & N_0 \newline
T_1 & B_1 & N_1 \newline
T_2 & B_2 & N_2
\end{bmatrix}.
$$
很显然，$\mathbf{TBN}$是个**正交矩阵**，意味着**它的转置等于它的逆**
$$
\mathbf{TBN}^\top = \mathbf{TBN}^{-1} = 
\begin{bmatrix}
T_0 & B_0 & N_0 \newline
T_1 & B_1 & N_1 \newline
T_2 & B_2 & N_2
\end{bmatrix}.
$$


接下来做线形空间变换将光照向量$\mathbf{L}$放到切空间中有：
$$
\mathbf{L} = [x_0,y_0,z_0]^T = a\mathbf{T}+b\mathbf{B}+c\mathbf{N}
$$

$$
\begin{bmatrix}
a \newline
b \newline
c
\end{bmatrix}
= \mathbf{TBN}^{-1}
\begin{bmatrix}
x_0 \newline
y_0 \newline
z_0
\end{bmatrix}
= \mathbf{TBN}^\top
\begin{bmatrix}
x_0 \newline
y_0 \newline
z_0
\end{bmatrix}
$$

$$
\left\lbrace\begin{array}{clcr}
a = T_0 x_0 + T_1 y_0 + T_2 z_0 = \mathbf{L \cdot T} \newline
b = B_0 x_0 + B_1 y_0 + B_2 z_0 = \mathbf{L \cdot B} \newline
c = N_0 x_0 + N_1 y_0 + N_2 z_0 = \mathbf{L \cdot N}
\end{array}
\right.
$$

$L$在$\mathbf{t}$所在平面投影有

$$
\mathbf{L_bn} = b\mathbf{B}+c\mathbf{N} = \mathbf{L} - a\mathbf{T}
$$

求$\theta$即为

$$
cos\theta=\mathbf{L_{bn} \cdot N} = \mathbf{L \cdot N}
$$

同理易得

$$
cos\phi = \mathbf{L \cdot T}
$$


### 切空间构造

相当多 Blender 中实现 SDF 的教程都作了“着色面法向量就是+Y轴”的假设，如设置 Driver 直接取指向光的 Euler Z作方位角

问题显而易见；接下来介绍一种**不作准确度妥协**以在 Blender 实现该trick的一种方式

#### Driver

考虑动画会在骨架上进行，这里考虑直接在骨骼上取法向量

游戏角色模型中一贯都会有一个`Head`骨骼，后面可以用到；方便期间，后文将使用简化模型进行部分演示

和之前取边缘高光的思路相似，直接取用**World Space**的Euler XYZ计算骨骼对应的法向量

![image-20250118204345581](/image-shading-reverse/image-20250118204345581.png)

 **World Space**的好处良多，其一就是该值的演算包含动画（FCurve），Constraint和Parenting - 即使对骨骼也是如此

效果如下：

<video autoplay style="width:100%" controls src="https://github.com/user-attachments/assets/1c159adf-81df-41a6-9274-2374ad7260d2
"/>

#### 构造正交基

当然，如此我们还没有切向量$t$；幸运的是从法向量构造正交基并非难事

以下代码块均来自参考论文[Building an Orthonormal Basis from a 3D Unit Vector Without Normalization - Frisvad, Jeppe Revall](https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf)

- 朴素的想法很简单：在离法向量更近的轴上选一个，正交化，做叉乘即得$n,t,b$正交基底

  ```c++
  void naive ( const Vec3f & n , Vec3f & b1 , Vec3f & b2 )
  {
    // If n is near the x-axis , use the y- axis . Otherwise use the x- axis .
    if(n.x > 0.9 f ) b1 = Vec3f (0.0f , 1.0f , 0.0f );
    else b1 = Vec3f (1.0f , 0.0f , 0.0f );
    b1 -= n* dot (b1 , n ); // Make b1 orthogonal to n
    b1 *= rsqrt ( dot (b1 , b1 )); // Normalize b1
    b2 = cross (n , b1 ); // Construct b2 using a cross product
  }
  ```

​	这种做法直觉地符合我们对$t$向量的要求：可以决定光向量在$b = n \times t$哪一侧

- 而论文中给出的构造方法则相当优美

  ```c++
  void frisvad ( const Vec3f & n , Vec3f & b1 , Vec3f & b2 )
  {
  if(n.z < -0.9999999 f) // Handle the singularity
  {
    b1 = Vec3f ( 0.0f , -1.0f , 0.0f );
    b2 = Vec3f ( -1.0f , 0.0f , 0.0f );
    return ;
    }
    const float a = 1.0 f /(1.0 f + n.z );
    const float b = -n.x*n .y*a ;
    b1 = Vec3f (1.0 f - n .x*n. x*a , b , -n .x );
    b2 = Vec3f (b , 1.0 f - n .y*n. y*a , -n .y );
  }
  ```

  可见我们**不需要叉乘，不需要归一化**也能得到两个额外基底

  篇幅（和水平）有限这里不多做介绍，有兴趣还请参考原文理解

  接下来在Blender中使用Shader Node实现就很轻松了（注：此处忽略了法线处于z轴分界点情形）

  ![image-20250118210701817](/image-shading-reverse/image-20250118210701817.png)

#### 整理实现

利用上文的$cos \phi, cos \theta$，实现轻而易举

<video autoplay style="width:100%" controls src="https://github.com/user-attachments/assets/dd0bb533-a46b-469f-a3d8-cf945faf051d
"/>
做好背光特判后即完成SDF阈值在Shader Node中的计算^^

### 素材处理

- 其一，由于游戏内素材只考虑水平角的光照方案，故计算$\theta$角时有必要丢掉$z$轴计算；否则该角度上带来的角度差只会带来错误
  - 带垂直角的方案请参见文首链接，暂不介绍

- 其二，贴图只包含水平角在$[0,90\degree]$的信息；意味着超过该范围（于$[-90\degree,0]$）需要借用对称性反转
  - 由此脸模也需要对称；当然，制作两张SDF贴图就不会有这样的限制


Node如下

![image-20250119111441108](/image-shading-reverse/image-20250119111441108.png)

用于反转的Node如下

![image-20250119111946143](/image-shading-reverse/image-20250119111946143.png)

反转即简单的$u = 1 - u$

### 最终效果
在动画中实践效果如下;可见阈值光照在骨骼动作上也有正确变化

<video autoplay style="width:100%" controls src="https://github.com/user-attachments/assets/f4fce0cf-2611-438d-96c3-529d69210c74"/>

## References

Real Time Rendering 4th Edition

https://zhuanlan.zhihu.com/p/670837192

https://zhuanlan.zhihu.com/p/411188212

