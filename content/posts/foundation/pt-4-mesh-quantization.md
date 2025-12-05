---
author: mos9527
lastmod: 2025-12-05T17:53:57.254786
title: Foundation 施工笔记 【4】- 网格数据量化
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---


## Preface

对网格数据而言，顶点数可以很多很多：如果存储每个顶点的开销能够减少，显然对GPU显存和磁盘存储压力而言是非常好的事情。以下介绍目前Editor内存在的一些量化操作。下面为完整精度的顶点struct，其足矣表示glTF标准中的任何非蒙皮网格。

```c++
struct FVertex
{
    float3 position;
    float3 normal;
    float3 tangent;
    float bitangentSign;
    float2 uv;
};
static_assert(sizeof(FVertex) == 48);
```

### FP16 量化

现代硬件基本都有硬件级别的 FP16（半精度）支持。业界包括 Unity 在内在存储顶点数据时也提供了[烘焙部分通道（vertices/normals/...）为该精度](https://docs.unity3d.com/6000.2/Documentation/Manual/types-of-mesh-data-compression.html#vertex-compression)的选项，实现上这也比较简单。

CPU，或者C++侧对有限精度浮点数的支持一直以来并不友好 - 毕竟硬件级相关指令在AVX512才有。在[C++23](https://en.cppreference.com/w/cpp/types/floating-point.html)中才有了语言级别的`float16/float64/float128`甚至是`bfloat16`的支持——鉴于某M姓编译器对23标准的支持仍是draft，暂时不考虑升级标准支持这些特性。

当然，手动进行FP32-FP16转化是可能的。全精度和半精度的二进制结构都在[IEEE 754]中定义了几十年，标准上不存在问题。以下转换实现再次来自[`meshoptimizer`](https://github.com/zeux/meshoptimizer/blob/master/src/quantization.cpp)：

```c++
union FloatBits
{
    float f;
    unsigned int ui;
};
unsigned short quantizeFP16(float v)
{
    FloatBits u = {v};
    unsigned int ui = u.ui;
    int s = (ui >> 16) & 0x8000;
    int em = ui & 0x7fffffff;
    // bias exponent and round to nearest; 112 is relative exponent bias (127-15)
    int h = (em - (112 << 23) + (1 << 12)) >> 13;
    // underflow: flush to zero; 113 encodes exponent -14
    h = (em < (113 << 23)) ? 0 : h;
    // overflow: infinity; 143 encodes exponent 16
    h = (em >= (143 << 23)) ? 0x7c00 : h;
    // NaN; note that we convert all types of NaN to qNaN
    h = (em > (255 << 23)) ? 0x7e00 : h;
    return static_cast<unsigned short>(s | h);
}
float dequantizeFP16(unsigned short h)
{
    const unsigned int s = static_cast<unsigned int>(h & 0x8000) << 16;
    const int em = h & 0x7fff;
    // bias exponent and pad mantissa with 0; 112 is relative exponent bias (127-15)
    int r = (em + (112 << 10)) << 13;
    // denormal: flush to zero
    r = (em < (1 << 10)) ? 0 : r;
    // infinity/NaN; note that we preserve NaN payload as a byproduct of unifying inf/nan cases
    // 112 is an exponent bias fixup; since we already applied it once, applying it twice converts 31 to 255
    r += (em >= (31 << 10)) ? (112 << 23) : 0;
        FloatBits u;
        u.ui = s | r;
        return u.f;
    }
```

应用上，我们只对`float3 position`做FP16存储 - 理由会在后面给出。

```c++
uint16_t position[4]; // quantized FP16 [xyz] padding [w]
...
result.position[0] = quantizeFP16(vertex.position[0]);
result.position[1] = quantizeFP16(vertex.position[1]);
result.position[2] = quantizeFP16(vertex.position[2]);
```

**注意：** 这里最后存在2字节的填充$w$。目的在于让最后整个vertex的大小为$4$的倍数——结构体中最大field的大小，且同时`meshoptmizer`也会利用$4$对齐的属性提供一些操作的SIMD加速。

### 定点量化

若值域已知，用定点方式表示浮点数也是很常见的有损压缩技巧——Unity对蒙皮权重就有这样的操作。特别地，对于值域在$[0,1]$ (UNROM) ,$[-1,1]$ (SNORM)的数而言，我们可以直接省去值域本身的存储：毕竟就在名字("Unsigned Normal","Signed Normal") 里嘛。

这两者实现上非常直觉：$\text{数值}*\text{值域区间}$ 即得到量化后数值，反过来也很简单。实现（及SNORM以UNORM无符号存储变式 `[de]quantizeSnormShifted`）如下，$N$为量化bit数

```c++
inline uint32_t quantizeUnorm(float v, int32_t N)
{
    const auto scale = static_cast<float>((1 << N) - 1);
    v = (v >= 0) ? v : 0;
    v = (v <= 1) ? v : 1;
    return static_cast<int>(v * scale + 0.5f);
}
inline float dequantizeUnorm(int32_t q, int32_t Nbits) { return q / static_cast<float>((1 << Nbits) - 1);}
inline int32_t quantizeSnorm(float v, int32_t N)
{
    const auto scale = static_cast<float>((1 << (N - 1)) - 1);
    float round = (v >= 0 ? 0.5f : -0.5f);
    v = (v >= -1) ? v : -1;
    v = (v <= +1) ? v : +1;
    return static_cast<int>(v * scale + round);
}
inline float dequantizeSnorm(int32_t q, int32_t Nbits) { return q / static_cast<float>((1 << (Nbits - 1)) - 1); }
inline uint32_t quantizeSnormShifted(float v, int32_t Nbits)
{
    return quantizeSnorm(v, Nbits) + (1 << (Nbits - 1));
}
inline float dequantizeSnormShifted(uint32_t q, int32_t Nbits)
{
    return dequantizeSnorm(q - (1 << (Nbits - 1)), Nbits);
}
```

应用上，UV坐标就属于UNORM的范畴。现在先不谈法向量(normal)，切向量(tagent)：他们（规范化为单位向量时）确实可以直接用SNORM表达，你也可以这么做！不过在此之前， 请阅读下一部分。

```c++
uint16_t uv[2]; // quantized UNORM16
...
result.uv[0] = quantizeUnorm(vertex.uv[0], 16);
result.uv[1] = quantizeUnorm(vertex.uv[1], 16);
```

### 法向量 (Normal) 及切向量 (Tangent) 压缩

高效法向量存储是个热门话题：延后（Deferred）渲染流行以来：因有在GBuffer存储法向量的必要，且内存有限，能够高效存储法线/切线/TBN(Tangent-Bitangent-Normal)矩阵做bump map是很值得追求的一个目标。

正如前文所述，对于单位的normal，tagent：定点量化也是一种选择。其他更为高效的选项也存在，参考：

- [Compact Normal Storage for small G-Buffers - Aras Pranckevičius](https://aras-p.info/texts/CompactNormalStorage.html)

这里只介绍其中部分的几个方案。

#### 单位向量投影

方案之一的idea来自：规范化的**三维**单位向量可以投影到某种**二维**坐标表示。更熟悉地，问题可以描述为：在**球坐标系**，规范长度为$1$时，就能用仅极角，方位角表示单位长度的所有向量。

不过注意，三角函数的执行是可能昂贵的——而且，往往在shader code中可以**完全**避免。这里可阅读iq大佬的这三篇博文：

- [Avoiding trigonometry I](https://iquilezles.org/articles/noacos)
- [Avoiding trigonometry II](https://iquilezles.org/articles/sincos)
- [Avoiding trigonometry III](https://iquilezles.org/articles/noatan)

参考 [Survey of Efficient Representations for Independent Unit Vectors - 2. 3D Unit Vector Representations](https://jcgt.org/published/0003/02/01/)，我们刚刚描述的正为其中**spherical**方案。除了计算需要三角函数，这样的朴素算法的误差也是很不理想的。

![image-20251205084118183](/image-foundation/image-20251205084118183.png)

文中的**oct**方案，也就是**八面体**，则被认为是"Best overall method"。以下为原论文中的参考实现：

```glsl
// Returns ±1
vec2 signNotZero(vec2 v) {
	return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}
// Assume normalized input. Output is on [-1, 1] for each component.
vec2 float32x3_to_oct(in vec3 v) {
    // Project the sphere onto the octahedron, and then onto the xy plane
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
}
vec3 oct_to_float32x3(vec2 e) {
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    return normalize(v);
}
```

直接应用于法向量，切向量的确是一种方案；最后有相关shadertoy演示有限bit数量化后压缩效果：[传送门](https://www.shadertoy.com/view/Mtfyzl)。可见即使在较低量化空间下，视觉效果也是可观的。

#### 四元数存储

不过，如果要做normal/bump mapping的话，完整的TBN/切空间基底是少不了的。

注意切空间和法线贴图的关系并没有绝对的标准（参见 [Tangent Space Normal Maps - Blender Wiki](https://archive.blender.org/wiki/2015/index.php/Dev:Shading/Tangent_Space_Normal_Maps/)），一个法向量*可以*对应无穷多的切向量。不过包括 glTF，Blender，Unity，UE在内基本都在用的是 [MikkTSpace](http://www.mikktspace.com/)，大多数法线贴图也是于这里提供的切空间烘焙——同时因为关系非1:1,一般地，拥有法线贴图的网格也的顶点会离线存储这样的切向量以确保正确性。

##### Bitangent 符号

一个常见trick即为离线存储时，只存储$\mathbf{n}, \mathbf{t}$和一个符号量：应为共面，$\mathbf{b}$即为$\mathbf{n} \mathbf{t}$叉乘，并做这样的翻转。glTF也是这么做的：

| Name         | Accessor Type(s) | Component Type(s)                                            | Description                                                  |
| :----------- | :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `POSITION`   | VEC3             | *float*                                                      | Unitless XYZ vertex positions                                |
| `NORMAL`     | VEC3             | *float*                                                      | Normalized XYZ vertex normals                                |
| `TANGENT`    | VEC4             | *float*                                                      | XYZW vertex tangents where the XYZ portion is normalized, **and the W component is a sign value (-1 or +1) indicating handedness of the tangent basis** |

假设完整精度直接存储，我们的tangent frame这样需要$4*(3+4)=28$字节。当然上述的投影+量化技巧也值得应用，不过原理上并没有新东西，这里不阐述。

##### 四元数压缩

其实直接间看矩阵形式：定义$\mathbf{TBN}$矩阵是个3x3的正交阵（一般如此，例外情况则需要正交化处理）——相当于旋转矩阵，这是可以直接用四元数表示的。在`glm`中可以直接利用`mat3_cast`和`quat_cast`相互转换。

该操作在 [Bindless Texturing for Deferred Rendering and Decals - MJP](https://therealmjp.github.io/posts/bindless-texturing-for-deferred-rendering-and-decals), [压缩tangent frame - KlayGE](http://www.klayge.org/2012/09/21/%e5%8e%8b%e7%bc%a9tangent-frame/)中都有提及。不过注意，直接存储四元数$(xyzw)$也是很浪费的（$4*4=16$字节！)。不过值得注意的是，我们的四元数也是**单位**四元数，即$x^2+y^2+z^2+w^2=1$。两篇文章都提到了如何利用该性质$4$字节完成四元数存储的任务，接下来将介绍**四元数压缩**的一种高精度实现。

KlayGE直接利用$A2BGR10$格式存储了$x,y,z$部分，最后$A$记录$w = \pm\sqrt{1-x^2-y^2-z^2}$的符号。不过，精度上的优化空间是可循的：MJP文章引用的[The BitSquid low level animation system](https://bitsquid.blogspot.com/2009/11/bitsquid-low-level-animation-system.html)用$A$去记录**四个分量中绝对值最大的分量【位置】**。原因引用原作者：

> ...You could use arithmetic encoding to store x, y and z using 10.67 bits per component for the range -1, 1 and this would give you slightly better precision for these values.
The problem comes when you want to reconstruct w using sqrt(1-x^2-y^2-z^2) because that function is numerically unstable for small w.
Have a look at this graph:
https://www.desmos.com/calculator/nfdbj0law4
Below w=0.5 the error starts to become bigger than the error (0.001) in the input values and as we get closer to zero the error becomes a lot bigger (0.03).

文中graph如下，对应$\sqrt{w^{2}+0.001}-w$：可见$w<0.5$之后的计算误差增长非常快。去重构*绝对值最大值*而非*可能*最小的$w$即可以避免这种数值误差。

![image-20251205163112243](/image-foundation/image-20251205163112243.png)

另外，一个实现细节在于：$q=(x,y,z,w)$与$-q=(-x,-y,-z,-w)$表示的是同一个变换。我们可以借此简化之前的$w$意义为**四个分量中最大的分量【位置】**，包括符号：这建立在若绝对值最大分量为负，则全体取反的基础上。

结合以上信息，以下为**四元数压缩**的具体实现——参考 [TheRealMJP/DeferredTexturing](https://github.com/TheRealMJP/DeferredTexturing/blob/master/SampleFramework12/v1.01/Shaders/Quaternion.hlsl)。稍作修改即可适用slang shader。

```c++
// Packs quaternion to [UNORM XYZ, 2-bit max index]
inline float4 packQuaternionXYZPositionBit(quat const& q)
{
    float4 Q(q.x, q.y, q.z, q.w);
    float4 absQ(abs(Q.x), abs(Q.y), abs(Q.z), abs(Q.w));
    float absMax = max(max(absQ.x, absQ.y), max(absQ.z, absQ.w));
    uint maxIndex = 0;
    if (absQ[0] == absMax)
        maxIndex = 0;
    if (absQ[1] == absMax)
        maxIndex = 1;
    if (absQ[2] == absMax)
        maxIndex = 2;
    if (absQ[3] == absMax)
        maxIndex = 3;
    if (Q[maxIndex] < 0) // ensure positive
        Q = -Q;
    float3 packed;
    if (maxIndex == 0)
        packed = Q.yzw();
    if (maxIndex == 1)
        packed = Q.xzw();
    if (maxIndex == 2)
        packed = Q.xyw();
    else /* maxIndex == 3 */
        packed = Q.xyz();
    packed *= sqrt(2.0f); // e.g. (1,0,0,1), max bounds
    packed = packed * 0.5f + 0.5f; // [-1,1] -> [0,1]
    return float4(packed, maxIndex / 3.0f);
}

// Unpacks quaternion from [UNORM XYZ, 2-bit max index] to quat
inline quat unpackQuaternionXYZPositionBit(float4 const& packed)
{
    uint maxIndex = packed.w * 3.0f;
    float3 p = packed.xyz() * 2.0f - 1.0f; // [0,1] -> [-1,1]
    p /= sqrt(2.0f);
    float4 Q;
    float maxValue = sqrt(max(.0f, 1 - p.x * p.x - p.y * p.y - p.z * p.z));
    if (maxIndex == 0)
        Q = float4(maxValue, p.xyz);
    else if (maxIndex == 1)
        Q = float4(p.x, maxValue, p.yz);
    else if (maxIndex == 2)
        Q = float4(p.xy, maxValue, p.z);
    else /* maxIndex == 3 */
        Q = float4(p.xyz, maxValue);
    return quat(Q.x, Q.y, Q.z, Q.w);
}
```

#### 法向量 + 切向量旋转量

![image-20251205173111991](/image-foundation/image-20251205173111991.png)

idea来自[RENDERING THE HELLSCAPE OF DOOM ETERNAL - SIGGRAPH 2020](https://advances.realtimerendering.com/s2020/RenderingDoomEternal.pdf)。思想上和之前的四元数方案有些异曲同工之妙：$\mathbf{n} \cdot \mathbf{t} = 0$是一定的。那不妨在法线所在平面内构造**运行时**某种正交基，我们的$\mathbf{t}$一定和他们共面：用这两个向量基底构建他就好。

法向量**在线**构造正交基的方法很多，参考：

- [Followup: Normal Mapping Without Precomputed Tangents](http://www.thetenthplanet.de/archives/1180)
- [Surface Gradient Bump Mapping Framework Overview](https://www.jeremyong.com/graphics/2023/12/16/surface-gradient-bump-mapping/) 及 [Surface Gradient–Based Bump Mapping Framework](https://jcgt.org/published/0009/03/04/)
- [Tangent-basis workflow for getting 100% correct normal-mapping #1252 - KhronosGroup/glTF](https://github.com/KhronosGroup/glTF/issues/1252)

最简单且快速的一种来自[Building an Orthonormal Basis from a 3D Unit Vector Without Normalization - Frisvad, 2012](https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf)，他在 [UE](https://github.com/EpicGames/UnrealEngine/blob/684b4c133ed87e8050d1fdaa287242f0fe2c1153/Engine/Source/Runtime/MeshUtilitiesCommon/Public/MeshUtilitiesCommon.h#L102) 中也能见到。以下为简化实现：
```c++
inline void BuildOrthonormalBasis(const float3 n, float3& b1, float3& b2)
{
    if (n.z < -0.9999999)
    {
        b1 = float3(0.0, -1.0, 0.0);
        b2 = float3(-1.0, 0.0, 0.0);
        return;
    }
    float a = 1.0 / (1.0 + n.z);
    float b = -n.x * n.y * a;
    b1 = float3(1.0 - n.x * n.x * a, b, -n.x);
    b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
}
```

不过，直接利用这些基底直接做bump map是不正确的——理由已在前文给出：法线贴图取决于烘焙到的tangent space，若要保持他们一致，则烘焙时和在线的结果也许一样：这很难做。但是知道**多**不正确，或者用他们如何**重构**$\mathbf{t}, \mathbf{b}$，则是很好做的一件事：$\mathbf{t}$投影即可。

```c++
float3 b1, b2;
BuildOrthonormalBasis(normal, b1, b2);
// To angle
float cosAngle = dot(tangent, b1), sinAngle = dot(tangent, b2);
float angle = atan2(sinAngle, cosAngle) / pi<float>();
// From angle
tangent = cos(angle) * b1 + sin(angle) * b2;    
```

TBD