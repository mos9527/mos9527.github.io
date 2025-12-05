---
author: mos9527
lastmod: 2025-12-05T13:17:53.077899
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

### 法向量/TBN 量化

高效法向量存储是个热门话题：延后（Deferred）渲染流行以来：因有在GBuffer存储法向量的必要，且内存有限，能够高效存储法线/切线/TBN(Tangent-Bitangent-Normal)矩阵做bump map是很值得追求的一个目标。

正如前文所述，对于单位的normal，tagent：定点量化也是一种选择。其他更为高效的选项也存在，参考：

- [Compact Normal Storage for small G-Buffers - Aras Pranckevičius](https://aras-p.info/texts/CompactNormalStorage.html)

这里只介绍其中部分的几个方案。

#### 单位向量投影

规范化的**三维**单位向量可以投影到某种**二维**坐标表示。更熟悉地，问题可以描述为：在**球坐标系**，规范长度为$1$时，就能用仅极角，方位角表示单位长度的所有向量。

不过注意，三角函数的执行是可能昂贵的——而且，往往在shader code中可以**完全**避免：iq大佬的这三篇博文非常值得拜读：

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

对单位法向量，我们选择了利用10bit存储投影后SNORM值。该部分实现如下：注意`shifted`，我们存储时并不处理符号。

```c++
float2 oct = PackUnitOctahedralSnorm(normal);
uint32_t nX = quantizeSnormShifted(oct.x, 10), nY = quantizeSnormShifted(oct.y, 10);
```

### 切向量表示

如果要做normal/bump mapping的话，完整的TBN基底是少不了的。$\mathbf{t}$的具体编码取决于构建资产的工具本身：参考 [Tangent Space Normal Maps - Blender Wiki](https://archive.blender.org/wiki/2015/index.php/Dev:Shading/Tangent_Space_Normal_Maps/)。幸运的是，当下应用一般采用同一种“标准”（包括glTF，Blender，Unity，UE等等），即 [MikkTSpace](http://www.mikktspace.com/)。

**注：**参考 [Surface Gradient–Based Bump Mapping Framework - Mikkelsen 2020](https://jcgt.org/published/0009/03/04/) ，[Surface Gradient Bump Mapping Framework Overview - jeremyong](https://www.jeremyong.com/graphics/2023/12/16/surface-gradient-bump-mapping/) - 在硬件上**在线**计算也是可能的：接下来将整理该方向实现的一些shader及数学细节。

TBD