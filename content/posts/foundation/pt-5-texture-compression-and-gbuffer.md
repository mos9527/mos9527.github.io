---
author: mos9527
lastmod: 2025-12-09T12:28:51.062891
title: Foundation 施工笔记 【5】- 纹理与延后渲染初步
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

Editor暂时还没有任何关于光照，甚至是texture map的相关实现。本篇文章将介绍纹理方面的*高效*存储及Editor GBuffer的布局，及PBR shading的初步测试。

## 容器格式

不看轮子和各种私有格式的话，常用的主要有[DDS](https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide)和最近的[KTX2](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html)。

### KTX

虽然[工具链很成熟](https://github.com/KhronosGroup/KTX-Software)，在新版[Vulkan Tutorial](https://docs.vulkan.org/tutorial/latest/15_GLTF_KTX2_Migration.html#_understanding_ktx2)中也被绝赞推荐：但由于全平台支持，直接使用的话由于各种编码产生的依赖会多的离谱（`--depth=1` clone大小约莫~1GB！），因此这里没有选择。

### DDS

最后挑的还是DDS，因为容器本身很简单：[`"DXT " + DDS_HEADER + DDS_HEADER_DXT10`](https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header)后即为对应编码下linear tiled数据，可以分成layer/mip直接上传 GPU。至于为什么不应该(Linear Tiling)整个上传（或者，GPU上的纹理到存储方式为什么是硬件相关的），请参考 [Image Copies - Vulkan Documentation](https://docs.vulkan.org/guide/latest/image_copies.html)。

实现上主要是读文档——部分结构直接从[DirectXTex - DDS.h](https://github.com/microsoft/DirectXTex/blob/main/DirectXTex/DDS.h) copy过来了。用DDS有一个好处就是能被第三方工具直接打开，同时，Editor序列化纹理也将如此存储。

## 纹理格式

运行时解码成全精度，原始的`R8G8B8A8UNORM`等格式存储诚然是...一种选择。不过记得之前和[某个Unity游戏](https://mos9527.com/posts/pjsk/archive-20240105/#3-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%85%A5)打交道见过其在资产里直接存了原始`RGB565`(RGB 16 bit)量化格式的纹理做[NPR 效果](https://mos9527.com/posts/pjsk/shading-reverse-part-2/#%E7%AE%A1%E7%BA%BF%E6%9D%90%E8%B4%A8)。但是至于*为什么*要用这种格式，搜了半天也没找到个结果（汗）

纹理压缩，可能除了UI带alpha材质的极端情况外，基本是必做的事情。GPU硬件解码其支持的格式开销是几乎不存在的——参考 [Unity](https://docs.unity3d.com/6000.4/Documentation/Manual/texture-choose-format-by-platform.html) 给出的材质格式推荐：

### DXTc/BCn/ETC

- **桌面端**，支持最广泛的包括各种BCn/Block Compression [n]压缩 - 参见 [Understanding BCn Texture Compression Formats - Nathan Reed](https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats)；同时，[BC1,BC2,BC3 也被微软叫成 DXT1, DXT3, DXT5](https://en.wikipedia.org/wiki/S3_Texture_Compression)，这里提及以防混淆。最后，在近十年支持DX11级别的桌面硬件中，无脑使用`BC6(H),BC7` 基本没问题。

- **移动端**，ETC曾是主流：Unity的[crunch](https://github.com/Unity-Technologies/crunch/tree/unity)就提供第一方DTXc/BCn和ETC的支持。现在的移动硬件一概支持更优秀且**开源**的ASTC：Unity自己也在使用[ARM官方的开源astc-codec](https://github.com/ARM-software/astc-encoder)处理编码。但需要注意的是，**桌面硬件上对astc的支持几乎不存在**。这点也可以自己利用Vulkan Caps Viewer检查：![image-20251206202629076](/image-foundation/image-20251206202629076.png)

### Basis Universal

可见硬件上——上述压缩编码在桌面/移动端的支持是没有交集的。但是纹理压缩做的工作本身会有很多重叠（如 DCT编码等等）—— [basisu](https://github.com/BinomialLLC/basis_universal) 即可高效地**存储一种统一**编码格式，并在运行时*非常快速*地转换到目标平台使用编码上传GPU。

额外的，basisu也自带一层[哈夫曼编码](https://github.com/BinomialLLC/basis_universal/wiki/.basis-file-format-overview) - 这是在所有block处理完之后全局进行的，因此整体压缩率在相同平均bpp下也会比单纯的BCn等编码一般地会更高。

### BC7

最后在GPU和文件本身存储上还是采用了BC7 - Editor内置了来自crunch作者[Richard Geldreich](https://richg42.blogspot.com/) 的 [bc7enc](https://github.com/richgel999/bc7enc) 实现。这样也方便直接读取并直接转码JPG/PNG存储的glTF模型纹理。

和之前网格一样，优化（转码）/烘焙部分是可以离线的。以下是结合mip生成以后产生的DDS在[tacentview](https://github.com/bluescan/tacentview)预览效果：具体实现细节太多，姑且就不贴在这里了。有兴趣可以[点这里](https://github.com/mos9527/Foundation/blob/vulkan/Editor/Texture.cpp)查看。

![image-20251207180343608](/image-foundation/image-20251207180343608.png)

## GBuffer

我们实现的是延后（Deferred）渲染。现在暂不考虑探索移动端（和果子M芯片）的[TBDR （Tile Based Deferred Rendering）](https://developer.apple.com/documentation/Metal/rendering-a-scene-with-deferred-lighting-in-c++)，在用的IMR (Immediate Mode Rendering) 模式还是需要几个Pass的。

Task/Mesh部分在前面已经讲得很详细，这里不再多说。接下来发生的事情，基本在 Pixel/Fragment 和 Compute 里进行。

### GBuffer 布局

![image-20251208162456691](/image-foundation/image-20251208162456691.png)

最终目标是能完整表现glTF的[Metallic-Roughness](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials)模型。除了Base Color/底色及法线贴图外，我们还有两个Metal/Rough参数是必须表现的，图中还有自发光材质。最后，可选的`occlusion`/预烘焙AO在此暂时不考虑。

结合上一篇介绍的一些packing和切空间压缩奇技淫巧，我们的GBuffer可以整理得很简洁，参见下表。所有RT格式皆为`R8G8B8A8Unorm`

| Target | R                 | G                 | B                | A                            |
| ------ | ----------------- | ----------------- | ---------------- | ---------------------------- |
| RT0    | BaseColor R [8]   | BaseColor G [8]   | BaseColor B [8]  | Material ID [7] + TBN手性[1] |
| RT1    | Normal 投影 X [8] | Normal 投影 Y [8] | Tangent 夹角 [8] | Metallic [8]                 |
| RT2    | Emissive R [8]    | Emissive G [8]    | Emissive B [8]   | Roughness [8]                |

- BaseColor, Emissive 直接存储，各占一对RGB通道
- Tangent Frame是完整保留的，且只用3字节+1bit（RT1 RGB + RT0 A[1]）。后面实现PBR时可以用来实现各向异性效果。
- Metallic, Roughness 各占一个通道
- 此外RT0还有存一个Material ID，不过因为目前glTF只有一种材质模型，所以还用不到。

### GBuffer 生成

效果如下，配置好MRT以后：以下为前两个RT的GBuffer表现。鉴于场景内不存在自发光对象则省略RT3。

![image-20251208200753938](/image-foundation/image-20251208200753938.png)
![image-20251208200901053](/image-foundation/image-20251208200901053.png)

## 光照及线性 Workflow

正经实现 PBR 光照开始。[Physically Based Rendering in Filament](https://google.github.io/filament/Filament.md.html) ，PBRT/[Physically Based Rendering:From Theory To Implementation](https://pbr-book.org/) 和手头的 RTR4/[Real-Time Rendering 4th Edition](https://www.realtimerendering.com/) （尤其是第九章）将是我们主要的信息来源。

### BRDF	

参考  [4.1 Standard model](https://google.github.io/filament/Filament.md.html#materialsystem/standardmodel) - 读者请自行完成相关阅读，这里将不在原理方面过多阐述。

[4.6 Standard model summary](https://google.github.io/filament/Filament.md.html#materialsystem/standardmodelsummary) 包括实现GGX Specular和Lambert Diffuse所需的一切Listing。整理 [4.10.1 Anisotropic specular BRDF](https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel/anisotropicspecularbrdf) 内容，以下是我们将要在本demo使用的BRDF中$F$及支持各向异性的$D,V$函数。

```glsl
float D_GGX_Anisotropic(float NoH, const float3 h, const float3 t, const float3 b, float at, float ab) {
    float ToH = dot(t, h);
    float BoH = dot(b, h);
    float a2 = at * ab;
    float3 v = float3(ab * ToH, at * BoH, a2 * NoH);
    float v2 = dot(v, v);
    float w2 = a2 / v2;
    return a2 * w2 * w2 * (1.0 / PI);
}
float3 F_Schlick(float u, float3 f0) {
    return f0 + (float3(1.0) - f0) * pow(1.0 - u, 5.0);
}
float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV,
        float ToL, float BoL, float NoV, float NoL) {
    float lambdaV = NoL * length(float3(at * ToV, ab * BoV, NoV));
    float lambdaL = NoV * length(float3(at * ToL, ab * BoL, NoL));
    float v = 0.5 / (lambdaV + lambdaL);
    return saturate(v);
}
```

**注**：关于$a$ - 注意 [glTF Spec Appendix B.2.3. Microfacet Surfaces](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#microfacet-surfaces) 及   [Fliament 4.8.3 Remapping](https://google.github.io/filament/Filament.md.html#materialsystem/parameterization/remapping) 中也有提到。对roughness做$\alpha = \text{roughness}^2$的mapping是被推荐的。

### PBR 相机

参考来自以下来源：

- [Automatic Exposure Using a Luminance Histogram](https://bruop.github.io/exposure/) 及 [Tonemapping -  Bruno Opsenica](https://bruop.github.io/tonemapping/)
- [Automatic Exposure - Krzysztof Narkowicz](https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/)
- [ACES Filmic Tone Mapping Curve - Krzysztof Narkowicz](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/)
- [8.1 Physically based camera - Filament](https://google.github.io/filament/Filament.md.html#imagingpipeline/physicallybasedcamera)

#### EV100

摘自 [EV与照明条件的关系 - 维基百科](https://zh.wikipedia.org/wiki/%E6%9B%9D%E5%85%89%E5%80%BC#EV%E4%B8%8E%E7%85%A7%E6%98%8E%E6%9D%A1%E4%BB%B6%E7%9A%84%E5%85%B3%E7%B3%BB)：
$$
\mathrm {EV} =\log _{2}{\frac {LS}{K}}
$$

其中$L$为场景**平均辉度**（average luminance/nit），$S$为ISO指数，$K$为校准指数：[一般为$12.5$](https://en.wikipedia.org/wiki/Light_meter#Calibration_constants)。代入ISO100及这个值，我们得到$EV_{100}$的定义：
$$
EV_{100} = log_2{L\frac{100}{12.5}}
$$
EV是一个控制量。对于饱和相机传感器的辐照度$L_{max}$，[Moving Frostbite to Physically based rendering V3](https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf) 及 [8.1 Physically based camera - Filament](https://google.github.io/filament/Filament.md.html#imagingpipeline/physicallybasedcamera) 都给出了以下式子：
$$
L_{max} &= 2^{EV_{100}} \frac{78}{q \cdot S} \\
$$
代入常用$q=0.65$与$ISO=100$简化为：
$$
L_{max} &= 2^{EV_{100}} \times 1.2 = {L\frac{100}{12.5}} \times 1.2 = 9.6 \times L \\
$$
曝光值$H$的定义为$H=\frac{1}{L_{max}}$ - 最后我们得到将场景辉度归一的完整式子，非常简单：
$$
L' = \frac{L_{pix}}{9.6 \times L}
$$

#### 测光

Tonemapping需要知道场景辉度情况。朴素的，可以直接对最终lighting buffer进行mip chain生成：字面地求平均后取其最后$1\times1$ mip值的辉度值。不过问题也很明显。摘自[Automatic Exposure - Krzysztof Narkowicz](https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/)：当场景*大部份很暗*或存在*少数极亮*光源时，*整体*平均值会受到很大影响：相机对着的主体可能并看不清楚。

利用直方图则可以忽略这些极值。实现上如下：我们将场景光照映射到一定曝光范围后去做binning，最后丢掉极值情况后加权取和得到**平均辉度**。完整实现如下：

```glsl
#include "ICommon.slang"

uniform UBO globalParams;

Texture2D<float4> lighting;
RWStructuredBuffer<Atomic<uint>> bins; // 64 bins. Don't forget to clear.

groupshared Atomic<uint> binGS[64];
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid: SV_DispatchThreadID, uint gid : SV_GroupIndex) {
    binGS[gid].store(0u);
    GroupMemoryBarrierWithGroupSync();

    float4 value = 0u;
    if (tid.x < globalParams.fbWidth && tid.y < globalParams.fbHeight)
        value = lighting.Load(int3(tid.x,tid.y,0));
    float luma = dot(value.xyz, float3(0.2126, 0.7152, 0.0722));
    float EV = log2(luma + EPS);
    int bin = clamp((int)floor(saturate((EV - globalParams.camMinEV) / (globalParams.camMaxEV - globalParams.camMinEV)) * 64.0),0, 63);

    binGS[bin].add(1);
    GroupMemoryBarrierWithGroupSync();
    bins[gid].add(binGS[gid].load());
}
```

加权平均部分沿用了之前wave intrinsic的求和trick，不再多说。实现如下：

```glsl
#include "ICommon.slang"

uniform UBO globalParams;
StructuredBuffer<uint> bins;
RWStructuredBuffer<float> output; // Final scene avg. luminance
groupshared Atomic<uint> countGS, weightGS;
[shader("compute")]
[numthreads(64, 1, 1)]
void reduce(uint gid : SV_GroupIndex, uint groupID : SV_GroupID) {
    countGS.store(0u); weightGS.store(0u);
    GroupMemoryBarrierWithGroupSync();
    // Drop extremities
    bool keep = (gid >= 2 && gid <= 48);
    uint count = bins[gid] * keep;
    uint weighted = count * gid;
    uint countWave = WaveActiveSum(count);
    uint weightWave = WaveActiveSum(weighted);
    if (WaveIsFirstLane()){
        countGS.add(countWave);
        weightGS.add(weightWave);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gid == 0) {
        // sum = Num_i * (EV_i - min)/(max-min)*64
        uint countAll = countGS.load(); // No. samples in range
        uint weightAll = weightGS.load(); // Weighted samples sum
        float meanEV = ((float)weightAll / countAll) / 64.0f * (globalParams.camMaxEV - globalParams.camMinEV) + globalParams.camMinEV;
        float meanLuma = exp2(meanEV);
        float meanLumaPrev = output[0];
        meanLumaPrev = isnan(meanLumaPrev) ? meanLuma : meanLumaPrev;
        float lumaAdapted = meanLumaPrev + (meanLuma - meanLumaPrev) * clamp(globalParams.camAdaptCoeff,0.0f,1.0f);
        output[0] = lumaAdapted;
    }
}

```

其中`camAdaptCoeff`来自 [8.1.4 Automatic exposure](https://google.github.io/filament/Filament.md.html#imagingpipeline/physicallybasedcamera/automaticexposure),在CPU侧计算。式子为：
$$
L_{avg} = L_{avg} + (L - L_{avg}) \times (1 - e^{-\Delta t \cdot \tau})
$$
借此可以产生“自适应”效果，同时规避场景变化可能带来辉度突变。

### Tonemapping
