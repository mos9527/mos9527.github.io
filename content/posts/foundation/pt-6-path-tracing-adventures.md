---
author: mos9527
lastmod: 2025-12-21T13:12:32.943085
title: Foundation 施工笔记 【6】- 路径追踪
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

Foundation现在(2025/12/16)也有了能用的RT相关API，Editor的GPUScene也有了BLAS上传/压缩(compact)与逐帧TLAS更新支持。到目前为止用rt做的只有inline query实现硬阴影——在做实时GI相关内容之前，不妨复习下采样/PBR相关知识——那就写个GPU Path Tracer吧？

PBRT/[Physically Based Rendering:From Theory To Implementation](https://pbr-book.org/)/[Kanition大佬v3翻译版](https://github.com/kanition/pbrtbook)，[Ray Tracing Gems 2](https://www.realtimerendering.com/raytracinggems/rtg2/index.html), [nvpro-samples/vk_gltf_renderer](https://github.com/nvpro-samples/vk_gltf_renderer/blob/master/shaders/gltf_pathtrace.slang) 将是我们这里主要的信息来源。

### SBT (Shader Binding Table) 及管线 API

之前用过了非常方便的Inline Ray Query - 从fragment/pixel，compute可以直接产生光线进行trace：从这里出发进行PT是可行的，这也是[nvpro-samples/vk_mini_path_tracer](https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html#moresamples) 的教学式做法。

不过完全利用硬件的RT管线会离不开[SBT/Shader Binding Table](https://docs.vulkan.org/spec/latest/chapters/raytracing.html#shader-binding-table)，即Shader绑定表。除了shader单体更小更快之外，调度也有由驱动优化的可能。此外RHI目前还没有SBT相关设施，借此一并处理。参考 [nvpro-samples/vk_raytracing_tutorial_KHR](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/#step-43-create-basic-ray-tracing-pipeline-structure)

...SBT API应该是Vulkan中最无语的一个设计了。在RHI里决定偷懒，在光追PSO创建过程中直接处理并分组SBT；最后实现完整PT所需的Renderer使用会很轻松，如下：

```c++
...
renderer->CreatePass(
    "Trace", RHIDeviceQueueType::Graphics, 0u,
    [=](PassHandle self, Renderer* r)
    {
        r->BindBackbufferUAV(self, 1u);
        r->BindBufferUniform(self, GlobalUBO, RHIPipelineStageBits::RayTracingShader, "globalParams");
        r->BindAccelerationStructureSRV(self, TLAS, RHIPipelineStageBits::RayTracingShader, "AS");
        r->BindShader(self, RHIShaderStageBits::RayGeneration, "RayGeneration", "data/shaders/ERTPathTracer.spv");
        r->BindShader(self, RHIShaderStageBits::RayClosestHit, "RayClosestHit", "data/shaders/ERTPathTracer.spv");
        r->BindShader(self, RHIShaderStageBits::RayMiss, "RayMiss", "data/shaders/ERTPathTracer.spv");
        r->BindBufferStorageRead(self, InstanceBuffer, RHIPipelineStageBits::ComputeShader, "instances");
        r->BindBufferStorageRead(self, PrimitiveBuffer, RHIPipelineStageBits::AllGraphics, "primitives");
        r->BindBufferStorageRead(self, MaterialBuffer, RHIPipelineStageBits::AllGraphics, "materials");
        r->BindTextureSampler(self, TexSampler, "textureSampler");
        r->BindDescriptorSet(self, "textures", gpu->GetTexturePool()->GetDescriptorSetLayout());

    }, [=](PassHandle self, Renderer* r, RHICommandList* cmd)
    {
        RHIExtent2D wh = r->GetSwapchainExtent();
        r->CmdSetPipeline(self, cmd);
        r->CmdBindDescriptorSet(self, cmd, "textures", gpu->GetTexturePool()->GetDescriptorSet());
        cmd->TraceRays(wh.x, wh.y, 1);
    });
...

```

Shader部分测试输出法线，也很简单。最终效果如下：

![image-20251217085106615](/image-foundation/image-20251217085106615.png)

CPU 部分的工作基本完成了。此外，后处理等（比如tonemapper）会在之后添加。

### 随机数生成及 Viewport 采样

参考 [Ray Tracing Gems 2](https://www.realtimerendering.com/raytracinggems/rtg2/index.html) 的 [Reference Path Tracer](https://github.com/boksajak/referencePT/) - 随机数生成使用了书中介绍的 PCG4；参考实现中有个很有趣的hack，从uint32直接产生$[0,1)$区间的浮点数：这里贴出来。

```glsl
// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
float uintToFloat(uint x) {
	return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}
```

下面PT本身做的蒙特卡洛和各种重要性采样之外，Viewport采样也值得一提——产生图像的并非逐帧刷新，而存在积累过程。

毕竟，我们的随机数种子也是由帧序号初始化的——这样做可以将多帧，可能不同但趋近最终积分的结过积累以逼近。利用UAV实现这一点很容易：

```glsl
RWTexture2D<float4> accumulation;
RWTexture2D<float4> output;
// ...
float4 previous = accumulation[pix];
// Reset when accumlated frame resets, e.g. through camera movement
if (globalParams.ptAccumualatedFrames == 0)
    previous = float4(0);
float4 current = previous + float4(radiance, 0);
accumulation[pix] = current;
float4 average = current / float(globalParams.ptAccumualatedFrames + 1);
output[pix] = float4(average.xyz, 1.0f);
```

还有一个好处是：因为是多帧平均采样，若对镜头做jitter，这里就是一种 ~~五毛钱~~ TAA/Temporal抗锯齿的实现。Primiary Ray生成如下：

```glsl
float3 GeneratePrimaryRay(uint2 pixel, PCG rng)
{
    float2 jitter = lerp(float2(-0.5f), float2(0.5f), float2(rng.sample(), rng.sample()));
    float2 uv = float2(pixel + jitter + 0.5f) / DispatchRaysDimensions().xy;
    float4 ndcPosition = float4(uv, 1.0f, 1.0f);
    ndcPosition.y = 1 - ndcPosition.y;
    ndcPosition.xy = ndcPosition.xy * 2.0f - 1.0f;
    float4 wsPosition = mul(globalParams.inverseViewProj, ndcPosition);
    return normalize(wsPosition.xyz / wsPosition.w - globalParams.camPosition);
}
```

### BRDF 函数

接下来的实现以PBRT的风格完成；为此，我们定义以下结构及界面：

```c++
// -- BxDF Interface
// https://www.pbr-book.org/4ed/Reflection_Models/BSDF_Representation
public struct BSDFSample {
    // Value of the BSDF
    public float3 f;
    // Incoming direction
    public float3 wi;
    // Sampler's PDF
    public float pdf;

    public __init(float3 f, float3 wi, float pdf) {
        this.f = f;
        this.wi = wi;
        this.pdf = pdf;
    }
    public bool IsValid() {
        return pdf >= 0.0;
    }
};
typedef float3 SampledSpectrum; // RGB spectrum
// https://www.pbr-book.org/4ed/Reflection_Models/BSDF_Representation#BxDFInterface
// * wo, wi are in local space (+z is normal direction)
public interface IBxDF {
    // Value of the distribution function for given pair of directions
    public float3 f(float3 wo, float3 wi);

    // [Importance] Sample a direction wi given outgoing direction wo and 2D random, uniform
    // samples uc and u.
    public BSDFSample Sample_f(float3 wo, float uc, float2 u);

    // Evaluates the PDF for a given pair of directions
    public float PDF(float3 wo, float3 wi);
};
```

#### 漫反射（朗伯反射）
![image-20251217172122062](/image-foundation/image-20251217172122062.png)

最简单的漫反射BRDF，也就是朗伯反射（Lambertian Diffuse）。图源Kanition翻译PBRTv3

他的BRDF Lobe很简单：分布是一个半球面；从评估/Eva（已知入射出射）和采样/Sample（已知出射/相机入射未知）两个方向看：

##### f/Eval

回顾渲染公式：
$$
L_o(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) \cos\theta \,d\omega_i
$$
朗伯反射的能量分布是无条件均匀的，那么设$f_r = kR$ 。保证能量守恒，$L_i = 1$积分有
$$
L_o = \int_{H^2}{kR cos\theta d w_i} = kR \int_{H^2}{cos\theta d w_i} = R \newline
k = \frac{1}{\int_{H^2}{cos\theta d w_i}} = \frac{1}{\pi}
$$
即$f_r = \frac{R}{\pi}$.

##### Sample_f/Sample

PBRT中使用了[SampleCosineHemisphere](https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#Cosine-WeightedHemisphereSampling)做重要性采样。

```c++
public float3 SampleCosineHemisphere(float2 u) {
    float2 d = SampleConcentricDisk(u);
    float z = SafeSqrt(1.0 - Sqr(d.x) - Sqr(d.y));
    return float3(d.x, d.y, z);
}
```

采样PDF推导也很直接；这里$x^2 + y^2 + z^2 = 1$，若记ConcentricDisk上极坐标为$(r, \phi)$,最后单位球坐标为$(1, \theta, \phi)$则代入：
$$
x^2 + y^2 = r^2 \newline
z^2 = 1 - r^2 = cos^2{\theta} \newline
sin^2\theta = r^2 \rarr sin \theta = r
$$
接下来求变换$(r,\phi) \rarr (\theta,\phi)$的雅可比行列式：
$$
J =
\left|
\frac{\partial(r, \phi)}{\partial(\theta, \phi)}
\right|
=
\begin{vmatrix}
\cos\theta & 0 \\
0 & 1
\end{vmatrix}
=
\cos\theta.
$$
[我们知道圆盘$(r,\phi)$上采样的PDF](https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#sec:unit-disk-sample)是 $\frac{r}{\pi}$；那么知道行列式后我们可以很轻松地得到该采样方式的PDF为$cos\theta \frac{r}{\pi} = \frac{cos\theta sin\theta}{\pi}$

注意PDF对应球面上的立体角/solid angle，即$dw = sin\theta d\theta d\phi$，$sin\theta$消掉，即得到我们最后采样的PDF

```c++
public float CosineHemispherePDF(float cosTheta) {
    return cosTheta * InvPi;
}
```

##### BxDF

整理完毕如下。这里（和以后的）的`IBxDF`界面和PBRT书中介绍完全一致。

```c++
public struct DiffuseBxDF : IBxDF {
    SampledSpectrum R;
    public __init(SampledSpectrum R) {
        this.R = R;
    }
    public BxDFFlags Flags(){
        return BxDFFlags::DiffuseReflection;
    }
    // BRDF evaluation
    // Constant reflection distribution where:
    // \int_H^2 f(wo,wi) CosTheta(wi) dwi = R
    public SampledSpectrum f(float3 wo, float3 wi, TransportMode) {
        if (!SameHemisphere(wo, wi)) {
            return float3(0.0, 0.0, 0.0);
        }
        return R * InvPi;
    }
    // Draw a sample from cosine-weighted hemisphere
    public BSDFSample Sample_f(float3 wo, float uc, float2 u,TransportMode, BxDFReflTransFlags flags) {
        if (!(flags & BxDFReflTransFlags::Reflection))
            return BSDFSample();
        float3 wi = SampleCosineHemisphere(u);
        if (wo.z < 0.0) wi.z *= -1.0; // Ensure same hemisphere
        float pdf = CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(R * InvPi, wi, pdf, BxDFFlags::DiffuseReflection);
    }
    // Evaluated PDF for a pair of directions
    public float PDF(float3 wo, float3 wi,TransportMode, BxDFReflTransFlags flags) {
        if (!(flags & BxDFReflTransFlags::Reflection) || !SameHemisphere(wo, wi))
            return 0.0;
        return CosineHemispherePDF(AbsCosTheta(wi));
    }
};
```



#### 光泽反射 （GGX）

TBD - Revisit

##### Multiscatter GGX

值得注意的是反射面中的场景有变暗的情况，进行Furnace Test：

TBD - Revisit

记得$G1$ Masking/Shadowing 函数表达的量：宏观面内沿某视角$\mathbf{v}$可见的微面比例。

![img](/image-foundation/diagram_shadowing_masking.png)

而现在所用的Single Scattering模型会导致折射接触到的面贡献被忽略，但现实中是能继续反弹出来的：图源 [4.7.2 Energy loss in specular reflectance](https://google.github.io/filament/Filament.md.html#materialsystem/improvingthebrdfs/energylossinspecularreflectance)。Roughness越大，散射走的越多，能量损失越大，这可以解释“变暗”情况。

![img](/image-foundation/diagram_single_vs_multi_scatter.png)

在完成微面内完整光路的**ground truth**方法由 [Multiple-Scattering Microfacet BSDFs with the Smith Model, Heitz 2016](https://eheitzresearch.wordpress.com/240-2/) 提出，不过，实践用的更多的是**查表估计**方法，包括Blender的实现来自 [Practical multiple scattering compensation for microfacet models, Emmanuel 2019](https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf) —— 接下来将对两种方式进行复现。

###### Ground Truth Random Walk

![img](/image-foundation/multiplescatteringsmith_volumeanalogy1.png)

进入微面/microfacet视‘高度’的层次为microflake，在这里walk采样frensel交互。在 [Blender 4.0 之前 (3.6.x)](https://projects.blender.org/blender/blender/src/tag/v3.6.20/intern/cycles/kernel/closure/bsdf_microfacet_multi.h) 也是其Multiscatter GGX的实现。

![walk-the-walk](/image-foundation/steps.svg)

TBD

##### 预计算查表

[Blender 4.0 以后](https://projects.blender.org/blender/blender/src/commit/fc680f0287cdf84261a50e1be5bd74b8bd73c65b/intern/cycles/kernel/closure/bsdf_microfacet.h#L862) 采用了该方法。

TBD

<h1 style="color:red">--- 施工中 ---</h1>