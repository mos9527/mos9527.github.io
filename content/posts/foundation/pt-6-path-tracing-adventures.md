---
author: mos9527
lastmod: 2025-12-21T18:58:11.496394
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

### BxDF 实现

接下来的实现以PBRT的风格完成，以下将给出的结构及界面将同PBRT书中定义一致。此外，自己将尽力给出以下模型的原理推导。

最后作为参考，还请参阅 [PBRT v4 - 9 Reflection Models](https://www.pbr-book.org/4ed/Reflection_Models.html) 以获取最权威信息；此外，这一部分在[Kanition PBRT v3翻译版](https://github.com/kanition/pbrtbook)中尚未完成，自己尝试的翻译和数学解释也许不够准确——如有错误还烦请指正！

#### 漫反射（朗伯反射）

![image-20251217172122062](/image-foundation/image-20251217172122062.png)

最简单的漫反射BRDF，也就是朗伯反射（Lambertian Diffuse）。图源Kanition翻译PBRTv3

他的BRDF Lobe很简单：分布是一个半球面，而且能量均匀。我们从评估/Eval（已知入射出射方向）和采样/Sample（已知出射/相机入射未知）两个方向解读 PBRT 的实现。

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

即 $f_r = \frac{R}{\pi}$，对应PBRT界面中的`f()`实现。

##### Sample_f/Sample

![image-20251221165710063](/image-foundation/image-20251221165710063.png)

PBRT中使用了[SampleCosineHemisphere](https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#Cosine-WeightedHemisphereSampling)做重要性采样（如图），采样单位圆内点后直接投影在单位球面上。

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
sin^2\theta = r^2 \newline
$$

我们有$sin \theta = r$, 接下来求变换$(r,\phi) \to (\theta,\phi)$的雅可比行列式：

$$
J =
\left|\frac{\partial(r, \phi)}{\partial(\theta, \phi)}\right |= 
\begin{vmatrix} \cos\theta & 0 \newline 0 & 1 \end{vmatrix} = \cos\theta
$$

[我们知道圆盘上采样的PDF](https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#sec:unit-disk-sample)是 $ \frac{r}{\pi} $ 那么知道行列式后我们可以很轻松地得到该采样方式的PDF为 $ cos\theta \frac{r}{\pi} = \frac{cos\theta sin\theta}{\pi} $
这里的“权重”，$cos\theta$，也正是该采样方法名字/Cosine Weighted的来源。

注意PDF对应球面上的立体角/solid angle，即$dw = sin\theta d\theta d\phi$；这里的$sin\theta$消掉，即得到我们最后采样的PDF

```c++
public float CosineHemispherePDF(float cosTheta) {
    return cosTheta * InvPi;
}
```

##### IBxDF 实现

整理完毕如下。这里（和以后的）的`IBxDF`界面和PBRT书中介绍将保证完全一致。

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

 #### Microfacet（微面）建模

在建模光泽反射之前，我们需要知道他是怎么【采样】光线的——不同于朗伯反射，材质本身也会影响Lobe的形状，而显得更“光滑”和“粗糙”。现代 PBR 建模会使用Microfacet（微面）理论描述这一情况。

![image-20251221170507477](/image-foundation/image-20251221170507477.png)

Microfacet 理论中存在以下三种事件：（a）表现 **Masking**，即**出射光**被微面遮挡，（b）表现 **Shadowing**，即**入射光**被微面遮挡，与（c）**内反射**，光路在微面内反射多次后来到视角。（图源 [9.6 Roughness Using Microfacet Theory](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory.html) -  "Three Important Geometric Effects to Consider with Microfacet Reflection Models"）

从宏观角度建模微观事件的手段往往是统计学——PBRT中使用 Trowbridge-Reitz （GGX）分布来建模微面（Microfacet）理论。其中定义以下函数：

- $D(w)$ - [Microfacet Distribution](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#TheMicrofacetDistribution)，代表宏观平面上一点从视角$w$观察，【指向视角$w$】的微面比例；直觉的，以下式子，也即从**所有视角**观察到的面积分，成立：
  $$
  \int_{H^2}{D(w_m)(w_m \cdot \mathbf n)dw_m} = 1
  $$
  
- $G(w)$ - [Masking Function](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#TheMaskingFunction)，代表宏观平面上一点从视角$w$观察，可被【直接观察到】的微面比例；类比平面情况（如图），我们也可以找到球面上他的积分的含义：从**一定视角**，所能“看到”的面的比例
	
	![image-20251221172332451](/image-foundation/image-20251221172332451.png)
	
	在光滑平面下这是熟悉的$cos\theta = N\cdot L$；微面情景中，以下式子也应成立：
	$$
	\int_{H^2}{D(w_m)G(w,w_m)(w_m \cdot \mathbf n)dw_m} = w \cdot \mathbf{n} = \cos\theta
	$$

两个重要的等式关系也将在后面推导VNDF采样中继续使用。GGX $D$, $G$本身的推导在此省略。

值得注意的是（c）情况在这里并未讨论，这里留了一个“问题”（伏笔）！最后在Multiscatter GGX中会再次提及。不过暂时，（c）情况先放一边...接下来则是更**重要**的一个问题

##### GGX 重要性采样 （VNDF）

![image-20251221174044698](/image-foundation/image-20251221174044698.png)

PBRT 书中的方法来自 [Sampling the GGX Distribution of Visible Normals, Heitz 2018](https://jcgt.org/published/0007/04/01/paper.pdf)：采样GGX的Lobe，可以很直觉——在前面，我们已经很清楚怎么采样一个**均匀**的半球Lobe；不妨将GGX的Lobe也“变形”成半球的形状，做同样的事情！

GGX的Lobe是个椭圆体：形状由我们提供的$\alpha$“粗糙度”决定。对各向异性情形则是$\alpha_x, \alpha_y$两个值，而将这个形状变回”圆“则很简单：平面本地切空间内表示($\mathbf n = (0,0,1)$)下，仅需一个缩放变换：
$$
A = \begin{bmatrix}a_x & 0 & 0 \newline 0 & a_y & 0 \newline 0 & 0 & 0\end{bmatrix}
$$
之后，用缩放后的$n$去采样，采样方法和之前几乎一致。但值得注意的是，不同于漫反射：这里的Lobe可能并不完全在正平面内：

![image-20251221174613721](/image-foundation/image-20251221174613721.png)

“裁切”掉这个情况并非难事：限制高度到$[cos\theta, 1]$即可。之后通过构造正交基就能把采样的向量投影做$A$的逆变换，回到正确的Lobe方向中

最后，我们完整的GGX采样实现如下。这里的PDF正是我们之前提到的$D$。

```c++
// https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
// Trowbridge-Reitz (GGX) distribution/shadow-masking functions
public struct TrowbridgeReitzDistribution {
    float alpha_x;
    float alpha_y;
    public __init(float alpha_x, float alpha_y) {
        this.alpha_x = alpha_x;
        this.alpha_y = alpha_y;
    }
    // https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#eq:tr-d-function
    // Distribution/percentage of microfacet normals at surface local point oriented towards wm
    public float D(float3 wm) {
        float tan2Theta = Tan2Theta(wm);
        if (isinf(tan2Theta)) return 0;
        float cos4Theta = Sqr(Cos2Theta(wm));
        float e = tan2Theta * (Sqr(CosPhi(wm) / alpha_x) +
                               Sqr(SinPhi(wm) / alpha_y));
        return 1 / (Pi * alpha_x * alpha_y * cos4Theta * Sqr(1 + e));
    }
    // Fallback to perfectly smooth case where PDF is dirac delta when true
    public bool EffectivelySmooth() {
        return max(alpha_x, alpha_y) < 1e-3f;
    }
    // Lambda for G1 masking-shadowing function at incident direction w
    public float Lambda(float3 w) {
        float tan2Theta = Tan2Theta(w);
        if (isinf(tan2Theta)) return 0;
        float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);
        return (sqrt(1 + alpha2 * tan2Theta) - 1) / 2;
    }
    // Masking function for single incident direction
    public float G1(float3 w) { return 1 / (1 + Lambda(w)); }
    // Height-correlated Masking-Shadowing function (Smith G)
    // ---
    // * G1(wo)G1(wi) assumes independence - which is conservative and can lead to energy loss/darkening
    // * This correlates height fields - assumed as a NDF - to reduce energy loss
    public float G(float3 wo, float3 wi) {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    // Normalized Visible Normal Distribution/VNDF
    // ---
    // For visible microfacets viewed from wm:
    //   \int_H^2 G1(w)G1(wm) max(0,w \cdot wm) dwm = w \cdot n = CosTheta(w) should hold
    // Thus we can derive the PDF for sampling visible normals with respect to incident direction w:
    //   D_w(wm) = \frac{G1(w)}{CosTheta(wm)} D(wm) max(0,w \cdot wm)
    public float D(float3 w, float3 wm) {
        return G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm);
    }
    // Alias of D, the PDF for visible normal's importance sampling described below.
    public float PDF(float3 w, float3 wm) { return D(w, wm); }
    // Importance sampling of Visible Normals
    // ---
    // https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#SamplingtheDistributionofVisibleNormals
    // See also "Sampling the GGX Distribution of Visible Normals" by Eric Heitz (https://jcgt.org/published/0007/04/01/paper.pdf)
    public float3 Sample_wm(float3 w, float2 u) {
        // 1. For anisotropic cases, transform the incident w back to a hemispherical configuration
        //    so we'd always work with a isotropic case.
        float3 wh = normalize(float3(alpha_x * w.x, alpha_y * w.y, w.z));
        if (wh.z < 0) wh = -wh; // Ensure wh is in the upper hemisphere
        // 2. Find a orthonormal basis around wh. This is PBRT's routine, though
        //    buildOrthonormalBasis [Frisvad, 2012] could be also used w/o normalization.
        float3 T1 = wh.z < 0.99999f ? normalize(cross(float3(0,0,1), wh)) : float3(1,0,0);
        float3 T2 = cross(wh, T1);
        // 3. Sample uniformly distributed point on a unit disk and project to clipped hemisphere
        float2 p = SampleUniformDiskPolar(u);
        float h = sqrt(1 - Sqr(p.x)); // Max height on hemisphere
        float s = (1 + wh.z) / 2;
        p.y = (1-s) * h + s * p.y; // Project to clipped hemisphere
        float pz = sqrt(max(0, 1 - LengthSquared(p)));
        float3 nh = p.x * T1 + p.y * T2 + pz * wh; // Apply TBN
        // 4. Reverse the anisotropic scaling to get the sampled microfacet normal
        return normalize(float3(alpha_x * nh.x, alpha_y * nh.y, max(1e-6f, nh.z)));
    }
}
```

注意几个细节：

- PBRT提供了 `EffectivelySmooth` 方法：这里是为了镜面反射情况（roughness=0或很小）下的分布：回顾之前的Lobe图案，他只在唯一一个完美反射的方向有信号。

  ![image-20251221181248305](/image-foundation/image-20251221181248305.png)

  - PDF的表达将很困难。该情况概率本身是个狄拉克$\delta$函数：全域除原点都为$0$，而积分是$1$。那么PDF在该点上则会是无穷大！

    PBRT的解决方案则是用$1$表达采样它的PDF；同时采样时在该case下特殊处理，入射光线直接取出射光线镜像，避免任何精度问题

- 这里的$G$函数同时表达Shadowing-Masking。在很多实现（如英伟达 https://github.com/nvpro-samples/nvpro_core2）及RTR4介绍中，混合Shadowing和Masking往往写成：
    $$
    G_1(w_i)G_1(w_o)
    $$
    这蕴含着入射（shadowing）和出射（masking）事件不相关。[PBRT指出这是过于保守的（太低）](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#TheMasking-ShadowingFunction)，但相关性存在：试想一个很高的‘山封’：从入射/出射两个角度都看不到。最后的形式来自 [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 2014](https://jcgt.org/published/0003/02/03/paper.pdf):

    ![image-20251221182209229](/image-foundation/image-20251221182209229.png)

    其中$\Lambda$已在实现中给出。

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