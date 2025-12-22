---
author: mos9527
lastmod: 2025-12-22T11:22:02.185892
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

### 准备工作

#### SBT (Shader Binding Table) 及管线 API

之前用过了非常方便的Inline Ray Query - 从fragment/pixel，compute可以直接产生光线进行trace：从这里出发进行PT是可行的，这也是[nvpro-samples/vk_mini_path_tracer](https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html#moresamples) 的教学式做法。

不过*完全利用硬件*的RT管线会离不开[SBT/Shader Binding Table](https://docs.vulkan.org/spec/latest/chapters/raytracing.html#shader-binding-table)，即Shader绑定表。除了shader单体更小更快之外，调度也有由驱动优化的可能。

实现上...初见确实给了不少震撼。庆幸自己选择了去写个RHI，不然Vulkan SBT方面API的繁文缛节实在是难受- -

保持不特化shader（即shader内各种魔法宏 - 点名Unity，UE）的原则，以下是最后实践的shader绑定API。此外，整个路径追踪过程的C++部分也在此一览无余。

```c++
renderer->CreatePass(
    "Trace", RHIDeviceQueueType::Graphics, 0u,
    [=](PassHandle self, Renderer* r)
    {
        r->BindBufferUniform(self, GlobalUBO, RHIPipelineStageBits::RayTracingShader, "globalParams");
        r->BindAccelerationStructureSRV(self, TLAS, RHIPipelineStageBits::RayTracingShader, "AS");
        r->BindShader(self, RHIShaderStageBits::RayGeneration, "RayGeneration", "data/shaders/ERTPathTracer.spv",
                      AsBytes(AsSpan(cfg.viewFlags)));
        r->BindShader(self, RHIShaderStageBits::RayClosestHit, "RayClosestHit", "data/shaders/ERTPathTracer.spv",
                      AsBytes(AsSpan(cfg.viewFlags)), /*hit group*/ 0);
        r->BindShader(self, RHIShaderStageBits::RayAnyHit, "RayOpacityAnyHit", "data/shaders/ERTPathTracer.spv",
          AsBytes(AsSpan(cfg.viewFlags)), /*hit group*/ 0);
        r->BindShader(self, RHIShaderStageBits::RayMiss, "RayMiss", "data/shaders/ERTPathTracer.spv",
                      AsBytes(AsSpan(cfg.viewFlags)));
        r->BindShader(self, RHIShaderStageBits::RayAnyHit, "ShadowRayAnyHit", "data/shaders/ERTPathTracer.spv",
                      AsBytes(AsSpan(cfg.viewFlags)), /*hit group*/ 1);
        r->BindShader(self, RHIShaderStageBits::RayMiss, "ShadowRayMiss", "data/shaders/ERTPathTracer.spv",
                      AsBytes(AsSpan(cfg.viewFlags)));
        r->BindBufferStorageRead(self, InstanceBuffer, RHIPipelineStageBits::ComputeShader, "instances");
        r->BindBufferStorageRead(self, PrimitiveBuffer, RHIPipelineStageBits::AllGraphics, "primitives");
        r->BindBufferStorageRead(self, MaterialBuffer, RHIPipelineStageBits::AllGraphics, "materials");
        r->BindTextureSampler(self, TexSampler, "textureSampler");
        r->BindDescriptorSet(self, "textures", gpu->GetTexturePool()->GetDescriptorSetLayout());
        r->BindTextureUAV(self, AccumulatedBuffer, "accumulation", RHIPipelineStageBits::RayTracingShader,
                          RHITextureViewDesc{.format = RHIResourceFormat::R16G16B16A16SignedFloat,
                                             .range = RHITextureSubresourceRange::Create()});

    }, [=](PassHandle self, Renderer* r, RHICommandList* cmd)
    {
        RHIExtent2D wh = r->GetSwapchainExtent();
        r->CmdSetPipeline(self, cmd);
        r->CmdBindDescriptorSet(self, cmd, "textures", gpu->GetTexturePool()->GetDescriptorSet());
        cmd->TraceRays(wh.x, wh.y, 1);
    });
```

#### 随机数生成及采样

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

### BxDF

重头戏。~~只会复制粘贴公式可使不得 （喂）~~ 

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

#### 镜面反射

https://www.pbr-book.org/4ed/Reflection_Models/Conductor_BRDF

TBD。勘误glTFBxDF Sample_f镜面！！

 #### Microfacet（微面）建模

在建模光泽（粗糙“镜面”）反射之前，我们需要知道他是怎么【采样】光线的——不同于朗伯反射，材质本身也会影响Lobe的形状，而显得更“光滑”和“粗糙”。现代 PBR 建模会使用Microfacet（微面）理论描述这一情况。

![image-20251221170507477](/image-foundation/image-20251221170507477.png)

Microfacet 理论中存在以下三种事件：（a）表现 **Masking**，即**出射光**被微面遮挡，（b）表现 **Shadowing**，即**入射光**被微面遮挡，与（c）**内反射**，光路在微面内反射多次后来到视角。（图源 [9.6 Roughness Using Microfacet Theory](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory.html) -  "Three Important Geometric Effects to Consider with Microfacet Reflection Models"）

从宏观角度建模微观事件的手段往往是统计学——PBRT中使用 **Trowbridge-Reitz （GGX）**分布来建模微面（Microfacet）理论。其中定义以下函数：

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

值得注意的是（c）情况在这里并未讨论，这里留了一个“问题”（伏笔）！之后在Multiscatter GGX中会再次提及...

##### VNDF

采样/利用$D$直接表达分布确实可以，但是我们有更优的方法。

回顾刚刚给出的第二个式子（往上划第一个）：在$w$视角观察下看得到的微观面比例为$cos \theta$。整理成以下形式：
$$
\int_{H^2} \frac{D(w_m)G(w,w_m)(w_m \cdot \mathbf n)}{\cos\theta}dw_m = 1
$$
左边的式子的积分是1！看起来是不是有PDF的感觉？而且这里“可见性”的概念也被$G$表达，实在方便。不妨拆出来记为$D_w(w_m)$：
$$
D_w(w_m) = \frac{D(w_m)G(w,w_m)(w_m \cdot \mathbf n)}{\cos\theta}
$$
而这个式子就是VNDF方法——**Visibile Normal Distribution Function**，或可见法线分布函数的所在：不必采样完整的$D$，从视角出发，有多少就采样多少。

#####  重要性采样

![image-20251221174044698](/image-foundation/image-20251221174044698.png)

PBRT 书中的方法来自 [Sampling the GGX Distribution of Visible Normals, Heitz 2018](https://jcgt.org/published/0007/04/01/paper.pdf)：其实很直觉——在前面，我们已经很清楚怎么采样一个**均匀**的半球Lobe：圆面投影半球；不妨将GGX的Lobe也“变形”成半球的形状，做同样的事情。

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

#### 光泽反射 （Torrance-Sparrow）

PBRT在介绍完漫反射后给出了ConductorBxDF及DieletricBxDF的定义——这里暂时不对他们进行直接介绍，但是其表达“粗糙度”的BRDF模型基础是一样的：来自 [Theory for Off-Specular Reflection From Roughened Surfaces, Torrance, Sparrow 1967](https://www.graphics.cornell.edu/~westin/pubs/TorranceSparrowJOSA1967.pdf)

之前提过对完全镜面/Specular情况的特殊处理，我们先很快地给出他PDF的定义：**恒为0**（回忆他是狄拉克函数$\delta(wi-wr)$）。对应的，其BSDF Eval（f）**也为0**,理解成球面上只【无穷小】的一点能表现入射光的【所有】能量：很显然，要表达将又是个无穷大，而这是做不到的。

##### 雅可比行列式

![image-20251222083829083](/image-foundation/image-20251222083829083.png)

图源RTR4 p337——建模微面BRDF时，利用half-vector （图中 $\mathbf h$，后面记为$w_m$） 建模微面的法线会很方便，这点之后也能见识到。

不过去采样$w_m$有个问题：我们最后给【要的】是$w_i$。采样前者的话，二者并不在同一个空间内（绿色vs紫色）：
![image-20251222084511331](/image-foundation/image-20251222084511331.png)

图源 [PBRT](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#x5-TheHalf-DirectionTransform)。在（a）的平面情况，我们有简单的 $\theta_m = \frac{\theta_o + \theta_i}{2} $映射，他的雅可比行列式即$\frac{d\theta_m}{d\theta_i}=\frac{1}{2}$；而在(b),(c),(d)的球面中，平面角映射本身并不好找，但是他的雅可比行列式：
$$
\frac{dw_m}{dw_i} = \frac{sin \theta_m d\theta_m d\phi_m}{sin \theta_i d\theta_i d\phi_i}
$$
是显然的。而球面中：

- $w_i$是反射向量，那么$w_o,w_m$夹角等于$w_i,w_m$夹角，即有$\theta_i = 2\theta_m$；因此也有 $cos \theta_m = w_i \cdot w_m = w_o \cdot w_m$ 。这里和（a）也可以感受到
- $w_o,w_m,w_i$共面，既有$\phi_m = \phi_i$

代入简化：
$$
\frac{dw_m}{dw_i} = \frac{sin \theta_m d\theta_m}{sin 2\theta_m d2\theta_m} \newline
\frac{dw_m}{dw_i} = \frac{sin \theta_m}{2sin 2\theta_m} \newline
$$
倍角公式：
$$
\frac{dw_m}{dw_i} = \frac{sin \theta_m }{4sin \theta_m cos \theta_m} = \frac{1}{4 cos \theta_m} = \frac{1}{4 w_i \cdot w_m} = \frac{1}{4 w_o \cdot w_m} \newline
$$

我们得到了这个变换的雅可比！接下来用于PDF计算也将马上用到。

##### f/Eval

和之前的VNDF理论一致，我们的分布也只关心“可见”部分。他的分布已经给出，但是$D_w(w_m)$是在half-vector空间的：好在我们已经知道了他到入射角变换的雅可比！

PDF $p_{(w_i)}$即为：
$$
p_{w_i} = D_w(w_m) \frac{dw_m}{dw_i} = \frac{D_{wo}(w_m)}{4(w_o\cdot w_m)}
$$

他的BRDF本身也很简单。同样回到渲染公式——引入单个样本蒙特卡洛的形式
$$
L_{\mathrm{o}}(\mathrm{p}, \omega_{\mathrm{o}}) = \int_{\mathrm{H}^2(\mathbf{n})} f_{\mathrm{r}}(\mathrm{p}, \omega_{\mathrm{o}}, \omega_{\mathrm{i}}) L_{\mathrm{i}}(\mathrm{p}, \omega_{\mathrm{i}}) |\cos \theta_{\mathrm{i}}| \, \mathrm{d}\omega_{\mathrm{i}} \approx \frac{f_{\mathrm{r}}(\mathrm{p}, \omega_{\mathrm{o}}, \omega_{\mathrm{i}}) L_{\mathrm{i}}(\mathrm{p}, \omega_{\mathrm{i}}) |\cos \theta_{\mathrm{i}}|}{p(\omega_{\mathrm{i}})}
$$

$w_o,w_i$已知的情况下是有解的：回顾之前$\cos\theta$和$G$的关系和菲涅耳公式表达的“反射率”，[书中给出了这样的恒等关系](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#eq:torrence-sparrow-step-1)：
$$
\frac{f_{\mathrm{r}}\left(\mathrm{p}, \omega_{\mathrm{o}}, \omega_{\mathrm{i}}\right) L_{\mathrm{i}}\left(\mathrm{p}, \omega_{\mathrm{i}}\right)\left|\cos \theta_{\mathrm{i}}\right|}{p\left(\omega_{\mathrm{i}}\right)} \stackrel{!}{=} F\left(\omega_{\mathrm{o}} \cdot \omega_{\mathrm{m}}\right) G_{1}\left(\omega_{\mathrm{i}}\right) L_{\mathrm{i}}\left(\mathrm{p}, \omega_{\mathrm{i}}\right)
$$
$p(w_i)$代入则有：
$$
f_{\mathrm{r}}\left(\mathrm{p}, \omega_{\mathrm{o}}, \omega_{\mathrm{i}}\right) = p(w_i)   F\left(\omega_{\mathrm{o}} \cdot \omega_{\mathrm{m}}\right) G_{1}\left(\omega_{\mathrm{i}}\right) = \frac{D_{wo}(w_m) F(w_o\cdot w_m)G_1(w_i)}{4(w_o\cdot w_m)}
$$
代入VNDF：
$$
f_{\mathrm{r}}\left(\mathrm{p}, \omega_{\mathrm{o}}, \omega_{\mathrm{i}}\right) = \frac{D(w_m) F(w_o\cdot w_m)G_1(w_i)G_1(w_o)}{4cos\theta_i cos \theta_o}
$$
回顾之前“细节”部分提到个高度相关$G(wi,wo)$，这里用上有：
$$
f_{\mathrm{r}}\left(\mathrm{p}, \omega_{\mathrm{o}}, \omega_{\mathrm{i}}\right) = \frac{D(w_m) F(w_o\cdot w_m)G(w_i, w_o)}{4cos\theta_i cos \theta_o}
$$
此即Torrance-Sparrow BRDF的现代形式。

##### Sample_f/Sample

VNDF重要性采样和BRDF本身已经介绍过，这里用起来即可。代码将在之后实现glTF材质时给出。

#### glTF 材质模型

PBRT里介绍的 [9.4 Conductor BRDF](https://www.pbr-book.org/4ed/Reflection_Models/Conductor_BRDF.html), [9.5 Dielectric BSDF](https://www.pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF.html)将不在本篇介绍。鉴于引擎目前实现的材质模型均为glTF标准，以下将利用这里有的数学工具在这里实现。

![pbr](/image-foundation/gltf-metal-rough-complete-model.svg)

##### 多重/Layered BRDF

glTF对多层BRDF Lobe的混合仅做了简单的菲涅耳线性插值（fresnel_mix）——原理上这属于single-scattering模型：介面直接的光路可以多次反弹，但这里没有考虑。在 PBRT 中的建模为`LayeredBxDF`, 来自 [14.3 Scattering from Layered Materials](https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Scattering_from_Layered_Materials.html)，其考虑了多重反射。不过因为还没有看，这里先只使用glTF给出的mix方法。

加权部分参考英伟达 [nvpro_core/nvvkhl/shaders/bsdf_functions.h](https://github.com/nvpro-samples/nvpro_core/blob/ba24b73e3a918adfe6ca932a6bf749a1d874d9b0/nvvkhl/shaders/bsdf_functions.h#L1066) 如下——这里和上图 [glTF Appendix B](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#material-structure) 描述是完全一致的。

```c++
// We combine the metallic and specular lobes into a single glossy lobe.
// The metallic weight is     metallic *    fresnel(f0 = baseColor)
// The specular weight is (1-metallic) *    fresnel(f0 = c_min_reflectance)
// The diffuse weight is  (1-metallic) * (1-fresnel(f0 = c_min_reflectance)) * baseColor
float c_min_reflectance = 0.04f;
float3 f0 = lerp(float3(c_min_reflectance), baseColor, metallic);
float3 fGlossy = SchlickFresnel(f0, 1.0f, VdotH);
float3 fDiffuse = SchlickFresnel(1.0f - c_min_reflectance, 0.0f, VdotH) * (1.0f - metallic);
```

##### 实现

`glTFBSDF`部分如下——这里并没有其他extension的存在，仅包含metallic-roughness模型。

```c++
TBD
```

和nvpro实现有相似之处，不过注意：

- 和PBRT IBxDF模型一致（而非nvpro写法），我们的`f()/Eval`**不包含** $cos\theta$
- 这里的$G$为改进，高度相关的形式；nvpro中为$G1G2$不相关形式
- 此外，我们的计算再次和 PBRT 一致，全部在表面本地切空间进行；法线对应我们的$+Z (0,0,1)$轴

`diffuseBSDF, diffusePDF` 及 `specularBSDF, specularPDF` 即我们之前介绍的式子，计算上只是写成shader而已。

实现完毕。最后是虽迟但到的测试场景~~《Cornell Box 中的维纳斯》~~

![image-20251222110215674](/image-foundation/image-20251222110215674.png)

### 能量守恒改进

值得注意的是反射面中的场景有变暗的情况。这不是巧合，进行白炉测试：

![image-20251222112155620](/image-foundation/image-20251222112155620.png)

### Multiscatter GGX

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