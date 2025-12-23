---
author: mos9527
lastmod: 2025-12-23T19:14:48.635279
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

#### 镜面反射（完美反射）

完美的镜面反射的BRDF Lobe是个“光线”——他的分布**在且仅在一个单独方向**上。这可以用狄拉克$\delta$函数表达：

$$
\int f(x)\, \delta(x - x_0)\, \mathrm{d}x = f(x_0)
$$
而在BRDF中积分后，渲染等式的以下恒等关系是应该成立的：出射能量等同$入射能量*菲涅耳反射量$
$$
L_o(\omega_o) =
\int_{H^2(\mathbf{n})}
f_r(\omega_o, \omega_i)\,
L_i(\omega_i)\,
\lvert \cos \theta_i \rvert\,
\mathrm{d}\omega_i = F_r(\omega_r)\, L_i(\omega_r)
$$
一个直觉的想法就是直接用狄拉克表达$f_r$。但是这样积分完后会多出一个 $cos \theta_r$：
$$
f_r(\omega_o, \omega_i) = \delta(\omega_i - \omega_r)\, F_r(\omega_i) \newline
L_o(\omega_o) =
\int_{H^2(\mathbf{n})}
\delta(\omega_i - \omega_r)\,
F_r(\omega_i)\,
L_i(\omega_i)\,
\lvert \cos \theta_i \rvert\,
\mathrm{d}\omega_i = F_r(\omega_r)\, L_i(\omega_r)\, \lvert \cos \theta_r \rvert
$$
...那么一个更直觉的做法就是把这个常数拆出来：要记得我们的反射方向是已知量。最后得到镜面BRDF的最终形式：
$$
f_r(\omega_o, \omega_i) = F_r(w_r)\frac{\delta(w_i-w_r)}{|cos\theta_r|}
$$

##### f/Eval

BRDF已经给出来了。不过处理他的PDF很棘手：这里是为了镜面反射情况（roughness=0或很小）下的分布：回顾之前的Lobe图案，他只在唯一一个完美反射的方向有信号。

![image-20251221181248305](/image-foundation/image-20251221181248305.png)

PDF的表达将很困难。该情况概率本身是个狄拉克$\delta$函数：全域除原点都为$0$，而积分是$1$。那么PDF在该点上则会是无穷大！

PBRT在这里对所有方向姑且直接返回$0$。应为单点真去表达的话，你会得到一个无穷亮的像素（firefly）！

##### Sample_f/Sample

出射向量是易知的。我们在本地切空间计算，那么$n(0,0,1)$，$w_o(x,y,z)$围绕他的反射向量很简单，`Reflect()`后就是$w_r(-x,-y,z)$

注：PBRT中的`Reflect(v,n)`和HLSL/GLSL中的`Reflect(v,n)`不一样：

- ```c++
  public float3 Reflect(float3 wo, float3 n) {
      //     vvv wo is flipped (pointing away)
      return -wo + 2 * dot(wo, n) * n;
  }
  ```

- GLSL/HLSL中为

```c++
genType reflect(genType I, genType N);
---
For a given incident vector I and surface normal N reflect returns the reflection direction calculated as I - 2.0 * dot(N, I) * N.
```

他的“入射”向量是翻转的：理由嘛，就是我们在PBRT建模光路时是习惯从**相机到入射**的“反向”。

最后，他的PDF仍旧是个狄拉克。但是采样积分继续用$0$表示会很难受：蒙特卡洛会除以这个PDF。PBRT在此规定让狄拉克PDF在采样中的值一直为$1$。

##### IBxDF 实现

没有，也不必要——这里的式子会在后面设计反射的BxDF反复利用...接下来介绍当反射面并非“完美”，而带粗糙度的情况。

 #### Microfacet（微面）理论及建模

在建模光泽（粗糙“镜面”）反射之前，我们需要知道他是怎么【采样】光线的——不同于朗伯反射，材质本身也会影响Lobe的形状，而显得更“光滑”和“粗糙”。现代 PBR 建模会使用Microfacet（微面）理论描述这一情况。

![image-20251221170507477](/image-foundation/image-20251221170507477.png)

Microfacet 理论中存在以下三种事件：（a）表现 **Masking**，即**出射光**被微面遮挡，（b）表现 **Shadowing**，即**入射光**被微面遮挡，与（c）**内反射**，光路在微面内反射多次后来到视角。（图源 [9.6 Roughness Using Microfacet Theory](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory.html) -  "Three Important Geometric Effects to Consider with Microfacet Reflection Models"）

从宏观角度建模微观事件的手段往往是统计学——PBRT中使用 **Trowbridge-Reitz （GGX）** 分布来建模微面（Microfacet）理论。其中定义以下函数：

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

值得注意的是（c）情况在这里并未讨论，这里留了一个伏笔——之后在Multiscatter GGX中会再次提及...

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

- PBRT提供了 `EffectivelySmooth` 方法，引导实现用前文介绍的镜面反射情况处理以避免（一定会出现且严重的）浮点数精度问题

- 这里的$G$函数同时表达Shadowing-Masking。在很多实现（如英伟达 https://github.com/nvpro-samples/nvpro_core2）及RTR4介绍中，混合Shadowing和Masking往往写成：
    $$
    G_1(w_i)G_1(w_o)
    $$
    这蕴含着入射（shadowing）和出射（masking）事件不相关。[PBRT指出这是过于保守的（太低）](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#TheMasking-ShadowingFunction)，但相关性存在：试想一个很高的‘山封’：从入射/出射两个角度都看不到。最后的形式来自 [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 2014](https://jcgt.org/published/0003/02/03/paper.pdf):

    ![image-20251221182209229](/image-foundation/image-20251221182209229.png)

    其中$\Lambda$已在实现中给出。

#### 光泽反射 （Torrance-Sparrow）

PBRT在介绍完漫反射后给出了ConductorBxDF及DieletricBxDF的定义——这里暂时不对他们进行直接介绍，但是其表达“粗糙度”的BRDF模型基础是一样的：来自 [Theory for Off-Specular Reflection From Roughened Surfaces - Torrance, Sparrow 1967](https://www.graphics.cornell.edu/~westin/pubs/TorranceSparrowJOSA1967.pdf)

之前提过对完全镜面/Specular情况的特殊处理，我们先很快地给出他PDF的定义：**恒为0**（回忆他是狄拉克函数$\delta(wi-wr)$）。对应的，其BSDF Eval（f）**也为0**,理解成球面上只【无穷小】的一点能表现入射光的【所有】能量：很显然，要表达将又是个无穷大，而这是做不到的。

##### 雅可比行列式

![image-20251222083829083](/image-foundation/image-20251222083829083.png)

图源RTR4 p337——建模微面BRDF时，利用half-vector （图中 $\mathbf h$，后面记为$w_m$） 建模微面的法线会很方便，这点之后也能见识到。

不过去采样$w_m$有个问题：我们最后给【要的】是$w_i$。采样前者的话，二者并不在同一个空间内（绿色vs紫色）：
![image-20251222084511331](/image-foundation/image-20251222084511331.png)

图源 [PBRT](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#x5-TheHalf-DirectionTransform)。在（a）的平面情况，我们有简单的 $\theta_m = \frac{\theta_o + \theta_i}{2} $映射，他的雅可比行列式即$\frac{d\theta_m}{d\theta_i}=\frac{1}{2}$；而在(b),(c),(d)的球面中，立体角映射本身并不好找，但是他的雅可比行列式：
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

VNDF重要性采样和BRDF本身已经介绍过，这里用起来即可。代码将在之后实现各x类BxDF时给出。

### 反射与折射

我们已经有了足够的数学工具建模光的【反射】概率模型。PBRT中，[9.3 Specular Reflection and Transmission](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission.html) 被放在之前介绍，不过前面其实也只有一笔带过的菲涅耳$F$等很少的一部分需要这里的知识，索性~~拖到~~现在记笔记。

#### 反射定律

现实中不存在完美的镜面：【能量】在接触表面后多少会被吸收：至于“多少”，这里之后同折射部分一并介绍。

不过就建模*光路*而言，满足入射角=出射角的情况一概归类于此：这里已在前面BxDF部分介绍过，不再多提。

#### 折射定律

![image-20251223140821247](/image-foundation/image-20251223140821247.png)

初高中学过的光的折射：**斯涅尔定律（Snell's Law）**告诉我们，光的折射角和入射角有以下关系（即入射面1,出射面2）：
$$
{\displaystyle {\frac {\sin \theta _{i}}{\sin \theta _{t}}}=n_{1,2}={\frac {n_{1}}{n_{2}}}={\frac {v_{2}}{v_{1}}}}
$$
其中折射率(Index Of Refraction, IOR) 记为 $n = \frac{n_t}{n_i}$。向量计算参考以下PBRT实现：

```c++
// https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#SnellrsquosLaw
public bool Refract(float3 wi, float3 n, float eta /* IOR */, out float3 wt, out float etap){
    float cosTheta_i = dot(n, wi);
    if (cosTheta_i < 0) { // Inside an object?
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }
    float sin2Theta_i = 1 - Sqr(cosTheta_i);
    float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1) // Total internal reflection - transmission impossible
        return false;
    float cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    wt = eta * -wi + (eta * cosTheta_i - cosTheta_t) * n;
    etap = eta;
    return true;
}
```

几个细节：

- 全反射情况下返回`false`，即光密到光疏的入射角$\theta_i > \theta_c = \sin^{-1}{\frac{1}{n}}$
- `etap`接受假设介面为入射面时对应的折射率。计算即$\frac{1}{n}$

#### 菲涅耳方程

##### 实数折射率

之前提到的$F_r$ - 菲涅耳定律给出了在光到材质上后，**反射与折射【能量】的关系**。计算本身涉及电磁相关波知识...大物好久没看也基本忘了，这里只给出形式

- 垂直(s)偏振的反射比为：

$$
{\displaystyle R_{s}=\left({\frac {n_{1}\cos \theta _{i}-n_{2}\cos \theta _{t}}{n_{1}\cos \theta _{i}+n_{2}\cos \theta _{t}}}\right)^{2}}
$$



- 平行(p)偏振的反射比为：
$$
{\displaystyle R_{p}=\left({\frac {n_{1}\cos \theta _{t}-n_{2}\cos \theta _{i}}{n_{1}\cos \theta _{t}+n_{2}\cos \theta _{i}}}\right)^{2}}
$$

- s,p偏振等量（无偏振）时，入射光的反射比即为：

$$
{\displaystyle F_r={\frac {R_{s}+R_{p}}{2}}\,}
$$

- 折射比很直接
  $$
  F_t = 1 - F_r
  $$

$\theta_i$已知，$\theta_t$的计算利用折射定律。以下即实数折射率计算反射比的PBRT实现：

```c++
// https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#TheFresnelEquations
// Percentage of light reflected (or otherwise transmitted) at a dielectric interface
// with respect of incident angle theta and IOR eta
public float FrDielectric(float cosTheta_i, float eta) {
    // vv Same snippet for calculating Snell's law
    if (cosTheta_i < 0) { // Inside an object?
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }
    float sin2Theta_i = 1 - Sqr(cosTheta_i);
    float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1) // Total internal reflection
        return 1.0f; // All scattering is in the reflection, no transmission component
    float cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    // ^^
    float r_parl = (eta * cosTheta_i - cosTheta_t) /
                   (eta * cosTheta_i + cosTheta_t);
    float r_perp = (cosTheta_i - eta * cosTheta_t) /
                   (cosTheta_i + eta * cosTheta_t);
   return (Sqr(r_parl) + Sqr(r_perp)) / 2; // Power of both polarizations
}
```

在电介质材料里，该式子足矣建模其反射/折射能量关系——但对于导体而言则不是如此。

##### 复数折射率

[9.3.6 The Fresnel Equations for Conductors](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#TheFresnelEquationsforConductors) 介绍了折射率为$n - ik$的复数形式时的计算。实现如下：

```c++
// https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#TheFresnelEquationsforConductors
// eta takes a complex form (IOR, k), where k is the absorption coefficient
public float FrComplex(float cosTheta_i, complex eta){
    cosTheta_i = clamp(cosTheta_i, 0, 1);
    // ^^ Ignore the case of inside an object for conductors as it's attenuated rapidly
    float sin2Theta_i = 1 - Sqr(cosTheta_i);
    complex sin2Theta_t = complex(sin2Theta_i) / Sqr(eta);
    complex cosTheta_t = sqrt(complex(1) - sin2Theta_t);
    complex r_parl = (eta * complex(cosTheta_i) - cosTheta_t) /
                     (eta * complex(cosTheta_i) + cosTheta_t);
    complex r_perp = (complex(cosTheta_i) - eta * cosTheta_t) /
                     (complex(cosTheta_i) + eta * cosTheta_t);
    return (norm(r_parl) + norm(r_perp)) / 2; // Power of both polarizations
}
```

$k$部分为[【消光系数】](https://en.wikipedia.org/wiki/Refractive_index#Complex_refractive_index)，再次是一个测量值——**金属的消光系数很大，以至于会对反射率产生不可忽略的影响**：吸收（折射）多到光线无法穿透而变得不透明！

离线管线，和部分参考渲染器在建模金属时一般用的也是该形式：Blender也是如此。作为例子：

这个值和IOR一样，这个值和入射波长也有关系。RGB光谱$\lambda = (630nm,532nm,465nm)$下在 https://refractiveindex.info/?shelf=main&page=Rakic查表：

- 金子有$n=(0.18836,0.54836,1.3319), k=(3.4034,2.2309,1.8693)$ —— 插入 Blender 的 Metallic BSDF 中如下：

![image-20251223145645767](/image-foundation/image-20251223145645767.png)

- 铝则为$n=(1.4303,0.93878,0.68603), k=(7.5081,6.4195,5.6351)$，如下：

![image-20251223150031876](/image-foundation/image-20251223150031876.png)

###### n,k 估计

输入这两个测量值很麻烦。[Artist Friendly Metallic Fresnel, Gulbrandsen 2014](https://jcgt.org/published/0003/04/03/paper.pdf) 给出了由两个RGB参数估计$n,k$的方法。这里只给出实现，来自 [Blender Cycles](https://projects.blender.org/blender/blender/src/commit/91800d13ff20aa4aae5c0b767014fafbab383107/intern/cycles/kernel/closure/bsdf_microfacet.h#L274)：

```c++
// Approx F0 to Complex Fresnel IOR terms
// "Artist Friendly Metallic Fresnel", Gulbrandsen 2014, https://jcgt.org/published/0003/04/03/paper.pdf
public void FresnelFromF0(float r /* baseColor */, float g /* specularTint */, out float n, out float k){
    r = clamp(0.01, 0.99, r);
    float sqrt_r = sqrt(r);
    n = lerp((1.0f + sqrt_r) / (1.0f - sqrt_r), (1.0f - r) / (1.0f + r), g);
    k = SafeSqrt((r * Sqr(n + 1) - Sqr(n - 1)) / (1.0f - r));
}
public void FresnelFromF0(float3 r /* baseColor */, float3 g /* specularTint */, out float3 n, out float3 k){
    for (int i = 0; i < 3; i++)
        FresnelFromF0(r[i], g[i], n[i], k[i]);
}
```

以上即为对电介质和导体反射率计算所需的一切工具。他们在接下来的材质建模会变得很方便！

PBRT里介绍的 [9.4 Conductor BRDF](https://www.pbr-book.org/4ed/Reflection_Models/Conductor_BRDF.html), [9.5 Dielectric BSDF](https://www.pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF.html)暂不直接记录：他们将间接地在接下来的模型中得到体现。

### glTF 材质模型

毕竟到目前为止，glTF是我们唯一的场景格式。要实现则需要把他的材质模型映射到我们目前PBRT风格的BxDF中。

![pbr](/image-foundation/gltf-metal-rough-complete-model.svg)

上图来自 [glTF 2.0 Spec Appendix B](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation)——glTF在**电介质**和**导体**间做线性插值，做法对应PBRT的 [MixMaterial](https://www.pbr-book.org/4ed/Textures_and_Materials/Material_Interface_and_Implementations#x1-MixMaterial)。对两者材质本身而言：

#### 电介质模型

<img src="/image-foundation/image-20251223153046049.png" alt="image-20251223153046049" style="zoom:50%;" />

glTF的该模型可以认为是和PBRT中的`DieletricBxDF`与`DiffuseBxDF`做的`LayeredBxDF`的“简化”版本：*上层光泽层【反射]+【折射】到下层漫【反射】再次【折射】出介面*。

不过，注意图中x部分：这里的层次间叠加(`fresnel_mix`)**不考虑介面间的反射**，二次反射会直接被忽略，能量消失！

再次地，这是一个single-scattering模型：所谓“简化”就是这个意思。PBRT中在介面**多次**NEE做Random Walk，还需考虑介面厚度及衰减问题...这是ground truth答案，虽然跑起来会很慢。在此，我们只做**一次** NEE——这不可避免地会产生能量损失，但这个问题可以留给未来的自己解决...

#### 导体模型

<img src="/image-foundation/image-20251223153124455.png" alt="image-20251223153124455" style="zoom:50%;" />

不必担心，这里（假设metallic=1）的表现和`ConductorBxDF`是一致的。我们已经介绍过他的（复数）$F_r$，插入之前我们的光泽反射BSDF即可得到导体情况的BSDF。

#### fresnel_mix 的由来

可以看到，**电介质**材质有两个BRDF Lobe需要采样：光泽$w$和漫反射$w\prime$。BRDF间的混合并非加法：这样做很显然是能量不守恒的。但从采样的角度出发：在一个点上，这两个$w$都可能是被采样到的光线（参考上图）。如果知道这两者采样**【可能性】**的话，岂不是可以做任意选择而去逼近混合后的结果？

这既是[LayeredBxDF中用到的NEE/Next Event Estimation（次事件估计）的思想](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Scattering_from_Layered_Materials#fragment-SamplenexteventforlayeredBSDFevaluationrandomwalk-0)。而回顾我们之间讨论过的菲涅耳方程：我们很清楚有**【多少】**能量会到达下一层（然后反射），又有多少会被直接反射：_反射率_准确地表达了这样的比例！

接下来采样中对两个Lobe的混合也将这么做。在此之前还有一个问题：导体/电介质二者的混合应该怎么做？再次根据`metallic`值NEE可取，但其实不必如此。

```c++
// Fresnel eval approximation of layered materials
// See https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#fresnel
public float3 glTFFresnelMix(float VdotH, float3 bottom, float3 top, float ior = 1.5f){
    float F0 = pow((ior - 1) / (ior + 1), 2);
    float3 F = SchlickFresnel(F0, 1.0f, VdotH);
    return lerp(bottom, top, F);
}
```

#### 导体/电介质合并

不难发现，这两者都的光泽BRDF仅依赖一个**一样的**$\alpha$粗糙度：这意味着他们的不同，**在且仅在于他们的菲涅耳值**——从采样到PDF都是一样的。既然是比例混合，不妨直接**线性混合他们最终的菲涅耳项**，丢给同样的BRDF计算？

这正是诸多glTF实现中的做法——如此，我们将电介质的lobe和导体lobe等效地合并成一个glossy lobe。这也是为什么在接下来的实现中，你只能看到一次NEE的原因。NEE概率采样如下：

```c++
// Fresnel sampling approximation of layered materials
// This is glTFFresnelMix in a statistical form.
public float glTFFresnelNEE(float NdotV, float ior = 1.5f)
{
    float F0 = pow((ior - 1) / (ior + 1), 2); // IOR=1.5->0.04
    return SchlickFresnel(F0, 1.0f, NdotV);
}
```

但是这还不够：设想metallic=1的情况，完全金属——diffuse lobe会消失。毕竟是线性组合，我们对概率加权metallic即可：

```c++
float probGlossy = glTFFresnelNEE(ClampedCosTheta(wo));
probGlossy = lerp(probGlossy, 1.0f, metallic); // Fully metallic means there's no diffuse lobe
```

#### 菲涅耳项估计

计算菲涅耳本身在之前介绍过——而前面用了`ShlickFresnel`。当然，mix `FrDieletric`和`FrConductor`在这里是正确的...但用到的三角函数是不是有些多？

此外，glTF对导体BSDF的表达依赖于RGB BaseColor。我们确实*也*知道如何估计表现他的$n,k$折射率及消光系数（见前文）——但在此有无必要则值得思考。

在各种glTF光栅器及实时渲染工具中，用到的并非之前从波向量出发的计算方式，而是以下估计形式，来自[An Inexpensive BRDF Model for Physically-based Rendering, Shlick 1994](https://web.archive.org/web/20200510114532/http://cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf)
$$
{\displaystyle R(\theta )=R_{0}+(1-R_{0})(1-\cos \theta )^{5}} \newline
{\displaystyle R_{0}=\left({\frac {n_{1}-n_{2}}{n_{1}+n_{2}}}\right)^{2}}
$$
用$n=\frac{n1}{n2}$表示
$$
R_0 = (\frac{n-1}{n+1})^2
$$

实现很简洁，如下：

```c++
public float SchlickFresnel(float F0, float F90, float cosTheta)
{
  return F0 + (F90 - F0) * pow(1.0F - cosTheta, 5.0F);
}
```

#### glTFBSDF

就此我们完成了整个 Single-scatter （单次反射）的glTF材质实现。Slang部分如下：

```c++
// Cheap, single-scattering LayeredBxDF alternative with glTF's metallic-roughness model
// See also Appendix B. https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#material-structure
public struct glTFBSDF : IBxDF {
    float3 baseColor;
    float metallic;
    TrowbridgeReitzDistribution mfDistrib;
    public __init(float3 baseColor, float metallic, float roughness) {
        this.baseColor = baseColor;
        this.metallic = metallic;
        float alpha = roughness * roughness;
        this.mfDistrib = TrowbridgeReitzDistribution(alpha, alpha);
    }
    public BxDFFlags Flags() {
        return BxDFFlags::Reflection | BxDFFlags::Glossy;
    }
    public float NEEGlossyProb(float3 wo) {
        float probGlossy = glTFFresnelNEE(ClampedCosTheta(wo));
        return lerp(probGlossy, 1.0f, metallic); // Fully metallic means there's no diffuse lobe
    }
    public BSDFSample Sample_f(float3 wo, float uc, float2 u, TransportMode, BxDFReflTransFlags) {
        float c_min_reflectance = 0.04f;
        // Mixed Metallic/Dielectric Fresnel F0
        float3 F0 = lerp(float3(c_min_reflectance), baseColor, metallic);
        float probGlossy = NEEGlossyProb(wo);
        if (uc < probGlossy) {
            // Sample Glossy
            if (mfDistrib.EffectivelySmooth()) {
                // Dirac delta case
                float3 wi = float3(-wo.x, -wo.y, wo.z); // = wr
                float3 fGlossy = SchlickFresnel(F0, 1.0f, AbsCosTheta(wi));
                // Sampled PDF would be delta, but we represent them as 1s w/o weighting
                // With NEE this is what you get:
                return BSDFSample(fGlossy / AbsCosTheta(wi), wi, 1 * probGlossy, BxDFFlags::SpecularReflection);
            } else {
                float3 wm = mfDistrib.Sample_wm(wo, u);
                // NEE weighting vvvvv
                float pdf = probGlossy * mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
                float3 wi = Reflect(wo, wm);
                wi = FaceForward(wi, float3(0,0,1));
                float3 fGlossy = SchlickFresnel(F0, 1.0f, ClampedDot(wo, wm));
                float3 f = mfDistrib.D(wm) * fGlossy * mfDistrib.G(wo, wi) / (4 * AbsCosTheta(wi) * AbsCosTheta(wo));
                return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
            }
        } else {
            // Sample Diffuse
            float3 wi = SampleCosineHemisphere(u);
            wi = FaceForward(wi, float3(0,0,1));
            // NEE weighting vvvvvvvvvvv
            float pdf = (1 - probGlossy) * CosineHemispherePDF(ClampedCosTheta(wi));
            // Diffuse lobe is the bottom lobe. Exiting from the dielectric we have an IOR of 1/1.5,
            // which very conveniently - still approx to a F0=0.04 for dielectrics.
            // This time though - we want the transmittance (1-reflectance).
            float fDiffuse = (1.0f - SchlickFresnel(c_min_reflectance, 1.0f, AbsCosTheta(wo))) ;
            return BSDFSample(baseColor * fDiffuse * (1.0f - metallic) * InvPi, wi, pdf, BxDFFlags::DiffuseReflection);
        }
    }
    public SampledSpectrum f(float3 wo, float3 wi, TransportMode) {
        if (wo.z <= 0 || wi.z <= 0) return 0; // Only mirror reflection.
        float3 wm = normalize(wo + wi);
        float c_min_reflectance = 0.04f;
        float3 f0 = lerp(float3(c_min_reflectance), baseColor, metallic);
        // Same fresnel terms as before.
        float3 fGlossy = SchlickFresnel(f0, 1.0f, ClampedDot(wo, wm));
        float3 fDiffuse = (1.0f - SchlickFresnel(c_min_reflectance, 1.0f, AbsCosTheta(wo))) ;
        // The two lobes - remember NEE only affects the PDF in sampling, not the eval.
        SampledSpectrum diffuseBSDF = baseColor * fDiffuse * (1.0f - metallic) * InvPi;
        SampledSpectrum specularBSDF = mfDistrib.D(wm) * fGlossy * mfDistrib.G(wo, wi) / (4 * AbsCosTheta(wi) * AbsCosTheta(wo));
        // NEE for the combined lobe
        if (mfDistrib.EffectivelySmooth()){
            // PDF is dirac delta, which for eval is impossible to represent
            // This is the specular case again, so eval there has zero contribution
            return diffuseBSDF;
        } else {
            return diffuseBSDF + specularBSDF;
        }
    }
    public float PDF(float3 wo, float3 wi, TransportMode, BxDFReflTransFlags) {
        float3 wm = normalize(wo + wi);
        float diffusePDF = CosineHemispherePDF(ClampedCosTheta(wi));
        float specularPDF = mfDistrib.PDF(wo, wm) / (4 * ClampedDot(wo, wm));
        if (mfDistrib.EffectivelySmooth())
            specularPDF = 0.0f;
        return diffusePDF * (1 - NEEGlossyProb(wo)) + specularPDF * NEEGlossyProb(wo);
    }
};
```

### 能量守恒改进

进行白炉测试：粗糙度越高变得越暗...? 同时，球体边缘部分情况更严重。

![image-20251223183648108](/image-foundation/image-20251223183648108.png)

我们的BRDF出问题了吗？并非。请看~~VCR~~ Blender中的同样场景：

![image-20251222220439188](/image-foundation/image-20251222220439188.png)

...消失了。Cycles通过白炉测试，但这是因为右下角的`Specular`选项使用的是`Multiscatter GGX`；若换成`GGX`：

![image-20251222220353801](/image-foundation/image-20251222220353801.png)

变暗“问题”复现！还记得之前讨论微面情况（c）的内反射：GGX是不处理，而当作被“吸收”而显得更“暗”。真实材质确实可能吸收能量，但对我们目前的建模而言这不应该。这里在[glTF Spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#coupling-diffuse-and-specular-reflection)中也有提及：

> Microfacet models often do not consider multiple scattering. The shadowing term suppresses light that intersects the microsurface a second time. [Heitz et al. (2016)](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#Heitz2016) extended the Smith-based microfacet models to include a multiple scattering component, which significantly improves accuracy of predictions of the model. We suggest to incorporate multiple scattering whenever possible, either by making use of the unbiased stochastic evaluation introduced by Heitz, or one of the approximations presented later, for example by [Kulla and Conty (2017)](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#KullaConty2017) or [Turquin (2019)](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#Turquin2019).

接下来就这里的Hetiz及Turquin方法进行复现。不过在此之前，对于边缘变暗的情况，我们可以提前处理。

#### Fresnel 补偿

记得之前提到【折射】出去时，仅仅看一次事件的话是只能传出$1-F_r$的能量的。但直觉和物理事实告诉我们，在这里并不吸收能量/转化成其他形式的模型中，多少传入就应该多少传出。

TODO:
$$
\int_{H^2} F(\theta) cos\theta d\omega
$$

$$
\int_{0}^{2\pi}{\int_{0}^{\frac{\pi}{2}}}{F(\theta)cos\theta d\omega}
$$

TBD

#### Heitz 2016 - Ground Truth Random Walk

记得$G1$ Masking/Shadowing 函数表达的量：宏观面内沿某视角$\mathbf{v}$可见的微面比例。

![img](/image-foundation/diagram_shadowing_masking.png)

而现在所用的Single Scattering模型会导致折射接触到的面贡献被忽略，但现实中是能继续反弹出来的：图源 [4.7.2 Energy loss in specular reflectance](https://google.github.io/filament/Filament.md.html#materialsystem/improvingthebrdfs/energylossinspecularreflectance)。Roughness越大，散射走的越多，能量损失越大，这可以解释“变暗”情况。

![img](/image-foundation/diagram_single_vs_multi_scatter.png)

在完成微面内完整光路的**ground truth**方法由 [Multiple-Scattering Microfacet BSDFs with the Smith Model, Heitz 2016](https://eheitzresearch.wordpress.com/240-2/) 提出，不过，实践用的更多的是**查表估计**方法，包括Blender的实现来自 [Practical multiple scattering compensation for microfacet models, Emmanuel 2019](https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf) —— 接下来将对两种方式进行复现。

![img](/image-foundation/multiplescatteringsmith_volumeanalogy1.png)

进入微面/microfacet视‘高度’的层次为microflake，在这里walk采样frensel交互。在 [Blender 4.0 之前 (3.6.x)](https://projects.blender.org/blender/blender/src/tag/v3.6.20/intern/cycles/kernel/closure/bsdf_microfacet_multi.h) 也是其Multiscatter GGX的实现。

![walk-the-walk](/image-foundation/steps.svg)

TBD

#### Turquin 2019 - 预计算查表

[Blender 4.0 以后](https://projects.blender.org/blender/blender/src/commit/fc680f0287cdf84261a50e1be5bd74b8bd73c65b/intern/cycles/kernel/closure/bsdf_microfacet.h#L862) 采用了该方法。

TBD

<h1 style="color:red">--- 施工中 ---</h1>