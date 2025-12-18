---
author: mos9527
lastmod: 2025-12-18T12:19:41.784298
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

### BRDF 采样

#### 漫反射（朗伯反射）
![image-20251217172122062](/image-foundation/image-20251217172122062.png)

先实现最简单的漫反射BRDF，也就是朗伯反射（Lambertian Diffuse）。此外若未提及，BRDF和之前在光栅路径中的选择会是一致的。图源Kanition翻译PBRTv3

回顾渲染函数：
$$
L_o = \int_{\Omega} \underbrace{f_r(x, \omega_i, \omega_o)}_{\text{BRDF}} \times L_i(x, \omega_i) \times \cos(\theta) \, d\omega_i
$$

朗伯反射分布均匀，分布函数$PDF = \frac{1}{\pi}$即$f_r=\frac{BaseColor}{\pi}$。注意在使用重要性采样器为[Cosine-Weighted Hemisphere Sampling](https://www.pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#Cosine-WeightedHemisphereSampling)选择出射光线时，其采样概律 $PDF = \frac{cos\theta}{\pi}$ （推导见前链接）。在蒙特卡罗积分时：


$$
L_o \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f(X_k)}{p(X_k)}
$$

代入BRDF有
$$
L_o \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f_r \times L_i \times cos(\theta)}{\frac{cos(\theta)}{\pi}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\frac{BaseColor}{\pi} \times L_i \times cos(\theta)}{\frac{cos(\theta)}{\pi}} = \frac{1}{N} \sum_{k=1}^{N} BaseColor \times L_i
$$

可以发现$pi$和$cos\theta$是被消掉的。RayGen shader中实现如下：

```glsl
// ... acculmalte throughput * radiance
// Importance sample next ray direction (in surface local space)
rayLocal = SampleCosineHemisphere(u);
// ^^ Samping/BRDF PDF (1/pi) cancels each other out vv
throughput *= baseColor * 1.0f;
```

仅实现该BRDF效果如下——注意到墙壁色彩在其他物体上的间接影响。

![image-20251217180829346](/image-foundation/image-20251217180829346.png)

#### 光泽反射 （GGX）

回顾 RTR4 p337 - 微面（Microfacet）BRDF形式一般如下:
$$
F_{specular} = \frac{D(h, \alpha) G(v, l, \alpha) F(v, h, f0)}{4(n \cdot v)(n \cdot l)}
$$
参考 [Crash Course in BRDF Implementation - Jakub Boksansky's Blog](https://boksajak.github.io/files/CrashCourseBRDF.pdf) （同样的也是 Ray Tracing Gems 2 介绍的方法）,$D$采样使用[Sampling the GGX Distribution of Visible Normals - Eric Heitz](https://jcgt.org/published/0007/04/01/)的Listing；和Diffuse BRDF叠加使用了Fresnel做选择，这里在[上一篇](https://mos9527.com/posts/foundation/pt-5-texture-compression-and-gbuffer/#gltf-metal-rough-%E6%A8%A1%E5%9E%8B)也有提及。

VNDF的PDF中的$D$可以被抵消掉，推导略；最后$PDF$残余的式子shader内记为`SampleGGXVNDFWeight`,如下：

```glsl
// Smith G1 term (masking function) further optimized for GGX distribution (by substituting G_a into G1_GGX)
float G1_SmithGGX(float alpha, float alphaSq, float NoSsq) {
	return 2.0f / (sqrt(((alphaSq * (1.0f - NoSsq)) + NoSsq) / NoSsq) + 1.0f);
}
// A fraction G2/G1 where G2 is height correlated can be expressed using only G1 terms
// Source: "Implementing a Simple Anisotropic Rough Diffuse Material with Stochastic Evaluation", Appendix A by Heitz & Dupuy
float G2_Over_G1_SmithHeightCorrelated(float alpha, float alphaSq, float NoL, float NoV) {
	float G1V = G1_SmithGGX(alpha, alphaSq, NoV * NoV);
	float G1L = G1_SmithGGX(alpha, alphaSq, NoL * NoL);
	return G1L / (G1V + G1L - G1V * G1L);
}
...
// https://jcgt.org/published/0007/04/01/
// PDF is 'G1(NoV) * D'
float3 SampleGGXVNDF(float3 Ve, float2 alpha2D, float2 u) {

	// Section 3.2: transforming the view direction to the hemisphere configuration
	float3 Vh = normalize(float3(alpha2D.x * Ve.x, alpha2D.y * Ve.y, Ve.z));

	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
	float3 T2 = cross(Vh, T1);

	// Section 4.2: parameterization of the projected area
	float r = sqrt(u.x);
	float phi = 2 * PI * u.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5f * (1.0f + Vh.z);
	t2 = lerp(sqrt(1.0f - t1 * t1), t2, s);

	// Section 4.3: reprojection onto hemisphere
	float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

	// Section 3.4: transforming the normal back to the ellipsoid configuration
	return normalize(float3(alpha2D.x * Nh.x, alpha2D.y * Nh.y, max(0.0f, Nh.z)));
}
// This should have D_GGX canceled out
float SampleGGXVNDFPdf(float alpha, float alphaSq, float NoH, float NoV, float LoH) {
	NoH = max(EPS, NoH);
	NoV = max(EPS, NoV);
	return (D_GGX(max(EPS, alphaSq), NoH) * G1_SmithGGX(alpha, alphaSq, NoV * NoV)) / (4.0f * NoV);
}
float SampleGGXVNDFWeight(float alpha, float alphaSq, float NoL, float NoV) {
    return G2_Over_G1_SmithHeightCorrelated(alpha, alphaSq, NoL, NoV);
}
```

Diffuse/Specular选择来自Ray Tracing Gems 2,用fresnel做通过概率选择

```glsl
// Specular over specular+diffuse
float EvalSpecularBRDFProbability(float3 baseColor, float metallic, float3 v, float3 n)
{
    float VoN = saturate(dot(v, n)); // Half-vector not known
    float Fspec = dot(LUMINANCE_RGB, F_Schlick(VoN, SpecularF0(baseColor, metallic)));
    float Fdiff = dot(LUMINANCE_RGB, DiffuseReflectance(baseColor, metallic)) * (1 - Fspec);
    return clamp(Fspec / max(EPS, Fspec + Fdiff), 0.1f, 0.9f);
}
```

最后RayGen中集成如下。附效果图：

```glsl
    float PspecIndirect = EvalSpecularBRDFProbability(baseColor, metallic, v, n);
    float alpha = roughness * roughness;
    if (rng.sample() < PspecIndirect){
        // Specular
        float3 hLocal;
        if (alpha <= EPS)
            hLocal = float3(0,0,1); // Perfect reflector
        else
            hLocal = SampleGGXVNDF(vLocal, float2(alpha), u);
        float3 lLocal = reflect(-vLocal, hLocal);
        float HoL = clamp(dot(hLocal, lLocal), EPS, 1.0f);
        float NoL = clamp(dot(nLocal, lLocal), EPS, 1.0f);
        float NoV = clamp(dot(nLocal, vLocal), EPS, 1.0f);
        float NoH = clamp(dot(nLocal, hLocal), EPS, 1.0f);
        float3 specularF0 = SpecularF0(baseColor, metallic);
        float3 F = F_Schlick(HoL, specularF0);
        rayLocal = lLocal;
        throughput *= F * SampleGGXVNDFWeight(alpha, alpha * alpha, NoL, NoV) / PspecIndirect;
    } else {
        // Diffuse
        rayLocal = SampleCosineHemisphere(u);
        // ^^ Samping/BRDF PDF (1/pi) cancels each other out vv
        throughput *= DiffuseReflectance(baseColor, metallic) / (1 - PspecIndirect);
        // Weighted with Specular by fresnel mix
        float3 hLocal = SampleGGXVNDF(vLocal, float2(alpha), u);
        float VoH = clamp(dot(vLocal, hLocal), EPS, 1.0f);
        float3 specularF0 = SpecularF0(baseColor, metallic);
        float3 F = F_Schlick(VoH, specularF0);
        throughput *= float3(1) - F;
    }
```



![image-20251217223535486](/image-foundation/image-20251217223535486.png)

##### TODO

值得注意的是反射面中的场景有变暗的情况，进行Furnace Test：

![image-20251218081811968](/image-foundation/image-20251218081811968.png)

这是之前用的single-scatter ggx模型的问题：高roughness下没有反射出来的光线会被视作“消失”，但现实中是能继续反弹的。

参考  [4.7.2 Energy loss in specular reflectance](https://google.github.io/filament/Filament.md.html#materialsystem/improvingthebrdfs/energylossinspecularreflectance) - 搞懂了再继续写。

#### 直接照明

![image-20251218093414866](/image-foundation/image-20251218093414866.png)

在之前的光栅路径的东西完全能够复用。图源Ray Tracing Gems 2；这里同样只有一个太阳光，集成如下：

```glsl
// Direct - we only have a sun light for now
{
    float3 l = -globalParams.sunDirection;
    if (!TraceShadowRay(p, l)){
        float3 h = normalize(v + l);
        float NoL = saturate(dot(n, l));
        float NoH = saturate(dot(n, h));
        float NoV = saturate(dot(n, v));
        float VoH = saturate(dot(v, h));
        float3 Fd = DiffuseReflectance(baseColor, metallic) / PI;
        float D = D_GGX(alpha * alpha, NoH);
        float V = V_SmithGGXCorrelated(NoV, NoL, roughness);
        float Fs = D * V;
        float3 metalBRDF = Fs * F_Schlick(VoH, baseColor);
        float3 dielectricBRDF = lerp(Fd, Fs, F_Schlick(VoH, float3(0.04)));
        float3 light = lerp(dielectricBRDF, metalBRDF, metallic);
        radiance += throughput * light * float3(globalParams.sunIntensity) * NoL;
    }
}
```

在 Intel Sponza 效果如下；存在Firefly需要处理。

![image-20251218121422042](/image-foundation/image-20251218121422042.png)

<h1 style="color:red">--- 施工中 ---</h1>