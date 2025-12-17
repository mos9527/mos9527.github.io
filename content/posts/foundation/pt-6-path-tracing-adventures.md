---
author: mos9527
lastmod: 2025-12-17T18:09:41.181717
title: Foundation 施工笔记 【6】- 路径追踪
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

Foundation现在(2025/12/16)也有了能用的RT相关API，Editor的GPUScene也有了BLAS上传/压缩(compact)与逐帧TLAS更新支持。到目前为止用rt做的只有inline query实现硬阴影——在做实时GI相关内容之前，不妨复习下采样/PBR/降噪相关知识——那就写个GPU Path Tracer吧？

PBRT/[Physically Based Rendering:From Theory To Implementation](https://pbr-book.org/)/[Kanition大佬v3翻译版](https://github.com/kanition/pbrtbook) 和手头的 RTR4/[Real-Time Rendering 4th Edition](https://www.realtimerendering.com/) （尤其是第九章）将是我们这里主要的信息来源。

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

仅实现该BRDF效果如下——注意到墙壁色彩在其他物体上的简洁影响。

![image-20251217180829346](/image-foundation/image-20251217180829346.png)

<h1 style="color:red">--- 施工中 ---</h1>