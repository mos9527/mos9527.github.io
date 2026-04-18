---
author: mos9527
lastmod: 2025-04-10
title: PJSK Blender卡通渲染管线重现【2】- 角色及舞台 Shader
tags: ["逆向","Unity","PJSK","Project SEKAI","Blender","CG","3D","NPR","Python"]
categories: ["PJSK", "逆向", "合集", "CG"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static
typora-copy-images-to: ../../../static/image-shading-reverse
---

# Preface

Shader部分其实已经有不少现成工作，比如

- https://github.com/KH40-khoast40/Shadekai
- https://github.com/festivities/SekaiToon

至于为什么要自己重新造轮子...有时间。还要什么理由？除此之外（*敲黑板*）最后实现是要进 Blender 的嘛...

PV：[愛して愛して愛して](https://www.bilibili.com/video/BV1cP4y1P7TM/)

![image-20250114210635573](/image-shading-reverse/image-20250114210635573.png)

在Xcode调试Metal甚至有Render Graph方式呈现的资源依赖...库克太谢谢你了🥲

![image-20250114164630468](/image-shading-reverse/image-20250114164630468.png)

上一篇并没有往SRP后处理前的工作去看；这里直击`SRPBatcher`Pass

很巧的是恰好这个Pass处理的正包含角色部分，下面逐步分析

![image-20250114210417461](/image-shading-reverse/image-20250114210417461.png)

**注：** 后文向量都将以图示方向表示
<p style="width=100%;text-align:center">
<img style="background-color:white;margin:0 auto;display:block;" src="/image-shading-reverse/220px-Blinn_Vectors.svg.png">
</p>

## 1. Eye-Highlight

简明概要的效果 - 即给角色眼睛表现添加卡通风格高光

![image-20250114210950337](/image-shading-reverse/image-20250114210950337.png)

这部分由额外Mesh表达（注意缩写`ehl`），在Pass中作为第一个Mesh出现

![image-20250114211355105](/image-shading-reverse/image-20250114211355105.png)

回到 Metal 调试，观察反编译可知几个效果上的细节

- 高光‘闪烁’效果

  ```glsl
  u_xlat0.x = FGlobals._SekaiGlobalEyeTime * UnityPerMaterial._DistortionFPS;
  u_xlat0.x = floor(u_xlat0.x);
  u_xlat0.x = u_xlat0.x / UnityPerMaterial._DistortionFPS;
  u_xlat1.x = u_xlat0.x * UnityPerMaterial._DistortionScrollX;
  u_xlat1.y = u_xlat0.x * UnityPerMaterial._DistortionScrollY;
  u_xlat0.xy = fma((-u_xlat1.xy), float2(UnityPerMaterial._DistortionScrollSpeed), input.TEXCOORD5.xy);
  u_xlat16_0.xy = _DistortionTex.sample(sampler_DistortionTex, u_xlat0.xy).xy;
  u_xlat16_2.xy = fma(u_xlat16_0.xy, half2(2.0, 2.0), half2(-1.0, -1.0));
  u_xlat0.xy = float2(u_xlat16_2.xy) + float2(UnityPerMaterial._DistortionOffsetX, UnityPerMaterial._DistortionOffsetY);
  u_xlat1.x = UnityPerMaterial._DistortionIntensity * UnityPerMaterial._DistortionIntensityX;
  u_xlat1.y = UnityPerMaterial._DistortionIntensity * UnityPerMaterial._DistortionIntensityY;
  u_xlat0.xy = fma(u_xlat0.xy, u_xlat1.xy, input.TEXCOORD0.xy);
  u_xlat0.xyz = float3(_MainTex.sample(sampler_MainTex, u_xlat0.xy, bias(FGlobals._GlobalMipBias.xyxx.x)).xyz);
  ...
  u_xlati18 = UnityPerMaterial._CharacterId;
  u_xlat1.xyz = float3(u_xlat16_3.xyz) * FGlobals._SekaiCharacterAmbientLightColorArray[u_xlati18].xyz;
  u_xlat16_4.xyz = half3((-u_xlat0.xyz) + float3(1.0, 1.0, 1.0));
  u_xlat16_4.xyz = u_xlat16_4.xyz + u_xlat16_4.xyz;
  u_xlat5.xyz = float3(1.0, 1.0, 1.0) + (-FGlobals._SekaiCharacterAmbientLightColorArray[u_xlati18].xyz);
  u_xlat5.xyz = fma((-float3(u_xlat16_4.xyz)), u_xlat5.xyz, float3(1.0, 1.0, 1.0));
  u_xlat16_3.xyz = half3(fma((-float3(u_xlat16_3.xyz)), FGlobals._SekaiCharacterAmbientLightColorArray[u_xlati18].xyz, u_xlat5.xyz));
  u_xlat16_2.xyz = half3(fma(float3(u_xlat16_2.xyz), float3(u_xlat16_3.xyz), u_xlat1.xyz));
  u_xlat1.xyz = float3(u_xlat16_2.xyz) * float3(FGlobals._SekaiCharacterAmbientLightIntensityArray[u_xlati18]);
  ```

  随时间对高光贴图进行**UV上偏移**后采样后根据**环境光强决定明亮**，同时产生动态效果

### Blender 实现

- 效果随时间变化意味着需要某种方式访问动画当前帧；在`Timeline > Current Frame`右键可Copy Driver放在Shader里

  ![image-20250116151222754](/image-shading-reverse/image-20250116151222754.png)

- 化繁为简考虑直接通过取模决定UV偏移方向，这样可仅指定$U,V$方向大小产生效果

- Shader实现如下

  ![image-20250116154033340](/image-shading-reverse/image-20250116154033340.png)

- 效果如下
  <video autoplay style="width:100%" controls src="/image-github/403745829-8ecb0d54-b221-4118-acb1-740bfc3635f0.mov"/>

环境光影响暂不考虑，之后进行对应支持

## 2. Toon-V3

可以观察到其他Mesh都经历这个Pipeline；下面先从非`Face`部分的渲染开始

用到的Tex变化不大

![image-20250114213118832](/image-shading-reverse/image-20250114213118832.png)

做些标记后就可以开始实现了，直接从Shader反编译代码看起

*鉴于寄存器复用，变量名也是如此，阅读上带来不便还且谅解*

### 结构体
```glsl
#include <metal_stdlib>
#include <metal_texture>
using namespace metal;
constant uint32_t rp_output_remap_mask [[ function_constant(1) ]];
constant const uint rp_output_remap_0 = (rp_output_remap_mask >> 0) & 0xF;
constant const uint rp_output_remap_1 = (rp_output_remap_mask >> 4) & 0xF;
constant const uint rp_output_remap_2 = (rp_output_remap_mask >> 8) & 0xF;
#define EXIT(X) output.SV_Target0 = half4(half3(X),1.0); return output;
struct FGlobals_Type
{
    float2 _GlobalMipBias;
    float4 _ZBufferParams;
    float4 _SekaiDirectionalLight;
    half4 _SekaiShadowColor;
    half _SekaiShadowThreshold;
    half4 _SekaiCharacterSpecularColorArray[10];
    float4 _CoCParams;
    float _SekaiAllLightIntensity;
    float _SekaiCharacterAmbientLightIntensityArray[10];
    float4 _SekaiCharacterAmbientLightColorArray[10];
    float4 _SekaiRimLightArray[10];
    float4 _SekaiRimLightColorArray[10];
    float4 _SekaiShadowRimLightColorArray[10];
    half4 _SekaiRimLightFactor[10];
    half _SekaiRimLightShadowSharpnessArray[10];
    float3 _SekaiGlobalSpotLightPos;
    float4 _SekaiGlobalSpotLightColor;
    float _SekaiGlobalSpotLightRadiusNear;
    float _SekaiGlobalSpotLightRadiusFar;
    int _SekaiGlobalSpotLightEnabled;
    float4 _SekaiFogColor;
};

struct UnityPerMaterial_Type
{
    float4 _MainTex_ST;
    float4 _ShadowTex_ST;
    float4 _ValueTex_ST;
    float4 _FaceShadowTex_ST;
    float4 _LightMapTex_ST;
    float _BumpScale;
    float _RimThreshold;
    float _OutlineWidth;
    float _OutlineL;
    float _OutlineOffset;
    float _SpecularPower;
    float4 _DefaultSkinColor;
    float4 _Shadow1SkinColor;
    float4 _Shadow2SkinColor;
    half _HeadNormalBlend;
    half _EyelashTransparent;
    half _EyelashFaceCameraEdge1;
    half _EyelashFaceCameraEdge2;
    half _IsLeftEyeClose;
    half _IsRightEyeClose;
    int _CharacterReflectionOff;
    float3 _FaceFront;
    int _CharacterId;
    float3 _HeadPosition;
    float _RangeLimit;
    half4 _PartsAmbientColor;
    float4 _HeadDotDirectionalLightValues;
    float _ShadowTexWeight;
};
```
### 管线输出
```glsl
struct Mtl_FragmentIn
{
    float3 NORMAL0 [[ user(NORMAL0) ]] ;
    float4 COLOR0 [[ user(COLOR0) ]] ;
    float4 TEXCOORD0 [[ user(TEXCOORD0) ]] ;
    float4 TEXCOORD1 [[ user(TEXCOORD1) ]] ;
    float3 TEXCOORD3 [[ user(TEXCOORD3) ]] ;
    half TEXCOORD6 [[ user(TEXCOORD6) ]] ;
    float3 TEXCOORD4 [[ user(TEXCOORD4) ]] ;
    float TEXCOORD7 [[ user(TEXCOORD7) ]] ;
    float3 TEXCOORD5 [[ user(TEXCOORD5) ]] ;
};

struct Mtl_FragmentOut
{
    half4 SV_Target0 [[ color(rp_output_remap_0) ]];
    half4 SV_Target1 [[ color(rp_output_remap_1) ]];
    half4 SV_Target2 [[ color(rp_output_remap_2) ]];
};
```
可以看到3个MRT的使用；结合上文易知

- `SV_Target0`为主图像
- `SV_Target1`为景深用模糊圈深度
- `SV_Target2`为Bloom用高光部分

显然，后两者仅在后处理时派上用场——这些在上文也以概述完毕。

![image-20250115170422329](/image-shading-reverse/image-20250115170422329.png)

接下来的观察都即将围绕第一个RenderTarget进行

### 管线材质

```glsl
fragment Mtl_FragmentOut xlatMtlMain(
    constant FGlobals_Type& FGlobals [[ buffer(0) ]],
    constant UnityPerMaterial_Type& UnityPerMaterial [[ buffer(1) ]],
    sampler sampler_MainTex [[ sampler (0) ]],
    sampler sampler_ShadowTex [[ sampler (1) ]],
    sampler sampler_ValueTex [[ sampler (2) ]],
    texture2d<half, access::sample > _MainTex [[ texture(0) ]] ,
    texture2d<half, access::sample > _ShadowTex [[ texture(1) ]] ,
    texture2d<half, access::sample > _ValueTex [[ texture(2) ]] ,
    Mtl_FragmentIn input [[ stage_in ]])
{
    Mtl_FragmentOut output;
    half4 mainTexSmp;
    float3 shadowValue;
    half3 shadowTexSmp;
    bool u_xlatb1;
    float3 u_xlat2;
    int charaId;
    half3 charaSpecular;
    float4 u_xlat4;
    half4 valueTexSmp;
    bool4 u_xlatb4;
    float3 u_xlat5;
    half3 skinValue;
    float3 u_xlat7;
    half3 shadowValue6_8;
    float3 shadowValue0;
    float3 lumaValue;
    int u_xlati11;
    bool u_xlatb11;
    half3 skinValue2;
    float shadowValue9;
    float u_xlat20;
    int charaId0;
    float u_xlat28;
    bool u_xlatb28;
    float rimIntensity;
    half lumaOffset;
    half charaSpecular3;
```

3份材质除外，在Mesh中也有一层Color信息；他们的用途会在下文一一介绍

方便期间，这里作以下简称

- `MainTex`: "C Tex",$T_c$
- `ShadowTex`: "S Tex",$T_s$
- `Value Tex`: "H Tex",$T_h$

前缀对应材质名称后缀`_C,_S,_H`

![image-20250115170933736](/image-shading-reverse/image-20250115170933736.png)

Vertex Shader部分将不直接查看；输出`TEXCOORD_`部分将在接下来对PS的分析中解释

### 阈值阴影

```glsl
    mainTexSmp = _MainTex.sample(sampler_MainTex, input.TEXCOORD1.xy, bias(FGlobals._GlobalMipBias.xyxx.x));
    shadowTexSmp.xyz = _ShadowTex.sample(sampler_ShadowTex, input.TEXCOORD1.xy, bias(FGlobals._GlobalMipBias.xyxx.x)).xyz;
    valueTexSmp = _ValueTex.sample(sampler_ValueTex, input.TEXCOORD1.xy, bias(FGlobals._GlobalMipBias.xyxx.x));
    /* -- lerp shadow / main tex */
    shadowValue.xyz = (-float3(mainTexSmp.xyz)) + float3(shadowTexSmp.xyz);    
    shadowValue.xyz = fma(float3(UnityPerMaterial._ShadowTexWeight), shadowValue.xyz, float3(mainTexSmp.xyz)); 
    // shadowValue: (1-w)*main + w*shadow = lerp(main, shadow, _ShadowTexWeight)
```
注意到真正的阴影部分由`_ShadowTexWeight`混合$T_c, T_s$
```glsl
    /* -- threshold shadow */
    charaId = UnityPerMaterial._CharacterId;
    charaSpecular.xyz = FGlobals._SekaiCharacterSpecularColorArray[charaId].www * FGlobals._SekaiCharacterSpecularColorArray[charaId].xyz;
    lumaOffset = fma(valueTexSmp.z, half(2.0), half(-1.0));
    // -1,1
    // TEXCOORD3 -> WS normal
    lumaValue.x = dot(FGlobals._SekaiDirectionalLight.xyz, input.TEXCOORD3.xyz);
    // 0,1
    lumaValue.x = fma(lumaValue.x, 0.5, 0.5);    
    lumaValue.x = float(lumaOffset) + lumaValue.x;
    lumaValue.x = clamp(lumaValue.x, 0.0f, 1.0f);
    // threshold shadow
    // comp luma to threshold
    u_xlatb11 = lumaValue.x>=float(FGlobals._SekaiShadowThreshold); 
    lumaValue.x = (u_xlatb11) ? 0.0 : 1.0;
    shadowValue.xyz = fma(shadowValue.xyz, float3(FGlobals._SekaiShadowColor.xyz), (-float3(mainTexSmp.xyz)));
    shadowValue.xyz = fma(lumaValue.xxx, shadowValue.xyz, float3(mainTexSmp.xyz));
    // shadowValue = lerp(shadowValue * shadowColor, mainTexSmp, lumaValue)
    // XXX: this could simply be a conditonal add
```
之后由一次阈值化选择阴影/$Tc$完毕

#### Blender 实现

首先可以注意到的是——这里点指向灯有且仅有**一个**

意味着可以在整个场景中**复用**这一个指向灯进行光照与后边会用到的一些Trick；毕竟指向/平行光源即一个入射光向量

##### 阈值来源

除了直接计算$N \cdot L$以外，直接使用Diffuse BSDF将会是个更好的选择

- 可提供动态阴影
- 专用控件/Gizmo

Shader设置如下；Color为纯白，显然产生的明亮度也能量守恒

![image-20250116155618383](/image-shading-reverse/image-20250116155618383.png)

##### 材质混合

暂时借用管线里的权重和阴影色彩；考虑到管线中这些值一致，故作Driver到全局对象处理；操作上之前已经介绍过

![image-20250116161026429](/image-shading-reverse/image-20250116161026429.png)

效果如下

![image-20250116161642565](/image-shading-reverse/image-20250116161642565.png)

### 皮肤特化

```glsl
    // -- skin color when shadowed..are they trying to emulate SSS?    
    u_xlat28 = shadowValue.x * UnityPerMaterial._ShadowTexWeight;
    u_xlat28 = fma(lumaValue.x, u_xlat28, float(mainTexSmp.x));
    // u_xlat28: main.r + [lit?]luma * (shadowValue.r * _ShadowTexWeight[?])
    // shadowed skin color
    lumaValue.xyz = float3(FGlobals._SekaiShadowColor.xyz) * UnityPerMaterial._Shadow1SkinColor.xyz;
    u_xlat5.xyz = float3(FGlobals._SekaiShadowColor.xyz) * UnityPerMaterial._Shadow2SkinColor.xyz;
    lumaOffset = half(u_xlat28 + u_xlat28);
    // [0,0.5]->[0,1] clamp upper values? 
    skinValue.x = half(fma(u_xlat28, 2.0, -1.0));
    skinValue.x = clamp(skinValue.x, 0.0h, 1.0h);
    skinValue2.xyz = half3(fma((-UnityPerMaterial._Shadow1SkinColor.xyz), float3(FGlobals._SekaiShadowColor.xyz), UnityPerMaterial._DefaultSkinColor.xyz));
    skinValue.xyz = half3(fma(float3(skinValue.xxx), float3(skinValue2.xyz), lumaValue.xyz));
    // skinValue = lerp([shadowedSkin1]lumaValue, DefaultSkinColor, u_xlat28[0.5,1]->[0,1])
    lumaOffset = lumaOffset;
    lumaOffset = clamp(lumaOffset, 0.0h, 1.0h); // same as skinValue.x
    skinValue.xyz = half3(fma((-UnityPerMaterial._Shadow2SkinColor.xyz), float3(FGlobals._SekaiShadowColor.xyz), float3(skinValue.xyz)));
    skinValue.xyz = half3(fma(float3(lumaOffset), float3(skinValue.xyz), u_xlat5.xyz));
    // skinValue = lerp([shadowedSkin2]u_xlat5, skinValue, u_xlat28[0,0.5]->[0,1])
    // pick between shadowed skin color and plain shadowed color
    // XXX: would using condtionals be less expensive? or it's the compiler being silly again
    u_xlatb28 = valueTexSmp.x>=half(0.5);
    lumaOffset = (u_xlatb28) ? half(1.0) : half(0.0); // value.R over 0.5?
    skinValue.xyz = half3((-shadowValue.xyz) + float3(skinValue.xyz));
    skinValue.xyz = half3(fma(float3(lumaOffset), float3(skinValue.xyz), shadowValue.xyz));
    // skinValue = lerp([shadowed]skinValue, shadowValue, lumaOffset) -> over 0.5: skin region   
```
#### Blender 实现

$T_h.R$决定是否使用肤色；借用管线中的一些值：

![image-20250116162651274](/image-shading-reverse/image-20250116162651274.png)

遗憾的是测试中的$T_h.R$全为$0$，故无任何效果；$R$全为$1$的情况效果如下：

![image-20250116163847472](/image-shading-reverse/image-20250116163847472.png)

### 边缘高光

```glsl
  // -- rim light
    shadowValue.xyz = FGlobals._SekaiRimLightColorArray[charaId].www * FGlobals._SekaiRimLightColorArray[charaId].xyz;
    // TEXCOORD4 -> normalized view space position -> View Vector -> V
    // XXX: since it's interpolated again this needs to be renormalized. apparently they didn't bother...
    u_xlat28 = dot(input.NORMAL0.xyz, input.TEXCOORD4.xyz); // NdotV
    lumaValue.x = dot(input.TEXCOORD4.xyz, FGlobals._SekaiRimLightArray[charaId].xyz); // VdotL
    lumaValue.x = max(lumaValue.x, 0.0);
    u_xlat20 = dot(input.NORMAL0.xyz, input.NORMAL0.xyz);
    u_xlat20 = rsqrt(u_xlat20);
    u_xlat5.xyz = float3(u_xlat20) * input.NORMAL0.xyz; // renormalize normal
    u_xlat20 = dot(FGlobals._SekaiRimLightArray[charaId].xyz, FGlobals._SekaiRimLightArray[charaId].xyz);
    u_xlat20 = rsqrt(u_xlat20);
    u_xlat7.xyz = float3(u_xlat20) * FGlobals._SekaiRimLightArray[charaId].xyz; // normalize L
    u_xlat20 = dot(u_xlat5.xyz, u_xlat7.xyz); // NdotL
    // -- rim light color
    lumaOffset = half(-1.0) + FGlobals._SekaiRimLightShadowSharpnessArray[charaId]; // sharp - 1
    charaSpecular3 = half(1.0) + (-FGlobals._SekaiRimLightShadowSharpnessArray[charaId]); // 1 - sharp
    rimIntensity = (-float(lumaOffset)) + float(charaSpecular3); // (1 - sharp) - (sharp - 1)
    u_xlat4.x = u_xlat20 + (-float(lumaOffset)); // NdotL - (sharp - 1)
    rimIntensity = float(1.0) / rimIntensity;
    rimIntensity = rimIntensity * u_xlat4.x;
    rimIntensity = clamp(rimIntensity, 0.0f, 1.0f);
    u_xlat4.x = fma(rimIntensity, -2.0, 3.0);    
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml
    rimIntensity = rimIntensity * rimIntensity;
    rimIntensity = rimIntensity * u_xlat4.x;
    // rimIntensity = smoothstep(sharp - 1, 1 - sharp, NdotL)
    u_xlat5.xyz = fma(FGlobals._SekaiShadowRimLightColorArray[charaId].xyz, FGlobals._SekaiShadowRimLightColorArray[charaId].www, (-shadowValue.xyz));
    shadowValue.xyz = fma(float3(rimIntensity), u_xlat5.xyz, shadowValue.xyz); // rim light color
		// lerp(rim, rimShadow, rimIntensity)
    u_xlat28 = max(u_xlat28, 0.0); // NdotV [0,1]
    u_xlat28 = (-u_xlat28) + 1.0; // [1,0]
    lumaOffset = half(10.0) + (-FGlobals._SekaiRimLightFactor[charaId].x); // 10 - factor.x
    u_xlat28 = log2(u_xlat28);
    u_xlat28 = u_xlat28 * float(lumaOffset);
    u_xlat28 = exp2(u_xlat28);
    // (1-NdotV) ^ (10 - factor.x)
    lumaValue.x = fma(u_xlat28, lumaValue.x, (-u_xlat28)); // (VdotL - 1) * u_xlat28
    u_xlat28 = fma(float(FGlobals._SekaiRimLightFactor[charaId].w), lumaValue.x, u_xlat28);
    // += factor_w * (VdotL - 1) * u_xlat28
    // u_xlat28 = (1 + factor_w * (VdotL - 1)) * u_xlat28
    lumaValue.x = (-u_xlat20) + 0.0500000007;
    charaId0 = int((0.0<lumaValue.x) ? 0xFFFFFFFFu : uint(0)); // NdotL < EPS
    u_xlati11 = int((lumaValue.x<0.0) ? 0xFFFFFFFFu : uint(0)); // NdotL > EPS
    u_xlati11 = (-charaId0) + u_xlati11;// NdotL == 0 
    lumaValue.x = float(u_xlati11);
    lumaValue.x = fma(u_xlat28, lumaValue.x, (-u_xlat28)); // (NdotL - 1) * u_xlat28
    u_xlat28 = fma(float(FGlobals._SekaiRimLightFactor[charaId].w), lumaValue.x, u_xlat28);
    // u_xlat28 = (1 + factor_w * (NdotL - 1)) * u_xlat28
    u_xlat28 = u_xlat28 + (-UnityPerMaterial._RimThreshold);
    lumaValue.x = float(1.0) / float(FGlobals._SekaiRimLightFactor[charaId].z);
    u_xlat28 = u_xlat28 * lumaValue.x;
    u_xlat28 = clamp(u_xlat28, 0.0f, 1.0f);
    lumaValue.x = fma(u_xlat28, -2.0, 3.0);
    u_xlat28 = u_xlat28 * u_xlat28;
    u_xlat28 = u_xlat28 * lumaValue.x;
    // smoothstep(0, factor.z, u_xlat28 - RimThreshold)
    shadowValue.xyz = shadowValue.xyz * float3(u_xlat28);
    lumaValue.xyz = shadowValue.xyz * input.COLOR0.yyy; // Vertex.G: Rim Intensity
    skinValue.xyz = half3(fma(shadowValue.xyz, input.COLOR0.yyy, float3(skinValue.xyz)));
```
代码量有些多..不过写成算式就很清晰了
$$
a = smoothstep(sharpness - 1, 1 - sharpness, \hat{\mathbf{N}} \cdot \hat{\mathbf{L}}) \newline
b = (1 - \hat{\mathbf{N}} \cdot \hat{\mathbf{V}})^{10 - factor_x} \newline
c = (1 + factor_w * (\hat{\mathbf{V}} \cdot \hat{\mathbf{L}})) * b \newline
d = c * (\hat{\mathbf{N}} \cdot \hat{\mathbf{L}} > 0)\newline
e = smoothstep(0, factor_z, d - threshold) \newline
I_{rim} = e * Color_0.y
$$

- $a$被用于计算灯光颜色
  $$
  C = lerp(C_{rim},C_{rimShadow},a)
  $$

- $b$即为[菲涅耳项的估算形式](https://en.wikipedia.org/wiki/Schlick%27s_approximation)
  $$
  R(\theta) = R_0 + (1 - R_0)(1 - \cos \theta)^5
  $$
  考虑$R_0$接近$0$省略；单独输出效果如图

  ![](/image-shading-reverse/image-20250116185400012.png)

- $c$中继续加上视角相关贡献

- $d$则clip掉背光部分

- 最后直接做加法到之前的计算
  $$
  C_{frag} += I_{rim} * C
  $$

#### Blender 实现

##### 高光光源？

首先值得注意的是这种光源是**每个角色一个**，同时，仍然是以**平行光**的形式出现

维护一个光源即可；但是很显然由于光照公式非常不PBR无法直接用Specular BSDF = = 这里选择另辟蹊径

使用一个Empty对象取其指向向量$-E$作为我们的$L$；从上述式子知不必考虑衰减等等故足矣

![image-20250116180538901](/image-shading-reverse/image-20250116180538901.png)

记法上入射$L$这里记录为着色点到光源指向；故需取$-E$而非$E$

Shader中可这样实现

![image-20250116182025563](/image-shading-reverse/image-20250116182025563.png)

![image-20250116183544832](/image-shading-reverse/image-20250116183544832.png)

- 实现后效果如图

  - 捕捉

    ![image-20250116193114870](/image-shading-reverse/image-20250116193114870.png)

  - 复现

    ![image-20250116193212974](/image-shading-reverse/image-20250116193212974.png)

混合后如图

![image-20250116195212888](/image-shading-reverse/image-20250116195212888.png)

### 总体高光

```glsl
	// charaSpecular.xyz = FGlobals._SekaiCharacterSpecularColorArray[charaId].www * FGlobals._SekaiCharacterSpecularColorArray[charaId].xyz;    
// -- directional light specular
    shadowValue.xyz = input.TEXCOORD4.xyz + FGlobals._SekaiDirectionalLight.xyz; // V + L
    u_xlat28 = dot(shadowValue.xyz, shadowValue.xyz);
    u_xlat28 = rsqrt(u_xlat28);
    shadowValue.xyz = float3(u_xlat28) * shadowValue.xyz; // normalize V + L
    shadowValue.x = dot(shadowValue.xyz, input.NORMAL0.xyz);
    shadowValue.x = max(shadowValue.x, 0.0);
    shadowValue0.x = 10.0 / UnityPerMaterial._SpecularPower;
    shadowValue.x = log2(shadowValue.x);
    shadowValue.x = shadowValue.x * shadowValue0.x;
    shadowValue.x = exp2(shadowValue.x);
    // ((V+L)*N) ^ (10/SpecularPower)
    shadowValue.xyz = float3(charaSpecular.xyz) * shadowValue.xxx; // specular color
    shadowValue.xyz = fma(shadowValue.xyz, float3(valueTexSmp.www), float3(skinValue.xyz)); // value.a: specular intensity
```
这里看起来像是经典的[Blinn-Phong模型](https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model#High-Level_Shading_Language_code_sample)

省事起见...实现就用Specular BSDF替代了

注意$T_a$对高光的选择性

#### Blender 实现

对部分‘金属’物体有效，对比如图

- 开启
  ![image-20250117142149291](/image-shading-reverse/image-20250117142149291.png)
- 关闭

	​	![image-20250117142120315](/image-shading-reverse/image-20250117142120315.png)


### 环境光

```glsl
    // -- ambient lights
    u_xlatb4.xzw = (shadowValue.xyz>=float3(0.5, 0.5, 0.5));
    charaSpecular.x = (u_xlatb4.x) ? half(1.0) : half(0.0);
    charaSpecular.y = (u_xlatb4.z) ? half(1.0) : half(0.0);
    charaSpecular.z = (u_xlatb4.w) ? half(1.0) : half(0.0);
    skinValue.xyz = half3(shadowValue.xyz + shadowValue.xyz);
    u_xlat4.xzw = float3(skinValue.xyz) * FGlobals._SekaiCharacterAmbientLightColorArray[charaId].xyz;
    shadowValue6_8.xyz = half3((-shadowValue.xyz) + float3(1.0, 1.0, 1.0));
    shadowValue6_8.xyz = shadowValue6_8.xyz + shadowValue6_8.xyz;
    shadowValue.xyz = float3(1.0, 1.0, 1.0) + (-FGlobals._SekaiCharacterAmbientLightColorArray[charaId].xyz);
    shadowValue.xyz = fma((-float3(shadowValue6_8.xyz)), shadowValue.xyz, float3(1.0, 1.0, 1.0));
    skinValue.xyz = half3(fma((-float3(skinValue.xyz)), FGlobals._SekaiCharacterAmbientLightColorArray[charaId].xyz, shadowValue.xyz));
    charaSpecular.xyz = half3(fma(float3(charaSpecular.xyz), float3(skinValue.xyz), u_xlat4.xzw));
    shadowValue.xyz = float3(charaSpecular.xyz) * float3(FGlobals._SekaiCharacterAmbientLightIntensityArray[charaId]);
    u_xlat4.xzw = shadowValue.xyz * float3(FGlobals._SekaiAllLightIntensity);
    charaSpecular.xyz = half3(fma((-shadowValue.xyz), float3(FGlobals._SekaiAllLightIntensity), float3(1.0, 1.0, 1.0)));
    charaSpecular.xyz = charaSpecular.xyz + charaSpecular.xyz;
    skinValue.xyz = (-UnityPerMaterial._PartsAmbientColor.xyz) + half3(1.0, 1.0, 1.0);
    charaSpecular.xyz = fma((-charaSpecular.xyz), skinValue.xyz, half3(1.0, 1.0, 1.0));
    skinValue.xyz = half3(fma(u_xlat4.xzw, float3(UnityPerMaterial._PartsAmbientColor.xyz), (-float3(charaSpecular.xyz))));
    charaSpecular.xyz = fma(UnityPerMaterial._PartsAmbientColor.www, skinValue.xyz, charaSpecular.xyz);
```
##### Blender 实现

混合模式为直接乘法，细节暂略

效果如图

![image-20250116202717244](/image-shading-reverse/image-20250116202717244.png)

### 点光源支持

```glsl
    // -- add spot attenuation 
    u_xlatb1 = 0x0<FGlobals._SekaiGlobalSpotLightEnabled;
    if(u_xlatb1){
        shadowValue.xyz = (-input.TEXCOORD5.xyz) + FGlobals._SekaiGlobalSpotLightPos.xyzx.xyz;
        shadowValue.xy = shadowValue.xy * shadowValue.xy;
        shadowValue.x = shadowValue.y + shadowValue.x;
        shadowValue.x = fma(shadowValue.z, shadowValue.z, shadowValue.x);
        shadowValue0.x = (-FGlobals._SekaiGlobalSpotLightRadiusNear) + FGlobals._SekaiGlobalSpotLightRadiusFar;
        shadowValue.x = shadowValue.x + (-FGlobals._SekaiGlobalSpotLightRadiusNear);
        shadowValue0.x = float(1.0) / shadowValue0.x;
        shadowValue.x = shadowValue0.x * shadowValue.x;
        shadowValue.x = clamp(shadowValue.x, 0.0f, 1.0f);
        shadowValue0.x = fma(shadowValue.x, -2.0, 3.0);
        shadowValue.x = shadowValue.x * shadowValue.x;
        shadowValue9 = shadowValue.x * shadowValue0.x;
        shadowValue.x = fma((-shadowValue0.x), shadowValue.x, 1.0);
        shadowValue0.xyz = float3(charaSpecular.xyz) * float3(shadowValue9);
        shadowValue0.xyz = shadowValue0.xyz * FGlobals._SekaiGlobalSpotLightColor.xyz;
        shadowValue.xyz = fma(shadowValue.xxx, float3(charaSpecular.xyz), shadowValue0.xyz);
    } else {
        shadowValue.xyz = float3(charaSpecular.xyz);
    }
```

### 合并

```glsl
    u_xlat2.xyz = lumaValue.xyz * float3(FGlobals._SekaiRimLightFactor[charaId].yyy);
    u_xlat2.xyz = fma(shadowValue.xyz, float3(valueTexSmp.yyy), u_xlat2.xyz);
    charaSpecular.x = (-input.TEXCOORD6) + half(1.0);
    u_xlat28 = float(charaSpecular.x) * FGlobals._SekaiFogColor.w;
    u_xlat4.xyz = (-shadowValue.xyz) + FGlobals._SekaiFogColor.xyz;
    mainTexSmp.xyz = half3(fma(float3(u_xlat28), u_xlat4.xyz, shadowValue.xyz));
    shadowValue.x = input.TEXCOORD7 / input.TEXCOORD0.w;
    shadowValue.x = fma(FGlobals._ZBufferParams.z, shadowValue.x, FGlobals._ZBufferParams.w);
    shadowValue.x = fma((-FGlobals._CoCParams.x), shadowValue.x, 1.0);
    shadowValue.x = shadowValue.x * FGlobals._CoCParams.y;
    shadowValue0.x = shadowValue.x;
    shadowValue0.x = clamp(shadowValue0.x, 0.0f, 1.0f);
    shadowValue.x = max(shadowValue.x, -1.0);
    shadowValue.x = min(shadowValue.x, 0.0);
    shadowValue.x = shadowValue.x + shadowValue0.x;
    shadowValue.x = shadowValue.x + 1.0;
    shadowValue.x = shadowValue.x * 0.5;
    output.SV_Target0 = mainTexSmp;
    output.SV_Target1.x = half(shadowValue.x);
    output.SV_Target1.yzw = half3(0.0, 0.0, 1.0);
    output.SV_Target2.xyz = half3(u_xlat2.xyz);
    output.SV_Target2.w = half(1.0);
    return output;
}
```

### 最终效果

（等待配图ing）

## 3. Stage/LightMap

此部分相当简单 - 混合烘焙好的LightMap，环境光与Diffuse部分即完成

注意Vertex Color部分的使用也会造成影响

```glsl
    ...
    u_xlat16_0 = _MainTex.sample(sampler_MainTex, input.TEXCOORD1.xy);
    u_xlat16_1 = _LightMapTex.sample(sampler_LightMapTex, input.TEXCOORD2.xy);
    u_xlat0 = float4(u_xlat16_0) * input.COLOR0;
    u_xlat0 = u_xlat0 * float4(u_xlat16_1);
    u_xlat19 = u_xlat0.w + u_xlat0.w;
    u_xlatb2.xyz = (u_xlat0.xyz>=float3(0.25, 0.25, 0.25));
    u_xlat16_3.x = (u_xlatb2.x) ? half(1.0) : half(0.0);
    u_xlat16_3.y = (u_xlatb2.y) ? half(1.0) : half(0.0);
    u_xlat16_3.z = (u_xlatb2.z) ? half(1.0) : half(0.0);
    u_xlat16_4.xyz = half3(u_xlat0.xyz * float3(4.0, 4.0, 4.0));
    u_xlat2.xyz = float3(u_xlat16_4.xyz) * FGlobals._SekaiAmbientLightColor.xyz;
    u_xlat16_5.xyz = half3(fma((-u_xlat0.xyz), float3(2.0, 2.0, 2.0), float3(1.0, 1.0, 1.0)));
    u_xlat16_5.xyz = u_xlat16_5.xyz + u_xlat16_5.xyz;
    u_xlat0.xyz = (-FGlobals._SekaiAmbientLightColor.xyz) + float3(1.0, 1.0, 1.0);
    u_xlat0.xyz = fma((-float3(u_xlat16_5.xyz)), u_xlat0.xyz, float3(1.0, 1.0, 1.0));
    u_xlat16_4.xyz = half3(fma((-float3(u_xlat16_4.xyz)), FGlobals._SekaiAmbientLightColor.xyz, u_xlat0.xyz));
    u_xlat16_3.xyz = half3(fma(float3(u_xlat16_3.xyz), float3(u_xlat16_4.xyz), u_xlat2.xyz));
    u_xlat0.xyz = float3(u_xlat16_3.xyz) * float3(FGlobals._SekaiLightIntensity);
    u_xlat0.xyz = u_xlat0.xyz * float3(FGlobals._SekaiAllLightIntensity);
    u_xlatb18 = 0x0<FGlobals._SekaiGlobalSpotLightEnabled;
    ...
```

### 最终效果

（等待配图ing）

## 4. Toon (描边)

**注：** 强烈推荐阅读[5 ways to draw an outline](https://ameye.dev/notes/rendering-outlines/)作为前置知识！！

直击Vertex Shader，标注后如下

```glsl
  Mtl_VertexOut output;
  float4 position;
  float4 position2;
  float3 u_xlat2;
  position.xyz = input.POSITION0.yyy * UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[1].xyz;
  position.xyz = fma(UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[0].xyz, input.POSITION0.xxx, position.xyz);
  position.xyz = fma(UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[2].xyz, input.POSITION0.zzz, position.xyz);
  position2.xyz = position.xyz + UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[3].xyz;
  // World Space
  output.TEXCOORD2.xyz = fma(UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[3].xyz, input.POSITION0.www, position.xyz);
  position.xyz = position2.xyz + (-VGlobals._WorldSpaceCameraPos.xyzx.xyz);
  // View space
  position.x = dot(position.xyz, position.xyz);
  position.x = sqrt(position.x); // View Distance
  position.x = position.x + (-VGlobals._SekaiOutlineFactor.x);
  position.x = position.x * VGlobals._SekaiOutlineFactor.y;
  position.x = clamp(position.x, 0.0f, 1.0f);
  position.x = position.x * VGlobals._SekaiOutlineFactor.z;
  position.x = min(position.x, 1.0); // [0,1]
  u_xlat2.x = (-VGlobals._SekaiOutlineWidth.x) + VGlobals._SekaiOutlineWidth.y;
  position.x = fma(position.x, u_xlat2.x, VGlobals._SekaiOutlineWidth.x);
  // px = lerp(OutlineWidth.x, OutlineWidth.y, clamp((length(View) - Factor.x) * Factor.y,0,1) * Factor.z)
  u_xlat2.x = dot(input.NORMAL0.xyz, input.NORMAL0.xyz);
  u_xlat2.x = rsqrt(u_xlat2.x);
  u_xlat2.xyz = u_xlat2.xxx * input.NORMAL0.xyz;
  // N = normalize(N)
  position.xyz = position.xxx * u_xlat2.xyz;
  // Extrude by normal
  position.xyz = fma(position.xyz, input.COLOR0.xxx, input.POSITION0.xyz);
  // Done!
  position2.xyz = position.yyy * UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[1].xyz;
  position.xyw = fma(UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[0].xyz, position.xxx, position2.xyz);
  position.xyz = fma(UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[2].xyz, position.zzz, position.xyw);
  position.xyz = position.xyz + UnityPerDraw.hlslcc_mtx4x4unity_ObjectToWorld[3].xyz;
  position2 = position.yyyy * VGlobals.hlslcc_mtx4x4unity_MatrixVP[1];
  position2 = fma(VGlobals.hlslcc_mtx4x4unity_MatrixVP[0], position.xxxx, position2);
  position = fma(VGlobals.hlslcc_mtx4x4unity_MatrixVP[2], position.zzzz, position2);
  position = position + VGlobals.hlslcc_mtx4x4unity_MatrixVP[3];
  // Screen Space Position
  output.mtl_Position = position;
  // Screen Space Position
  output.TEXCOORD0 = position;
  output.COLOR0 = input.COLOR0;
  // Z depth (NDC)
  output.TEXCOORD4 = position.z;
  position.x = position.z / VGlobals._ProjectionParams.y;
  position.x = (-position.x) + 1.0;
  position.x = position.x * VGlobals._ProjectionParams.z;
  position.x = max(position.x, 0.0);
  position.x = fma((-VGlobals._SekaiFogFactor.xyxx.y), position.x, VGlobals._SekaiFogFactor.xyxx.x);
  position.x = clamp(position.x, 0.0f, 1.0f);
  output.TEXCOORD3 = half(position.x);
  output.TEXCOORD1.xy = fma(input.TEXCOORD0.xy, UnityPerMaterial._MainTex_ST.xy, UnityPerMaterial._MainTex_ST.zw);
  return output;
```

管线上将 `FrontFacingWinding` 改为 `CounterClockwise` 即翻转Vertex顺序，直接Cull掉几何原正面面向视角的部分

![image-20250120193720688](/image-shading-reverse/image-20250120193720688.png)

实现细节即一句话，在 World Space 直接做法线延伸，比较经典
$$
P_{shell} = P + N * VertexColor_x * lerp(OutlineWidth_x, OutlineWidth_y, clamp((length(View) - Factor_x) * Factor_y,0,1) * Factor_z)
$$

同时也有给美术微调描边强度的Color层，这点也是业界常规

![image-20250120194641606](/image-shading-reverse/image-20250120194641606.png)

（图源：https://cgworld.jp/article/202306-hifirush01.html）

### 最终效果

（等待配图ing）