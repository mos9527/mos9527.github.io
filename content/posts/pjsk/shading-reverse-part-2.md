---
author: mos9527
lastmod: 2025-01-14T01:08:34.159822
title: PSJK Blender卡通渲染管线重现【2】：角色 Shader
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

# 1. Eye-Highlight

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

  随时间对高光贴图进行**UV上偏移**后采样后根据**环境光强决定明亮**

# 2. Toon-V3

可以观察到其他Mesh都经历这个Pipeline；下面先从非`Face`部分的渲染开始

用到的Tex变化不大

![image-20250114213118832](/image-shading-reverse/image-20250114213118832.png)

Pixel Shader部分自成一体；毕竟PJSK的NPR做的也相对基础

人肉翻译（WIP）：

```glsl
#include <metal_stdlib>
#include <metal_texture>
using namespace metal;
constant uint32_t rp_output_remap_mask [[ function_constant(1) ]];
constant const uint rp_output_remap_0 = (rp_output_remap_mask >> 0) & 0xF;
constant const uint rp_output_remap_1 = (rp_output_remap_mask >> 4) & 0xF;
constant const uint rp_output_remap_2 = (rp_output_remap_mask >> 8) & 0xF;
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
    float u_xlat29;
    half lumaOffset;
    half charaSpecular3;
    mainTexSmp = _MainTex.sample(sampler_MainTex, input.TEXCOORD1.xy, bias(FGlobals._GlobalMipBias.xyxx.x));
    shadowTexSmp.xyz = _ShadowTex.sample(sampler_ShadowTex, input.TEXCOORD1.xy, bias(FGlobals._GlobalMipBias.xyxx.x)).xyz;
    valueTexSmp = _ValueTex.sample(sampler_ValueTex, input.TEXCOORD1.xy, bias(FGlobals._GlobalMipBias.xyxx.x));
    /* -- lerp shadow / main tex */
    shadowValue.xyz = (-float3(mainTexSmp.xyz)) + float3(shadowTexSmp.xyz);    
    shadowValue.xyz = fma(float3(UnityPerMaterial._ShadowTexWeight), shadowValue.xyz, float3(mainTexSmp.xyz)); 
    // shadowValue: (1-w)*main + w*shadow = lerp(main, shadow, _ShadowTexWeight)
    
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


    // -- rim light
    shadowValue.xyz = FGlobals._SekaiRimLightColorArray[charaId].www * FGlobals._SekaiRimLightColorArray[charaId].xyz;
    // TEXCOORD4 -> normalized view space position -> View Vector -> V
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

    // -- todo
    lumaOffset = half(-1.0) + FGlobals._SekaiRimLightShadowSharpnessArray[charaId];
    charaSpecular3 = half(1.0) + (-FGlobals._SekaiRimLightShadowSharpnessArray[charaId]);
    u_xlat29 = (-float(lumaOffset)) + float(charaSpecular3);
    u_xlat4.x = u_xlat20 + (-float(lumaOffset));
    u_xlat29 = float(1.0) / u_xlat29;
    u_xlat29 = u_xlat29 * u_xlat4.x;
    u_xlat29 = clamp(u_xlat29, 0.0f, 1.0f);
    u_xlat4.x = fma(u_xlat29, -2.0, 3.0);
    u_xlat29 = u_xlat29 * u_xlat29;
    u_xlat29 = u_xlat29 * u_xlat4.x;
    u_xlat5.xyz = fma(FGlobals._SekaiShadowRimLightColorArray[charaId].xyz, FGlobals._SekaiShadowRimLightColorArray[charaId].www, (-shadowValue.xyz));
    shadowValue.xyz = fma(float3(u_xlat29), u_xlat5.xyz, shadowValue.xyz);
    u_xlat28 = max(u_xlat28, 0.0);
    u_xlat28 = (-u_xlat28) + 1.0;
    lumaOffset = half(10.0) + (-FGlobals._SekaiRimLightFactor[charaId].x); // Phong?
    u_xlat28 = log2(u_xlat28);
    u_xlat28 = u_xlat28 * float(lumaOffset);
    u_xlat28 = exp2(u_xlat28);
    lumaValue.x = fma(u_xlat28, lumaValue.x, (-u_xlat28));
    u_xlat28 = fma(float(FGlobals._SekaiRimLightFactor[charaId].w), lumaValue.x, u_xlat28);
    lumaValue.x = (-u_xlat20) + 0.0500000007;
    charaId0 = int((0.0<lumaValue.x) ? 0xFFFFFFFFu : uint(0));
    u_xlati11 = int((lumaValue.x<0.0) ? 0xFFFFFFFFu : uint(0));
    u_xlati11 = (-charaId0) + u_xlati11;
    lumaValue.x = float(u_xlati11);
    lumaValue.x = fma(u_xlat28, lumaValue.x, (-u_xlat28));
    u_xlat28 = fma(float(FGlobals._SekaiRimLightFactor[charaId].w), lumaValue.x, u_xlat28);
    u_xlat28 = u_xlat28 + (-UnityPerMaterial._RimThreshold);
    lumaValue.x = float(1.0) / float(FGlobals._SekaiRimLightFactor[charaId].z);
    u_xlat28 = u_xlat28 * lumaValue.x;
    u_xlat28 = clamp(u_xlat28, 0.0f, 1.0f);
    lumaValue.x = fma(u_xlat28, -2.0, 3.0);
    u_xlat28 = u_xlat28 * u_xlat28;
    u_xlat28 = u_xlat28 * lumaValue.x;
    shadowValue.xyz = shadowValue.xyz * float3(u_xlat28);
    lumaValue.xyz = shadowValue.xyz * input.COLOR0.yyy;
    skinValue.xyz = half3(fma(shadowValue.xyz, input.COLOR0.yyy, float3(skinValue.xyz)));
    shadowValue.xyz = input.TEXCOORD4.xyz + FGlobals._SekaiDirectionalLight.xyz;
    u_xlat28 = dot(shadowValue.xyz, shadowValue.xyz);
    u_xlat28 = rsqrt(u_xlat28);
    shadowValue.xyz = float3(u_xlat28) * shadowValue.xyz;
    shadowValue.x = dot(shadowValue.xyz, input.NORMAL0.xyz);
    shadowValue.x = max(shadowValue.x, 0.0);
    shadowValue0.x = 10.0 / UnityPerMaterial._SpecularPower;
    shadowValue.x = log2(shadowValue.x);
    shadowValue.x = shadowValue.x * shadowValue0.x;
    shadowValue.x = exp2(shadowValue.x);
    shadowValue.xyz = float3(charaSpecular.xyz) * shadowValue.xxx;
    shadowValue.xyz = fma(shadowValue.xyz, float3(valueTexSmp.www), float3(skinValue.xyz));
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

