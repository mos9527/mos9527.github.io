---
author: mos9527
lastmod: 2025-12-07T18:04:12.895642
title: Foundation 施工笔记 【5】- 纹理与 GBuffer 存储
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

现在为止，Editor还没有任何关于光照，甚至是texture map的相关实现。本篇文章将介绍纹理方面的*高效*存储及Editor GBuffer的布局，及PBR shading的初步测试。

## 容器格式

不看轮子和各种私有格式的话，常用的主要有[DDS](https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide)和最近的[KTX2](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html)。

### KTX

虽然[工具链很成熟](https://github.com/KhronosGroup/KTX-Software)，在新版[Vulkan Tutorial](https://docs.vulkan.org/tutorial/latest/15_GLTF_KTX2_Migration.html#_understanding_ktx2)中被绝赞推荐，但由于全平台支持，直接使用的话由于各种编码产生的依赖会多的离谱（`--depth=1` clone大小约莫~1GB！），这里没有选择。

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

最后在GPU和文件本身存储上还是采用了BC7 - Editor内置了来自crunch作者[Richard Geldreich](https://richg42.blogspot.com/) 的 [bc7enc](https://github.com/richgel999/bc7enc) 实现。这样也方便直接读取JPG/PNG存储的glTF模型纹理。

集成上没有出现太多意外情况，这里就不多说了。和之前网格一样，优化（转码）/烘焙部分是可以离线的。以下是产生的DDS在[tacentview](https://github.com/bluescan/tacentview)预览效果：

![image-20251207180343608](/image-foundation/image-20251207180343608.png)

