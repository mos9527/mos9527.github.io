---
author: mos9527
lastmod: 2025-12-06T21:12:12.154973
title: Foundation 施工笔记 【5】- 纹理与 GBuffer 存储
tags: ["CG","Vulkan","Foundation"]
categories: ["CG","Vulkan"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

现在为止，Editor还没有任何关于光照，甚至是texture map的相关实现。本篇文章将介绍纹理方面的*高效*存储及Editor GBuffer的布局，及PBR shading的初步测试。

## 纹理格式

运行时解码成全精度，原始的`R8G8B8A8UNORM`等格式存储诚然是...一种选择。不过记得之前和[某个Unity游戏](https://mos9527.com/posts/pjsk/archive-20240105/#3-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%85%A5)打交道见过其在资产里直接存了原始`RGB565`(RGB 16 bit)量化格式的纹理做[NPR 效果](https://mos9527.com/posts/pjsk/shading-reverse-part-2/#%E7%AE%A1%E7%BA%BF%E6%9D%90%E8%B4%A8)。但是至于*为什么*要用这种格式，搜了半天也没找到个结果（汗）

纹理压缩，可能除了UI带alpha材质的极端情况外，基本是必做的事情。GPU硬件解码其支持的格式开销是几乎不存在的——参考 [Unity](https://docs.unity3d.com/6000.4/Documentation/Manual/texture-choose-format-by-platform.html) 给出的材质格式推荐：

### DXTc/BCn/ETC

- **桌面端**，支持最广泛的包括各种BCn/Block Compression [n]压缩 - 参见 [Understanding BCn Texture Compression Formats - Nathan Reed](https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats)；同时，[BC1,BC2,BC3 也被微软叫成 DXT1, DXT3, DXT5](https://en.wikipedia.org/wiki/S3_Texture_Compression)，这里提及以防混淆。最后，在近十年支持DX11级别的桌面硬件中，无脑使用`BC6(H),BC7` 基本没问题。

- **移动端**，ETC曾是主流：Unity的[crunch](https://github.com/Unity-Technologies/crunch/tree/unity)就提供第一方DTXc/BCn和ETC的支持。现在的移动硬件一概支持更优秀且**开源**的ASTC：Unity自己也在使用[ARM官方的开源astc-codec](https://github.com/ARM-software/astc-encoder)处理编码。但需要注意的是，**桌面硬件上对astc的支持几乎不存在**。这点也可以自己利用Vulkan Caps Viewer检查：![image-20251206202629076](/image-foundation/image-20251206202629076.png)

### Basis Universal

可见硬件上——上述压缩编码在桌面/移动端的支持是没有交集的。但是纹理压缩做的工作本身会有很多重叠（如 DCT编码等等）—— [basisu](https://github.com/BinomialLLC/basis_universal) 即可高效地**存储一种统一**编码格式，并在运行时*非常快速*地转换到目标平台使用编码上传GPU。

额外的，basisu也自带一层[哈夫曼编码](https://github.com/BinomialLLC/basis_universal/wiki/.basis-file-format-overview) - 这是在所有block处理完之后全局进行的，因此整体压缩率在相同平均bpp下也会比单纯的BCn等编码一般地会更高。

## 容器格式

不看轮子和各种私有格式的话，常用的主要有[DDS](https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide)和最近的[KTX2](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html)。

### KTX

鉴于KTX2格式有官方的[KTX-Software](https://github.com/KhronosGroup/KTX-Software)和相关集成来完成读取、编解码等操作，以及在新版[Vulkan Tutorial](https://docs.vulkan.org/tutorial/latest/15_GLTF_KTX2_Migration.html#_understanding_ktx2)中被绝赞推荐，这里我们选择后者进行集成。依赖管理上我们继续使用CMake `FetchContent`完成。

```cmake
FetchContent_Declare(
    KTX-Software
    GIT_REPOSITORY https://github.com/KhronosGroup/KTX-Software
    GIT_TAG v4.4.2
    GIT_SHALLOW TRUE
)
...
set(KTX_FEATURE_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(KTX-Software)
...
target_link_libraries(
    Editor_Assets PUBLIC
    Foundation_Math
    Foundation_Core
    meshoptimizer
    ktx
)
```

