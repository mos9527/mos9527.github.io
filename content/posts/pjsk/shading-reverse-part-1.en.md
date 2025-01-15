---
author: mos9527
lastmod: 2025-01-14T01:08:34.159822
title: PSJK Blender Cartoon Render Pipeline Revisited [1]: Preparations
tags: ["reverse engineering","Unity","PJSK","Project SEKAI","Blender","CG","3D","NPR","Python"]
categories: ["PJSK", "reverse engineering", "collection", "CG"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static
typora-copy-images-to: ../../../static/image-shading-reverse
---

# Preface

[Same as the last time I closed the book](https://mos9527.com/posts/pjsk/archive-20240105/#preface) It's kind of a follow up, just in time to (re)start with the part of figuring out the cartoon rendering... *After all, this aspect of the old work left far more problems than it solved ()

There's plenty of time before the rally, so let's see how much I can write

- Version: Japanese 5.0.0 *(Yakimori update)*
- Equipment: Mac Mini M4 (2024)

# 1. preparatory work

I've run captures with [RenderDoc](https://renderdoc.org/) on a live Android device (note: Pixel 2 XL) before... Usability is actually pretty good, except for the relatively weak device performance and slow capture.

But nowadays, with the Mac, it may be possible to avoid imposing such limitations.

I read the README for [GPTK](https://developer.apple.com/games/game-porting-toolkit/) when I was tinkering with it earlier, and noticed this rather interesting paragraph:

![image-20250113214124143](/image-shading-reverse/image-20250113214124143.png)

> ...and inserting the following environment variables to enable Metal debugging and processing of debug information: **MTL_CAPTURE_ENABLED=1 D3DM_DXIL_PROCESS_DEBUG_INFORMATION=1**

Change environment variables and you can debug... Much easier than capturing a third-party Game on Win; with the latter you learn all sorts of weird injections (?) with RenderDoc/PIX! --But, back to the point, does this variable work well outside of the `wine + D3DMetal` translation layer?

## Game preparation

Apple Silicon's Macs without exception *can* run iPhone/iPad OS apps natively -- zero CPU, zero GPU overhead!

But note the “can”, there aren't many apps on the AppStore that can be installed directly...

But fortunately there is also [PlayCover](https://github.com/PlayCover/PlayCover) which can load decrypted IPA directly; [the latter armconverter can be found by searching](https://armconverter.com/decryptedappstore/jp/%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88%E3%82%BB%E3%82%AB%E3%82%A4)

![image-20250113215418461](/image-shading-reverse/image-20250113215418461.png)

## Environment variables

I can't believe I'm trying to go straight to the settings with my muscle memory on Windows ==

But other than launching after calling variables from the shell there's actually something like this on the mac: [EnvPane](https://github.com/hschmidt/EnvPane)

Try turning on `MTL_DEBUG_LAYER`; here `MTL_HUD_ENABLED` just turns on Metal performance HUD globally.

That said, it can also be set up directly in Xcode, so this step can be skipped.

![image-20250113215800060](/image-shading-reverse/image-20250113215800060.png)

## Xcode configuration

Starting with empty projects

![image-20250113215829549](/image-shading-reverse/image-20250113215829549.png)

I don't know why `Debug > Attach To Process` never refreshes, so I just use `Debug > Debug Executable`.

![image-20250113220102952](/image-shading-reverse/image-20250113220102952.png)

Be careful not to select the app icon created by PlayCover when selecting the destination - Xcode will treat it as an iOS app resulting in no debugging Destination afterward!

Here you need to select the App's own binary `ProductName` in the following location

![image-20250113220404819](/image-shading-reverse/image-20250113220404819.png)

![image-20250113220412072](/image-shading-reverse/image-20250113220412072.png)

⌘⇧+G locates the file where it is located and selects it

![image-20250113220919904](/image-shading-reverse/image-20250113220919904.png)

`Options > GPU Frame Capture` to `Metal`; this can also be changed later by `Product > Scheme > Edit Scheme`.

![image-20250113221144724](/image-shading-reverse/image-20250113221144724.png)

Launching directly, you can observe that the game spits out some logs

![image-20250113221415658](/image-shading-reverse/image-20250113221415658.png)

Not useful for now; HUD indicators can be observed after startup

![image-20250113221734517](/image-shading-reverse/image-20250113221734517.png)

Enter 3D MV (脳內革命ガール - https://www.youtube.com/watch?v=ZKuk7PeBc0U,https://www.bilibili.com/video/BV1Xz4y147c1) after freeing FPS in the settings.

![image-20250113222636949](/image-shading-reverse/image-20250113222636949.png)

**Native 4K 120**... We are told at least two things here

1. 24 Mac Mini really smells good😋
2. As with most mobile games - the pipeline is rather basic; it feels like you're picking on a soft target ()

# 2. GPU capture!

Interestingly PJSK doesn't pause the rendering thread when pausing the 3D PV previews

Just tap the screen to come back to capture, press `Capture`.

**Note:** `Profile after Replay` can be turned off to speed up shader editing.

![image-20250113223356040](/image-shading-reverse/image-20250113223356040.png)

![image-20250113223746077](/image-shading-reverse/image-20250113223746077.png)

Pass is in full view... *¶¶It's quite a sight to behold. ¶¶*

## Some observations

### Present flip

You can notice that the graphics in the pipeline are flipped up and down, [here it has something to do with Metal's NDC space](https://developer.apple.com/documentation/metal/using_a_render_pipeline_to_render_primitives ?language=objc)

![img](/image-shading-reverse/viewports_gl_vk.png)

[Metal and Vulkan similar](https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/), positive $Y$-axis facing down after emulation

![Vertex function coordinate transformation](https://docs-assets.developer.apple.com/published/f630339b30/rendered2x-1582928923.png)

In terms of fixing this, it is possible to flip all the affine $Y$-axes in the Vertex shader (e.g., by directly transforming the $P$ of the $PVM$ matrix).

Unity doesn't do that, though, and adds a Flip pass before Present...

![image-20250113225640248](/image-shading-reverse/image-20250113225640248.png)

![image-20250113225737968](/image-shading-reverse/image-20250113225737968.png)

...Reverse UV reblit all over again; considering the amount of shader the user would otherwise need to modify by hand seems inexcusable?

### Render mode

Game使用的仍是经典的前向渲染 (Forward Rendering)，没用新鲜的[TBDR](https://developer.apple.com/documentation/metal/tailor_your_apps_for_apple_gpus_and_tile-based_deferred_rendering);貌似URP支持后者？

![image-20250113231200281](/image-shading-reverse/image-20250113231200281.png)

但很显然这里用的其实是SRP

![image-20250113231746750](/image-shading-reverse/image-20250113231746750.png)

# 3. 浅看后处理

### DoF

上图可见管线在处理完几何之后吐出了5个tex；一个图像，两个一对Depth-Stencil,还意外地留下了一个“Depth”和...Brightness？

![image-20250113233550721](/image-shading-reverse/image-20250113233550721.png)

这里Depth是个只使用了$R$通道的深度buffer；但不同于作z-test的buffer，这个buffer的取值范围并不对应NDC深度

参阅反编译易知这原来是做景深效果用的Buffer（注意`_CoCParams`,CoC即[Circle Of Confusion](https://www.reedbeta.com/blog/circle-of-confusion-from-the-depth-buffer/)）

线性深度套用简化版（和上述链接非常相似的）公式后放到$[0,1]$区间存储

![image-20250113233911261](/image-shading-reverse/image-20250113233911261.png)

材质在后期被整合到`_ColorCocTex`的Alpha通道

![image-20250113234827783](/image-shading-reverse/image-20250113234827783.png)

之后在简单Mip下采样以后快速产生模糊版图像并根据之前的CoC值叠加过去成景深后图像

（很好奇这里为什么会有个没用的`Fg`...会被优化掉吗？）

![image-20250114000545488](/image-shading-reverse/image-20250114000545488.png)

Sampler全是Linear/Nearest Mip Filter，图略

![image-20250114000229101](/image-shading-reverse/image-20250114000229101.png)

...相当简单粗暴

### Bloom

因为game没在做HDR渲染，Brightness tex在这里派上用场

做Box Blur后在后处理最后合成

![image-20250114001307424](/image-shading-reverse/image-20250114001307424.png)

![image-20250114002138546](/image-shading-reverse/image-20250114002138546.png)

### 饱和度？

此外，后处理部分还有LUT色阶处理与一些其他参数控制的效果

![image-20250114002353725](/image-shading-reverse/image-20250114002353725.png)

和一个同之前CoC下采样后的`_SatTex`,不知道干什么的，不过...

![image-20250114002419564](/image-shading-reverse/image-20250114002419564.png)

Xcode中调试Metal shader相当容易：

![image-20250114002903568](/image-shading-reverse/image-20250114002903568.png)

![A screenshot of the Reload Shaders button in the debug bar.](/image-shading-reverse/gputools-metal-debugger-se-reload.png)

一点即可即时获得编辑响应；来看看`SatTex`干的活

在这里调节`_SatAlpha`值，首先$0.5$时：

![image-20250114004723856](/image-shading-reverse/image-20250114004723856.png)

$1.0$时

![image-20250114004749054](/image-shading-reverse/image-20250114004749054.png)

$0.0$时

![image-20250114004807677](/image-shading-reverse/image-20250114004807677.png)

参阅代码也易知图像随这个值在模糊后与正式framebuffer见进行Lerp；值越小效果越’清晰‘

命名有点奇怪...同时正式buffer也有原来模糊过没有的后处理，过渡并不自然

具体实现什么目的暂时还猜不到orz

### SMAA

后处理完毕做一遍SMAA就基本完毕了

![image-20250114005326735](/image-shading-reverse/image-20250114005326735.png)

最后来到上文提及的Flip - 呈现完毕

# 结语

第一次写逐帧分析，不得不说工作量比自己想象的大orz

未来希望在系列结束时能完成[`sssekai_blender_io`立下的几个flag](https://github.com/mos9527/sssekai_blender_io?tab=readme-ov-file#todo)，路漫漫其修远兮...

除了以下citation，这里还要特地感谢UnityPy及其群组和几位来自Q群/Discord不方便透露名字的朋友的帮助+指正+资源+...

***SEE YOU SPACE COWBOY...***

# References

Real Time Rendering 4th Edition

https://mamoniem.com/behind-the-pretty-frames-detroit-become-human/

https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@12.0/manual/rendering/deferred-rendering-path.html

https://developer.apple.com/documentation/metal/tailor_your_apps_for_apple_gpus_and_tile-based_deferred_rendering?language=objc

https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/docs/techniques/depth-of-field.md

https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field

https://www.reedbeta.com/blog/circle-of-confusion-from-the-depth-buffer/

https://en.wikipedia.org/wiki/Circle_of_confusion#Determining_a_circle_of_confusion_diameter_from_the_object_field

https://developer.apple.com/documentation/accelerate/fma

https://developer.apple.com/documentation/xcode/inspecting-shaders

https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
