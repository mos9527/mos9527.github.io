---
author: mos9527
lastmod: 2026-02-09T15:33:31.033000+08:00
title: PJSK Blender卡通渲染管线重现【1】- 预备工作
tags: ["逆向","Unity","PJSK","Project SEKAI","Blender","CG","3D","NPR","Python"]
categories: ["PJSK", "逆向", "合集", "CG"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static
typora-copy-images-to: ../../../static/image-shading-reverse
---

## Preface

[同上次封笔](https://mos9527.com/posts/pjsk/archive-20240105/#preface)算是承上启下了，刚好能从琢磨卡通渲染的部分（重新）入手...毕竟旧工作这方面留下的问题远远比解决的多

集训前时间也算充裕，看看能写多少吧

- 版本: 日服 5.0.0 *（烤森更新）*
- 设备：Mac Mini M4 (2024)

## 1. 预备工作

之前有在 Android 实机（注：Pixel 2 XL）上用 [RenderDoc](https://renderdoc.org/) 跑过捕捉...除了设备性能相对羸弱和捕捉死慢以外可用性其实不错

不过如今有了 Mac 这方面也许可以不用强加这种限制

先前捣鼓 [GPTK](https://developer.apple.com/games/game-porting-toolkit/) 的时候读过它的 README，注意到这段相当有趣：

![image-20250113214124143](/image-shading-reverse/image-20250113214124143.png)

> ...and inserting the following environment variables to enable Metal debugging and processing of debug information: **MTL_CAPTURE_ENABLED=1 D3DM_DXIL_PROCESS_DEBUG_INFORMATION=1**

改个环境变量就能调试...比在 Win 上捕捉第三方 Game 得方便太多；应付后者用 RenderDoc/PIX 还学会了各种奇怪的注入方式（？）——不过，回到正题，这个变量放到`wine + D3DMetal`转译层以外好用吗？

### 游戏准备

Apple Silicon 的 Mac 无一例外都*可以*原生跑 iPhone/iPad OS 上的 App —— CPU，GPU零 overhead

不过注意“可以”，AppStore上并搜不到多少能直接装的应用...

但是万幸还有[PlayCover](https://github.com/PlayCover/PlayCover)可以直接装解密过的 IPA；[后者 armconverter 上一搜即得](https://armconverter.com/decryptedappstore/jp/%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88%E3%82%BB%E3%82%AB%E3%82%A4)

![image-20250113215418461](/image-shading-reverse/image-20250113215418461.png)

### 环境变量

凭 Windows 上的肌肉记忆竟然想直接去设置找 = =

不过除了从shell调变量后启动以外其实mac上还真有这样的东西：[EnvPane](https://github.com/hschmidt/EnvPane)

尝试开启`MTL_DEBUG_LAYER`；这里`MTL_HUD_ENABLED`只是全局开启 Metal 性能HUD

话说回来Xcode里也能直接设置，此步可以略过

![image-20250113215800060](/image-shading-reverse/image-20250113215800060.png)

### Xcode 配置

从空项目开始

![image-20250113215829549](/image-shading-reverse/image-20250113215829549.png)

不知道为什么`Debug > Attach To Process`一直不刷新，索性直接使用`Debug > Debug Executable`

![image-20250113220102952](/image-shading-reverse/image-20250113220102952.png)

注意选择目标时不要选PlayCover创建的App图标 - Xcode会把它当成iOS应用处理导致之后没有调试Destination

这里需要选择App自己的二进制`ProductName`，位置如下

![image-20250113220404819](/image-shading-reverse/image-20250113220404819.png)

![image-20250113220412072](/image-shading-reverse/image-20250113220412072.png)

⌘⇧+G定位到所在文件，选择即可

![image-20250113220919904](/image-shading-reverse/image-20250113220919904.png)

`Options > GPU Frame Capture`调成`Metal`；这里后面也可以通过`Product > Scheme > Edit Scheme`修改

![image-20250113221144724](/image-shading-reverse/image-20250113221144724.png)

直接启动，可以观察到游戏吐了一些log

![image-20250113221415658](/image-shading-reverse/image-20250113221415658.png)

暂时没用；启动后可以观察到HUD指标

![image-20250113221734517](/image-shading-reverse/image-20250113221734517.png)

设置里解放FPS以后进入3D MV（脳内革命ガール - https://www.youtube.com/watch?v=ZKuk7PeBc0U,https://www.bilibili.com/video/BV1Xz4y147c1)

![image-20250113222636949](/image-shading-reverse/image-20250113222636949.png)

**原生4K120**...这里告诉我们至少两件事情

1. 24款Mac Mini真的很香😋
2. 同大多数移动game一样 - 管线比较基础；有种挑了软柿子捏的感觉（）

## 2. GPU捕捉！

有趣的是PJSK在暂停3D PV预览时并不会暂停渲染thread

点一下屏幕回来捕捉即可，按下`Capture`

**注：** `Profile after Replay` 可以考虑关掉加速shader编辑

![image-20250113223356040](/image-shading-reverse/image-20250113223356040.png)

![image-20250113223746077](/image-shading-reverse/image-20250113223746077.png)

Pass一览无余...*可以说是相当养眼*

### 一些观察

#### Present翻转

可以注意到管线中的图形是上下翻转的，[这里和Metal的NDC空间有关系](https://developer.apple.com/documentation/metal/using_a_render_pipeline_to_render_primitives?language=objc)

![img](/image-shading-reverse/viewports_gl_vk.png)

[Metal和Vulkan类似](https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/),仿射后的$Y$轴正方向朝下

![Vertex function coordinate transformation](https://docs-assets.developer.apple.com/published/f630339b30/rendered2x-1582928923.png)

修复而言在Vertex shader全部翻转仿射后的$Y$轴（比如直接对$PVM$矩阵之$P$变换）可行

不过Unity并没有这么做，而是在Present之前加了一层Flip pass...

![image-20250113225640248](/image-shading-reverse/image-20250113225640248.png)

![image-20250113225737968](/image-shading-reverse/image-20250113225737968.png)

...倒着UV重新blit了一遍；考虑否则用户需要手写修改的shader量貌似也无可厚非？

#### 渲染模式

[URP](https://media.colorfulpalette.co.jp/n/n51bf8818b89d).

![image-20250113231200281](/image-shading-reverse/image-20250113231200281.png)

![image-20250113231746750](/image-shading-reverse/image-20250113231746750.png)

## 3. 浅看后处理

#### DoF

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

#### Bloom

因为game没在做HDR渲染，Brightness tex在这里派上用场

做Box Blur后在后处理最后合成

![image-20250114001307424](/image-shading-reverse/image-20250114001307424.png)

![image-20250114002138546](/image-shading-reverse/image-20250114002138546.png)

#### 饱和度？

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

#### SMAA

后处理完毕做一遍SMAA就基本完毕了

![image-20250114005326735](/image-shading-reverse/image-20250114005326735.png)

最后来到上文提及的Flip - 呈现完毕

## 结语

第一次写逐帧分析，不得不说工作量比自己想象的大orz

未来希望在系列结束时能完成[`sssekai_blender_io`立下的几个flag](https://github.com/mos9527/sssekai_blender_io?tab=readme-ov-file#todo)，路漫漫其修远兮...

除了以下citation，这里还要特地感谢UnityPy及其群组和几位来自Q群/Discord不方便透露名字的朋友的帮助+指正+资源+...

***SEE YOU SPACE COWBOY...***

## References

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
