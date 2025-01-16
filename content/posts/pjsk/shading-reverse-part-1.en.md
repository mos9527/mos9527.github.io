---
author: mos9527
lastmod: 2025-01-14T01:08:34.159822
title: PSJK Blender Cartoon Render Pipeline Revisited [1]： Preparations
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

You can notice that the graphics in the pipeline are flipped up and down, [here it has something to do with Metal's NDC space](https://developer.apple.com/documentation/metal/using_a_render_pipeline_to_render_primitives?language=objc)

![img](/image-shading-reverse/viewports_gl_vk.png)

[Metal and Vulkan similar](https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/), positive $Y$-axis facing down after emulation

![Vertex function coordinate transformation](https://docs-assets.developer.apple.com/published/f630339b30/rendered2x-1582928923.png)

In terms of fixing this, it is possible to flip all the affine $Y$-axes in the Vertex shader (e.g., by directly transforming the $P$ of the $PVM$ matrix).

Unity doesn't do that, though, and adds a Flip pass before Present...

![image-20250113225640248](/image-shading-reverse/image-20250113225640248.png)

![image-20250113225737968](/image-shading-reverse/image-20250113225737968.png)

...Reverse UV reblit all over again; considering the amount of shader the user would otherwise need to modify by hand seems inexcusable?

### Render mode

Game still uses classic Forward Rendering, not the new [TBDR](https://developer.apple.com/documentation/metal/tailor_your_apps_for_apple_gpus_and_tile-based_deferred_rendering); looks like URP supports the latter?

![image-20250113231200281](/image-shading-reverse/image-20250113231200281.png)

But it's clear that what's being used here is actually an SRP

![image-20250113231746750](/image-shading-reverse/image-20250113231746750.png)

# 3. Shallow view reprocessing

### DoF

Above, you can see that the pipeline spits out 5 tex after processing the geometry; one image, two pairs of Depth-Stencils, and accidentally leaves a “Depth” and.... Brightness?

![image-20250113233550721](/image-shading-reverse/image-20250113233550721.png)

Here Depth is a depth buffer that uses only the $R$ channel; however, unlike the z-test buffer, the range of values in this buffer does not correspond to the NDC depth.

See the decompiler to see that this is a buffer for depth-of-field effects (note `_CoCParams`, CoC is [Circle Of Confusion](https://www.reedbeta.com/blog/circle-of-confusion-from-the-depth-buffer/)

The linear depth is applied to a simplified version (very similar to the above link) of the formula and put into the $[0,1]$ interval for storage

![image-20250113233911261](/image-shading-reverse/image-20250113233911261.png)

The material is integrated into the Alpha channel of `_ColorCocTex` at a later stage

![image-20250113234827783](/image-shading-reverse/image-20250113234827783.png)

Afterwards, after simple Mip downsampling, a blurred version of the image is quickly generated and superimposed on the previous CoC value to form a post-depth-of-field image.

(Curious why there's a useless `Fg` here... Will it be optimized away?)

![image-20250114000545488](/image-shading-reverse/image-20250114000545488.png)

Sampler is full of Linear/Nearest Mip Filter, figure omitted

![image-20250114000229101](/image-shading-reverse/image-20250114000229101.png)

...It's pretty simple and brutal.

### Bloom

Since the game isn't doing HDR rendering, Brightness tex comes in handy here!

Doing Box Blur after post-processing and final compositing

![image-20250114001307424](/image-shading-reverse/image-20250114001307424.png)

![image-20250114002138546](/image-shading-reverse/image-20250114002138546.png)

### Saturation?

In addition, the post-processing part of the LUT color gradation processing and some other parameters to control the effect of the

![image-20250114002353725](/image-shading-reverse/image-20250114002353725.png)

And a `_SatTex` after the same previous CoC downsampling, don't know what for, but...

![image-20250114002419564](/image-shading-reverse/image-20250114002419564.png)

Debugging Metal shaders in Xcode is fairly easy:

![image-20250114002903568](/image-shading-reverse/image-20250114002903568.png)

![A screenshot of the Reload Shaders button in the debug bar.](/image-shading-reverse/gputools-metal-debugger-se-reload.png)

Instant editorial response at the touch of a button; see `SatTex` in action!

Adjust the `_SatAlpha` value here, first at $0.5$:

![image-20250114004723856](/image-shading-reverse/image-20250114004723856.png)

At $1.0$

![image-20250114004749054](/image-shading-reverse/image-20250114004749054.png)

At $0.0$

![image-20250114004807677](/image-shading-reverse/image-20250114004807677.png)

Refer to the code is also easy to know that the image with this value in the blur after the formal framebuffer to see the Lerp; the smaller the value of the effect of the more 'clear'

The naming is a bit strange... Also the formal buffer has post-processing that the original blur didn't have, and the transition isn't natural

I can't guess what I'm trying to accomplish yet orz.

### SMAA

After the post-processing, do a SMAA and you're basically done.

![image-20250114005326735](/image-shading-reverse/image-20250114005326735.png)

And finally to the aforementioned Flip - rendering over

# Conclusion

First time writing a frame-by-frame analysis, and I have to say the workload is bigger than I thought it would be orz

In the future, I hope to fulfill a few of the flags [`sssekai_blender_io` set](https://github.com/mos9527/sssekai_blender_io?tab=readme-ov-file#todo) by the end of the series, there's a long way to go...

In addition to the following citation, here's a special thanks to UnityPy and its groups and a couple of friends from Q Groups/Discord who are not at liberty to be named for their help + corrections + resources +...

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
