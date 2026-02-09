---
author: mos9527
lastmod: 2026-02-09T17:02:53.429000+08:00
title: RenderDoc 抓帧 Steam 及带启动器游戏通解
tags: ["CG","RenderDoc"]
categories: ["CG","RenderDoc"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static/
---

## Preface

谁抓过谁知道。启动带 DRM 的游戏通常需要一个或多个启动器的支持（Steam，R* 等等）

这种情况下 RenderDoc 直接启动 EXE 本身只会拉起启动器，然后由常驻启动器本身启动游戏进程

反思RenderDoc帧本身需要在设备初始化*之前*就已经hook上——保证这一点的方法其实相当多：

![RenderDocをGoogleChromeで起動してWebXRのデバッグを行う - 夜風のMixedReality](/images-cg/20220901212235.png)从自带的全局`appinit/Global Hook`方法（上图），利用 [Steamless](https://github.com/atom0s/Steamless) 自制学习版绕过 Steam 到[魔改 RenderDoc hook 过程 + 子进程 Capture](https://zhuanlan.zhihu.com/p/1085018723)，殊途同归

不过代价有大有小（且通常不小...），碰上一些难缠的DRM也容易暴毙

但 PC 玩家的创造力是无穷的——接下来介绍的手段将适用于 Windows 平台一切存在第三方修改/MOD的游戏

### 利用现成 ASI Loader

相当多 PC游戏 MOD 社区的起点都是这里。是否记得曾今打过 MOD 的后缀经常是 `.asi`?

他们只是更换了后缀名的`.dll`。在此，如果你的目标游戏用了`.asi`后缀，或者类似DLL Loader本身，你可以直接（注：假设Game为64位）：

- 找到你 RenderDoc 安装目录下的 `renderdoc.dll`

  ![image-20260209161033955](/images-cg/image-20260209161033955.png)

- 和其他 ASI/DLL Mod 一样，复制到游戏目录并更名为`renderdoc.asi`

- 以大表哥2为例

  ![image-20260209161258413](/images-cg/image-20260209161258413.png)

- 正常启动游戏，左上角应时刻都显示着 RenderDoc Hud

  ![image-20260209161806595](/images-cg/image-20260209161806595.png)
  
- 此时可选的，你可以启动 RenderDoc，然后在 `File > Attach to Running Instance` 中找到，并同正常抓帧流程一样处理

- 不过即使不启动也是可以抓的：直接按下`F12`,你将在`%TEMP%/RenderDoc`中找到你的replay

  ![image-20260209162151000](/images-cg/image-20260209162151000.png)
  
  ![image-20260209162159976](/images-cg/image-20260209162159976.png)
  
- 照常打开 Replay 即可

  ![image-20260209162555278](/images-cg/image-20260209162555278.png)

### 手动注入 ASI Loader

有些时候可能社区并没有在MOD 注入方面做太多工作。如果没有类似的DLL Mod Loader，对没有刻意阻止注入的Game（如反作弊、反篡改），自己动手也值得一试

以极乐迪斯科为例（64位，Unity）。放到[DIE](https://github.com/horsicq/DIE-engine/releases)中看看`UnityPlayer.dll` Imports（注：Unity 游戏应该通用）

![image-20260209165205480](/images-cg/image-20260209165205480.png)

熟悉的`version.dll`。找到对应的[Ultimate-ASI-Loader](https://github.com/ThirteenAG/Ultimate-ASI-Loader/releases) DLL伪装（即同样地，64位`version.dll`)，同更名后的`renderdoc.dll -> renderdoc.asi`复制到game根目录

![image-20260209165233379](/images-cg/image-20260209165233379.png)

和之前过程一样操作即可。注意DLL替换若不成功可尝试更换其他伪装选项。其次，也值得怀疑目标game似乎否存在反篡改——届时自求多福...

![image-20260209165539938](/images-cg/image-20260209165539938.png)

- 注：如图为纹理流送Buffer，实现是[Amplify Texture 2](https://amplify.pt/unity/amplify-texture-2/)。设定集中也有提及：

  ![image-20260209170031751](/images-cg/image-20260209170031751.png)

## Tips

- 注入成功但Capture/Replay掉设备可以考虑降低分辨率
- Die分析完别忘了关掉= =||
