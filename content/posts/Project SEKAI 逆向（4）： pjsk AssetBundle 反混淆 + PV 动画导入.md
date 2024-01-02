---
author: mos9527
title: Project SEKAI 逆向（4）： pjsk AssetBundle 反混淆 + PV 动画导入
tags: ["逆向","unity","pjsk","api","project sekai","miku", "3d","cg","blender","unity"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ./..\..\static
---

# Project SEKAI 逆向（4）： pjsk AssetBundle 反混淆 + PV 动画导入

本来想放假再弄的，不过没忍住（啊？

毕竟有个命名+types很完备的database在手，分析起来也不会*那么*难办

### 1. 数据提取

pjsk资源采用热更新模式；本体运行时之外，还会有~~3~4G左右的资源~~ （不定量，见下一篇...）

*暂时*不考虑线上拉取（不过貌似也很好办？）这里先尝试从本机提取资源

![image-20231231183530650](/assets/image-20231231183530650.png)

就是这些啦

![image-20231231183558159](/assets/image-20231231183558159.png)

可惜文件并不是直接的Unity AssetBundle. 不过有些片段很眼熟...

加上在logcat里也经常看到对assetbundle的日志，考虑有可能ab文件有某种混淆

### 2. 加载流程分析

进dnSpy直接搜assetbundle，可以发现`Sekai`空间下有这些class

![image-20231231183823530](/assets/image-20231231183823530.png)

进ida看实现，可以很轻松的找到加载ab的嫌疑流程

![image-20231231183917530](/assets/image-20231231183917530.png)

![image-20231231183933304](/assets/image-20231231183933304.png)

最后直接调用了unity的`LoadFromStream`，那么加载流程解密一定就在这里标记的`Sekai.AssetBundleStream`衍生类了

![image-20231231184111015](/assets/image-20231231184111015.png)

看下我们最关心的`Read`

![image-20231231184246728](/assets/image-20231231184246728.png)

可以注意到

- 加载时根据 `_isInverted` flag 决定是否进行反混淆操作
- 如果有，则先跳过4bytes,之后5bytes按位取反
- 最后移交`InvertedBytesAB`继续处理
  - 注意到`n00`应为128，`v20`为读取offset
  - 这里考虑offset=0情况，那么仅前128字节需要处理

跟进`InvertedBytesAB`

![image-20231231184647711](/assets/image-20231231184647711.png)

**注：** 这里的`0xFF FF FF FF FF`，是5个*0xff（第一眼还真没看出来orz*

可见，这里即**跳过4by tes后，每 8bytes，取反前5bytes**

综上，解密流程分析完毕

附脚本

```python
import sys
import os
def decrypt(infile, outfile):
    with open(infile, 'rb') as fin:        
        magic = fin.read(4)    
        if magic == b'\x10\x00\x00\x00':
            with open(outfile,'wb') as fout:  
                for _ in range(0,128,8):
                    block = bytearray(fin.read(8))
                    for i in range(5):
                        block[i] = ~block[i] & 0xff
                    fout.write(block)
                while (block := fin.read(8)):
                    fout.write(block)    
        else:
            print('copy %s -> %s', infile, outfile)
            fin.seek(0)
            with open(outfile,'wb') as fout:  
                while (block := fin.read(8)):
                    fout.write(block)    

if len(sys.argv) == 1:
    print('usage: %s <in dir> <out dir>' % sys.argv[0])
else:
    for root, dirs, files in os.walk(sys.argv[1]):
        for fname in files:
            file = os.path.join(root,fname)
            if (os.path.isfile(file)):
                decrypt(file, os.path.join(sys.argv[2], fname))
```

### 3. 提取资源

![image-20231231192311049](/assets/image-20231231192311049.png)

文件处理完后，就可以靠https://github.com/Perfare/AssetStudio提取资源了

![image-20231231192416677](/assets/image-20231231192416677.png)

不过版本号在metadata里有引用；这里是`2020.3.21f1`

![image-20231231192541641](/assets/image-20231231192541641.png)

把apk里摸到的资源也拉进来；这里的资源没有加密

![image-20240101094346904](/assets/image-20240101094346904.png)

调整后就可以随意读取了

![image-20231231192616533](/assets/image-20231231192616533.png)

ok。全部提取吧

![image-20231231193948735](/assets/image-20231231193948735.png)

### 4. AssetBundleInfo?

解密的时候发现有这个东西...干嘛的？

在`Sekai_AssetBundleManager__LoadClientAssetBundleInfo`中：

![image-20231231194342801](/assets/image-20231231194342801.png)

用的是和API一样的密钥和封包手段，解开看看

**注：** 工具移步 https://github.com/mos9527/sssekai；内部解密流程在文章中都有描述

```bash
python -m sssekai apidecrypt .\AssetBundleInfo .\AssetBundleInfo.json
```

![image-20231231202455181](/assets/image-20231231202455181.png)

这里还有ab依赖（？）资源类型的信息

或许以后更新的时候可以拿来做diff？暂时不知道有什么用orz

### 5. 资源使用？to be continued

检查下模型列表，发现角色模型数出奇的少？

![image-20231231203837242](/assets/image-20231231203837242.png)

只有25时和另外两个角色？这里的资源是按需加载缓存的吗？

暂时不看；尝试把东西放进blender

![image-20231231204536443](/assets/image-20231231204536443.png)

 bind pose就有问题；看来AssetStudio不能很好地应付这类资源？可惜AS已经不再被维护了

试试看 https://github.com/AssetRipper/AssetRipper/ 吧

注意使用 https://nightly.link/AssetRipper/AssetRipper/workflows/publish/master/AssetRipper_win_arm64.zip alpha build，写本文时stable的0.3.4.0不支持stripped unity version

![image-20231231212152781](/assets/image-20231231212152781.png)

![image-20231231212236240](/assets/image-20231231212236240.png)

![image-20231231212822730](/assets/image-20231231212822730.png)

![image-20231231213007649](/assets/image-20231231213007649.png)

试试暂时先不处理不能解码的texture；diff在这

```c#
diff --git a/Source/AssetRipper.Export.UnityProjects/Textures/TextureConverter.cs b/Source/AssetRipper.Export.UnityProjects/Textures/TextureConverter.cs
index 4bdcb657..d9d941f8 100644
--- a/Source/AssetRipper.Export.UnityProjects/Textures/TextureConverter.cs
+++ b/Source/AssetRipper.Export.UnityProjects/Textures/TextureConverter.cs
@@ -196,213 +196,220 @@ namespace AssetRipper.Export.UnityProjects.Textures
 
 		private static bool TryDecodeTexture(TextureFormat textureFormat, int width, int height, ReadOnlySpan<byte> inputSpan, Span<byte> outputSpan)
 		{
-			switch (textureFormat)
+			try
 			{
-				//ASTC
-				case TextureFormat.ASTC_RGB_4x4:
-				case TextureFormat.ASTC_RGBA_4x4:
-					AstcDecoder.DecodeASTC(inputSpan, width, height, 4, 4, outputSpan);
-					return true;
-
-				case TextureFormat.ASTC_RGB_5x5:
-				case TextureFormat.ASTC_RGBA_5x5:
-					AstcDecoder.DecodeASTC(inputSpan, width, height, 5, 5, outputSpan);
-					return true;
-
-				case TextureFormat.ASTC_RGB_6x6:
-				case TextureFormat.ASTC_RGBA_6x6:
-					AstcDecoder.DecodeASTC(inputSpan, width, height, 6, 6, outputSpan);
-					return true;
-
-				case TextureFormat.ASTC_RGB_8x8:
-				case TextureFormat.ASTC_RGBA_8x8:
-					AstcDecoder.DecodeASTC(inputSpan, width, height, 8, 8, outputSpan);
-					return true;
-
-				case TextureFormat.ASTC_RGB_10x10:
-				case TextureFormat.ASTC_RGBA_10x10:
-					AstcDecoder.DecodeASTC(inputSpan, width, height, 10, 10, outputSpan);
-					return true;
-
-				case TextureFormat.ASTC_RGB_12x12:
-				case TextureFormat.ASTC_RGBA_12x12:
-					AstcDecoder.DecodeASTC(inputSpan, width, height, 12, 12, outputSpan);
-					return true;
-
-				//ATC
-				case TextureFormat.ATC_RGB4:
-					AtcDecoder.DecompressAtcRgb4(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.ATC_RGBA8:
-					AtcDecoder.DecompressAtcRgba8(inputSpan, width, height, outputSpan);
-					return true;
-
-				//BC
-				case TextureFormat.BC4:
-				case TextureFormat.BC5:
-				case TextureFormat.BC6H:
-				case TextureFormat.BC7:
-					return DecodeBC(inputSpan, textureFormat, width, height, outputSpan);
-
-				//DXT
-				case TextureFormat.DXT1:
-				case TextureFormat.DXT1Crunched:
-					DxtDecoder.DecompressDXT1(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.DXT3:
-					DxtDecoder.DecompressDXT3(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.DXT5:
-				case TextureFormat.DXT5Crunched:
-					DxtDecoder.DecompressDXT5(inputSpan, width, height, outputSpan);
-					return true;
-
-				//ETC
-				case TextureFormat.ETC_RGB4:
-				case TextureFormat.ETC_RGB4_3DS:
-				case TextureFormat.ETC_RGB4Crunched:
-					EtcDecoder.DecompressETC(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.EAC_R:
-					EtcDecoder.DecompressEACRUnsigned(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.EAC_R_SIGNED:
-					EtcDecoder.DecompressEACRSigned(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.EAC_RG:
-					EtcDecoder.DecompressEACRGUnsigned(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.EAC_RG_SIGNED:
-					EtcDecoder.DecompressEACRGSigned(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.ETC2_RGB:
-					EtcDecoder.DecompressETC2(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.ETC2_RGBA1:
-					EtcDecoder.DecompressETC2A1(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.ETC2_RGBA8:
-				case TextureFormat.ETC_RGBA8_3DS:
-				case TextureFormat.ETC2_RGBA8Crunched:
-					EtcDecoder.DecompressETC2A8(inputSpan, width, height, outputSpan);
-					return true;
-
-				//PVRTC
-				case TextureFormat.PVRTC_RGB2:
-				case TextureFormat.PVRTC_RGBA2:
-					PvrtcDecoder.DecompressPVRTC(inputSpan, width, height, true, outputSpan);
-					return true;
-
-				case TextureFormat.PVRTC_RGB4:
-				case TextureFormat.PVRTC_RGBA4:
-					PvrtcDecoder.DecompressPVRTC(inputSpan, width, height, false, outputSpan);
-					return true;
-
-				//YUY2
-				case TextureFormat.YUY2:
-					Yuy2Decoder.DecompressYUY2(inputSpan, width, height, outputSpan);
-					return true;
-
-				//RGB
-				case TextureFormat.Alpha8:
-					RgbConverter.Convert<ColorA<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.ARGB4444:
-					RgbConverter.Convert<ColorARGB16, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGBA4444:
-					RgbConverter.Convert<ColorRGBA16, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGB565:
-					RgbConverter.Convert<ColorRGB16, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.R8:
-					RgbConverter.Convert<ColorR<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RG16:
-					RgbConverter.Convert<ColorRG<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGB24:
-					RgbConverter.Convert<ColorRGB<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGBA32:
-					RgbConverter.Convert<ColorRGBA<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.ARGB32:
-					RgbConverter.Convert<ColorARGB32, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.BGRA32_14:
-				case TextureFormat.BGRA32_37:
-					inputSpan.CopyTo(outputSpan);
-					return true;
-
-				case TextureFormat.R16:
-					RgbConverter.Convert<ColorR<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RG32:
-					RgbConverter.Convert<ColorRG<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGB48:
-					RgbConverter.Convert<ColorRGB<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGBA64:
-					RgbConverter.Convert<ColorRGBA<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RHalf:
-					RgbConverter.Convert<ColorR<Half>, Half, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGHalf:
-					RgbConverter.Convert<ColorRG<Half>, Half, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGBAHalf:
-					RgbConverter.Convert<ColorRGBA<Half>, Half, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RFloat:
-					RgbConverter.Convert<ColorR<float>, float, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGFloat:
-					RgbConverter.Convert<ColorRG<float>, float, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGBAFloat:
-					RgbConverter.Convert<ColorRGBA<float>, float, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				case TextureFormat.RGB9e5Float:
-					RgbConverter.Convert<ColorRGB9e5, double, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
-					return true;
-
-				default:
-					Logger.Log(LogType.Error, LogCategory.Export, $"Unsupported texture format '{textureFormat}'");
-					return false;
-			}
+				switch (textureFormat)
+				{
+					//ASTC
+					case TextureFormat.ASTC_RGB_4x4:
+					case TextureFormat.ASTC_RGBA_4x4:
+						AstcDecoder.DecodeASTC(inputSpan, width, height, 4, 4, outputSpan);
+						return true;
+
+					case TextureFormat.ASTC_RGB_5x5:
+					case TextureFormat.ASTC_RGBA_5x5:
+						AstcDecoder.DecodeASTC(inputSpan, width, height, 5, 5, outputSpan);
+						return true;
+
+					case TextureFormat.ASTC_RGB_6x6:
+					case TextureFormat.ASTC_RGBA_6x6:
+						AstcDecoder.DecodeASTC(inputSpan, width, height, 6, 6, outputSpan);
+						return true;
+
+					case TextureFormat.ASTC_RGB_8x8:
+					case TextureFormat.ASTC_RGBA_8x8:
+						AstcDecoder.DecodeASTC(inputSpan, width, height, 8, 8, outputSpan);
+						return true;
+
+					case TextureFormat.ASTC_RGB_10x10:
+					case TextureFormat.ASTC_RGBA_10x10:
+						AstcDecoder.DecodeASTC(inputSpan, width, height, 10, 10, outputSpan);
+						return true;
+
+					case TextureFormat.ASTC_RGB_12x12:
+					case TextureFormat.ASTC_RGBA_12x12:
+						AstcDecoder.DecodeASTC(inputSpan, width, height, 12, 12, outputSpan);
+						return true;
+
+					//ATC
+					case TextureFormat.ATC_RGB4:
+						AtcDecoder.DecompressAtcRgb4(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.ATC_RGBA8:
+						AtcDecoder.DecompressAtcRgba8(inputSpan, width, height, outputSpan);
+						return true;
+
+					//BC
+					case TextureFormat.BC4:
+					case TextureFormat.BC5:
+					case TextureFormat.BC6H:
+					case TextureFormat.BC7:
+						return DecodeBC(inputSpan, textureFormat, width, height, outputSpan);
+
+					//DXT
+					case TextureFormat.DXT1:
+					case TextureFormat.DXT1Crunched:
+						DxtDecoder.DecompressDXT1(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.DXT3:
+						DxtDecoder.DecompressDXT3(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.DXT5:
+					case TextureFormat.DXT5Crunched:
+						DxtDecoder.DecompressDXT5(inputSpan, width, height, outputSpan);
+						return true;
+
+					//ETC
+					case TextureFormat.ETC_RGB4:
+					case TextureFormat.ETC_RGB4_3DS:
+					case TextureFormat.ETC_RGB4Crunched:
+						EtcDecoder.DecompressETC(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.EAC_R:
+						EtcDecoder.DecompressEACRUnsigned(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.EAC_R_SIGNED:
+						EtcDecoder.DecompressEACRSigned(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.EAC_RG:
+						EtcDecoder.DecompressEACRGUnsigned(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.EAC_RG_SIGNED:
+						EtcDecoder.DecompressEACRGSigned(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.ETC2_RGB:
+						EtcDecoder.DecompressETC2(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.ETC2_RGBA1:
+						EtcDecoder.DecompressETC2A1(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.ETC2_RGBA8:
+					case TextureFormat.ETC_RGBA8_3DS:
+					case TextureFormat.ETC2_RGBA8Crunched:
+						EtcDecoder.DecompressETC2A8(inputSpan, width, height, outputSpan);
+						return true;
+
+					//PVRTC
+					case TextureFormat.PVRTC_RGB2:
+					case TextureFormat.PVRTC_RGBA2:
+						PvrtcDecoder.DecompressPVRTC(inputSpan, width, height, true, outputSpan);
+						return true;
+
+					case TextureFormat.PVRTC_RGB4:
+					case TextureFormat.PVRTC_RGBA4:
+						PvrtcDecoder.DecompressPVRTC(inputSpan, width, height, false, outputSpan);
+						return true;
+
+					//YUY2
+					case TextureFormat.YUY2:
+						Yuy2Decoder.DecompressYUY2(inputSpan, width, height, outputSpan);
+						return true;
+
+					//RGB
+					case TextureFormat.Alpha8:
+						RgbConverter.Convert<ColorA<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.ARGB4444:
+						RgbConverter.Convert<ColorARGB16, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGBA4444:
+						RgbConverter.Convert<ColorRGBA16, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGB565:
+						RgbConverter.Convert<ColorRGB16, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.R8:
+						RgbConverter.Convert<ColorR<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RG16:
+						RgbConverter.Convert<ColorRG<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGB24:
+						RgbConverter.Convert<ColorRGB<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGBA32:
+						RgbConverter.Convert<ColorRGBA<byte>, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.ARGB32:
+						RgbConverter.Convert<ColorARGB32, byte, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.BGRA32_14:
+					case TextureFormat.BGRA32_37:
+						inputSpan.CopyTo(outputSpan);
+						return true;
+
+					case TextureFormat.R16:
+						RgbConverter.Convert<ColorR<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RG32:
+						RgbConverter.Convert<ColorRG<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGB48:
+						RgbConverter.Convert<ColorRGB<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGBA64:
+						RgbConverter.Convert<ColorRGBA<ushort>, ushort, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RHalf:
+						RgbConverter.Convert<ColorR<Half>, Half, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGHalf:
+						RgbConverter.Convert<ColorRG<Half>, Half, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGBAHalf:
+						RgbConverter.Convert<ColorRGBA<Half>, Half, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RFloat:
+						RgbConverter.Convert<ColorR<float>, float, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGFloat:
+						RgbConverter.Convert<ColorRG<float>, float, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGBAFloat:
+						RgbConverter.Convert<ColorRGBA<float>, float, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					case TextureFormat.RGB9e5Float:
+						RgbConverter.Convert<ColorRGB9e5, double, ColorBGRA32, byte>(inputSpan, width, height, outputSpan);
+						return true;
+
+					default:
+						Logger.Log(LogType.Error, LogCategory.Export, $"Unsupported texture format '{textureFormat}'");
+						return false;
+				}
+			} catch
+			{
+				Logger.Log(LogType.Error, LogCategory.Export, $"Failed to convert '{textureFormat}'");
+				return false;
+			}	
 		}
 
 		private static bool DecodeBC(ReadOnlySpan<byte> inputData, TextureFormat textureFormat, int width, int height, Span<byte> outputData)

```

 貌似转换失败的材质都被识别成`ATSC_6x6`格式，原因未知

不过能够转换的资源可以直接进Unity Editor

（此前我还没用过unity😰

先拉个动画试试？

![image-20240101141156185](/assets/image-20240101141156185.png)

这里用了白葱模；注意shader并没有被拉出来，暂时用standard替补

![image-20240101152353581](/assets/image-20240101152353581.png)

face/body mesh分开；需绑定face root bone(Neck)到body (Neck)

```c#
using UnityEngine;

public class BoneAttach : MonoBehaviour
{
    public GameObject src;

    public GameObject target;
    
    void Start()
    {
        Update();
    }
    void Update()
    {
        target.transform.position = src.transform.position;
        target.transform.rotation = src.transform.rotation;
        target.transform.localScale = src.transform.localScale;
    }
}
```

![image-20240101141256456](/assets/image-20240101141256456.png)

注意到blendshape/morph名字对不上

![image-20240101141815895](/assets/image-20240101141815895.png)

![image-20240101141909497](/assets/image-20240101141909497.png)

爬了下assetripper的issue：这里的数字是名称的crc32（见 https://github.com/AssetRipper/AssetRipper/issues/954）

![image-20240101142406334](/assets/image-20240101142406334.png)

![image-20240101142422934](/assets/image-20240101142422934.png)

确实可以一一对应!

拿blendshape名字做个map修复后，动画key就正常了

![image-20240101150057515](/assets/image-20240101150057515.png)

加上timeline后的播放效果

![Animation](/assets/Animation.gif)

nice.貌似没问题了

![image-20240101152745827](/assets/image-20240101152745827.png)

哦！原来还可以直接build出webgl版本！

*~~websekai指日可待~~*

不知道什么时候写之后的，暂时画几个饼：

- 资源导入blender + toon shader 复刻
- 资源导入 [Foundation](https://github.com/mos9527/Foundation/tree/master/Source) 
  - 不过到目前为止还没做动画系统...
- 脱离游戏自力解析+下载资源
  - 搞清楚为什么模型资源就这点（

~fin

...and merry 2024!