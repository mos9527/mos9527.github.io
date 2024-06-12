---
author: mos9527
lastmod: 2024-01-02T20:49:46.483516+08:00
title: Project SEKAI 逆向（6）：Live2D 资源
tags: ["逆向","unity","pjsk","api","project sekai","miku","unity","live2d"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Project SEKAI 逆向（6）：Live2D 资源

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

### 1. Live2D 模型

![image-20240102205059463](/assets/image-20240102205059463.png)

- 所有live2d资源都可以在 `[abcache]/live2d/`下找到；包括模型及动画

​	首先，`.moc3`,`.model3`,`.physics3`资源都可以直接利用[Live2D Cubism Editor](https://www.live2d.com/en/cubism/download/editor/)直接打开

​	而模型材质需要额外更名；这些信息都在`BuildModelData`中

![image-20240102205701929](/assets/image-20240102205701929.png)

- 补全后即可导入，效果如图

![image-20240102205542299](/assets/image-20240102205542299.png)

### 2. 动画 Key 预处理

- 可惜动画并不是`.motion3`格式

  封包中有的是Unity自己的Animation Clip

  在提取资源时，所有的动画key只能读到对应key string的CRC32 hash；导出/操作必须知道string-hash关系

![image-20240102210045486](/assets/image-20240102210045486.png)

- 这些string在`moc3`以外的文件中未知：当然，碰撞出string也不现实；猜想string和Live2D参数有关

![image-20240102210113370](/assets/image-20240102210113370.png)

​	尝试搜索无果

![image-20240102210134670](/assets/image-20240102210134670.png)

- 幸运的是Live2D Unity SDK可以免费取得，而且附带样例

  还记得前文处理BlendShape时，可以知道`AnimationClip`的源`.anim`会有path的源string，而不是crc

![image-20240102210341040](/assets/image-20240102210341040.png)

尝试加入前缀

![image-20240102210356955](/assets/image-20240102210356955.png)

![image-20240102210406498](/assets/image-20240102210406498.png)

可以定位；下面介绍如何构建CRC表，完成crc-string map

### 3. moc3 反序列化 + CRC打表

- 每次读取都从`moc3`文件构造应该可行；不过考虑到有导入纯动画的需求，显然一个常量map是需要的

- 故需要能读取`moc3`中所有参数名；参照https://raw.githubusercontent.com/OpenL2D/moc3ingbird/master/src/moc3.hexpat

  在 ImHex 中可见：

![image-20240103090132409](/assets/image-20240103090132409.png)

- 提取参数名脚本如下：

```python
from typing import BinaryIO
from struct import unpack

# https://github.com/OpenL2D/moc3ingbird/blob/master/src/moc3.hexpat
class moc3:    
    Parameters : list
    Parts: list
    def __init__(self, file : BinaryIO) -> None:        
        # Header: 64 bytes
        file.seek(0)
        assert file.read(4) == b'MOC3'
        version = unpack('<c',file.read(1))[0]
        isBigEndian = unpack('<b',file.read(1))[0]
        assert not isBigEndian
                
        # TODO: Other fields
        file.seek(0x40)
        pCountInfo = unpack('<I',file.read(4))[0]
        
        file.seek(pCountInfo)
        numParts = unpack('<I',file.read(4))[0]
        file.seek(0x10, 1)
        numParameters = unpack('<I',file.read(4))[0]

        file.seek(0x4C)
        pParts = unpack('<I',file.read(4))[0]

        file.seek(0x108)
        pParameters = unpack('<I',file.read(4))[0]
        
        def read_strings(offset, count):
            for i in range(0,count):
                file.seek(offset + i * 0x40)   
                buffer = bytearray()  
                while b := file.read(1)[0]:
                    buffer.append(b)
                yield buffer.decode(encoding='utf-8')
        
        self.Parts = list(read_strings(pParts,numParts))
        self.Parameters = list(read_strings(pParameters,numParameters))

```

- 之后，构造CRC表就很简单了

```python
from io import BytesIO
from sssekai.unity.AssetBundle import load_assetbundle
from sssekai.fmt.moc3 import moc3
import sys, os
from UnityPy.enums import ClassIDType

ParameterNames = set()
PartNames = set()
tree = os.walk(sys.argv[1])
for root, dirs, files in tree:
    for fname in files:
        file = os.path.join(root,fname)
        with open(file,'rb') as f:
            env = load_assetbundle(f)
            for obj in env.objects:
                if obj.type == ClassIDType.TextAsset:
                    data = obj.read()
                    out_name : str = data.name
                    if out_name.endswith('.moc3'):
                        moc = moc3(BytesIO(data.script.tobytes()))                        
                        for name in moc.Parameters:
                            ParameterNames.add(name)
                        for name in moc.Parts:
                            PartNames.add(name)                        
from zlib import crc32
print('NAMES_CRC_TBL = {')
for name in sorted(list(PartNames)):
    fullpath = 'Parts/' + name
    print('    %d:"%s",' % (crc32(fullpath.encode('utf-8')), fullpath))
for name in sorted(list(ParameterNames)):
    fullpath = 'Parameters/' + name
    print('    %d:"%s",' % (crc32(fullpath.encode('utf-8')), fullpath))    
print('}')
```

- 导出结果如下：

![image-20240102225301658](/assets/image-20240102225301658.png)

### 4. AnimationClip 转换

Live2D有自己私有的动画格式`motion3`，幸运的是[UnityLive2DExtractor](https://github.com/Perfare/UnityLive2DExtractor)已做了相当多的解析实现，可供参考

由于上文介绍的细节出入，对PJSK的转换并不能直接使用这个工具

索性在[sssekai](https://github.com/mos9527/sssekai)重现；细节**非常**繁琐，再次不多说；有兴趣的话还请参考源码

- 使用例：将转化所有找到的`AnimationClip`为`.motion3.json`

```bash
sssekai live2dextract c:\Users\mos9527\.sssekai\abcache\live2d\motion\21miku_motion_base .
```

- 效果如图

![sssekai-live2d-anim-import-demo](/assets/sssekai-live2d-anim-import-demo.gif)

***SEE YOU SPACE COWBOY...***

### References

https://github.com/AssetRipper/AssetRipper

https://github.com/OpenL2D/moc3ingbird

https://github.com/Perfare/UnityLive2DExtractor

