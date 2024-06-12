---
author: mos9527
lastmod: 2023-12-31T18:28:57.301186+08:00
title: Project SEKAI 逆向（4）： pjsk AssetBundle 反混淆 + PV 动画导入
tags: ["逆向","unity","pjsk","api","project sekai","miku", "3d","cg","blender","unity"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Project SEKAI 逆向（4）： pjsk AssetBundle 反混淆 + PV 动画导入

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

### 1. 数据提取

pjsk资源采用热更新模式；本体运行时之外，还会有~~3~4G左右的资源~~ （**注：**不定量，见下一篇）

- 尝试从本机提取资源

![image-20231231183530650](/assets/image-20231231183530650.png)

![image-20231231183558159](/assets/image-20231231183558159.png)

没有magic `UnityFS`,考虑ab文件有混淆

### 2. 加载流程分析

- 进dnSpy直接搜assetbundle找相关Class

![image-20231231183823530](/assets/image-20231231183823530.png)

- 进ida看impl，可以很轻松的找到加载ab的嫌疑流程

![image-20231231183917530](/assets/image-20231231183917530.png)

![image-20231231183933304](/assets/image-20231231183933304.png)

- 最后直接调用了unity的`LoadFromStream`，`Sekai.AssetBundleStream`实现了这样的Stream：

![image-20231231184111015](/assets/image-20231231184111015.png)

![image-20231231184246728](/assets/image-20231231184246728.png)

可以注意到

- 加载时根据 `_isInverted` flag 决定是否进行反混淆操作
- 如果有，则先跳过4bytes,之后5bytes按位取反
- 最后移交`InvertedBytesAB`继续处理
  - 注意到`n00`应为128，`v20`为读取offset

- 这里考虑offset=0情况，那么仅前128字节需要处理

跟进`InvertedBytesAB`

![image-20231231184647711](/assets/image-20231231184647711.png)

可见，这里即**跳过4bytes后，每 8bytes，取反前5bytes**

综上，解密流程分析完毕；附脚本：

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

- 文件处理完后，就可以靠https://github.com/Perfare/AssetStudio查看资源了：

![image-20231231192416677](/assets/image-20231231192416677.png)

- 不过版本号很好找，这里是`2020.3.21f1`：

![image-20231231192541641](/assets/image-20231231192541641.png)

- 加载可行，如图：

![image-20231231192616533](/assets/image-20231231192616533.png)

### 4. AssetBundleInfo?

在数据目录里发现了这个文件，同时在`Sekai_AssetBundleManager__LoadClientAssetBundleInfo`中：

![image-20231231194342801](/assets/image-20231231194342801.png)

用的是和API一样的密钥和封包手段，解开看看

**注：** 工具移步 https://github.com/mos9527/sssekai；内部解密流程在文章中都有描述

```bash
python -m sssekai apidecrypt .\AssetBundleInfo .\AssetBundleInfo.json
```

![image-20231231202455181](/assets/image-20231231202455181.png)

### 5. 资源使用？

- 角色模型数很少

![image-20231231203837242](/assets/image-20231231203837242.png)

- 猜测这里的资源被热加载；在blender直接看看已经有的mesh吧：

  bind pose有问题，修正FBX导出设置可以解决；不过暂且不往这个方向深究

![image-20231231204536443](/assets/image-20231231204536443.png)

- 同时也许可以试试导入 Unity？

https://github.com/AssetRipper/AssetRipper/ 可以做到这一点，尝试如下：

![image-20231231212152781](/assets/image-20231231212152781.png)

![image-20231231212236240](/assets/image-20231231212236240.png)

![image-20231231212822730](/assets/image-20231231212822730.png)

- 拖进 Editor

![image-20240101141156185](/assets/image-20240101141156185.png)

- 注意shader并没有被拉出来，暂时用standard替补

![image-20240101152353581](/assets/image-20240101152353581.png)

- face/body mesh分开；需绑定face root bone(Neck)到body (Neck)

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

- 注意到blendshape/morph名字对不上

![image-20240101141815895](/assets/image-20240101141815895.png)

![image-20240101141909497](/assets/image-20240101141909497.png)

爬了下issue：这里的数字是名称的crc32（见 https://github.com/AssetRipper/AssetRipper/issues/954）

![image-20240101142406334](/assets/image-20240101142406334.png)

![image-20240101142422934](/assets/image-20240101142422934.png)

- 拿blendshape名字做个map修复后，动画key正常

![image-20240101150057515](/assets/image-20240101150057515.png)

- 加上timeline后的播放效果

![Animation](/assets/Animation.gif)

不知道什么时候写之后的，暂时画几个饼：

- 资源导入Blender + toon shader 复刻
- 资源导入 [Foundation](https://github.com/mos9527/Foundation/tree/master/Source) 
- 脱离游戏解析+下载资源

***SEE YOU SPACE COWBOY...***

### References

https://github.com/AssetRipper/AssetRipper/

https://github.com/AssetRipper/AssetRipper/issues/954

https://github.com/mos9527/Foundation