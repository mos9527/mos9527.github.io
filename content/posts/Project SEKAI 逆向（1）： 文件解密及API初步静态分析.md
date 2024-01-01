---
author: mos9527
title: Project SEKAI 逆向（1）： 文件解密及API初步静态分析
tags: ["逆向","pjsk","project sekai","miku"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ./..\..\static
---

# Project SEKAI 逆向（1）： 文件解密及API初步静态分析

# 样例

### 前言

~~作案动机~~起因还是想等放假再搞算法题和忙死人的[Foundation](https://github.com/mos9527/Foundation)，毕竟也要期末了嘛*（你也知道！*

不过不写反倒太闲了orz；于是乎重新装上project sekai开始摸鱼——

*~~但是谱面都没解锁就很难受~~*

除此之外，也早有日后提取游戏内资源给[Foundation](https://github.com/mos9527/Foundation)测试用的想法🤔

索性这几天就给它做个逆向吧...

**注：**本文为笔记性质，疏漏不可避免；如有疏忽，烦请指正！

### 1. 解密 global-metadata.dat

按SEGA惯性，**保护™**是一定要有的

提取apk文件后上https://github.com/Perfare/Il2CppDumper，很轻松就能发现*global-metadata.dat*...

![image-20231228181726715](/assets/image-20231228181726715.png)

*...确实被动过*

按[这篇文章](https://katyscode.wordpress.com/2021/02/23/il2cpp-finding-obfuscated-global-metadata/)思路，先调查`libunity.so`, 定位到`il2cpp_init`

![image-20231228182008675](/assets/image-20231228182008675.png)

中规中矩！`metadata`反混淆有很大可能就会在`libil2cpp.so`里了

ida加载后，碰运气试试看直接搜`global-metadata.dat`

![image-20231228182229242](/assets/image-20231228182229242.png)

有定位？跟进xref很快就能找到这里`MetadataLoader`的实现

*(注：部分变量已更名)*

```c++
_BYTE *__fastcall MetadataLoader::LoadMetadataFile(char *a1)
{
  unsigned __int64 v2; // x8
  char *v3; // x9
  __int64 v4; // x0
  _BYTE *v5; // x8
  unsigned __int64 v6; // x9
  const char *v7; // x0
  int v8; // w21
  int v9; // w20
  _BYTE *mapped_metadata; // x19
  __int64 v11; // x8
  __int64 v13; // [xsp+0h] [xbp-E0h] BYREF
  unsigned __int64 v14; // [xsp+8h] [xbp-D8h]
  char *v15; // [xsp+10h] [xbp-D0h]
  size_t len[2]; // [xsp+30h] [xbp-B0h]
  __int64 v17[2]; // [xsp+80h] [xbp-60h] BYREF
  char *v18; // [xsp+90h] [xbp-50h]
  char *v19; // [xsp+98h] [xbp-48h] BYREF
  __int64 v20; // [xsp+A0h] [xbp-40h]
  unsigned __int8 v21; // [xsp+A8h] [xbp-38h]
  _BYTE v22[15]; // [xsp+A9h] [xbp-37h] BYREF
  _BYTE *v23; // [xsp+B8h] [xbp-28h]

  sub_17A953C();
  v19 = "Metadata";
  v20 = 8LL;
  v2 = (unsigned __int64)(unsigned __int8)v13 >> 1;
  if ( (v13 & 1) != 0 )
    v3 = v15;
  else
    v3 = (char *)&v13 + 1;
  if ( (v13 & 1) != 0 )
    v2 = v14;
  v17[0] = (__int64)v3;
  v17[1] = v2;
  sub_173B820(v17, &v19);
  if ( (v13 & 1) != 0 )
    operator delete(v15);
  v4 = strlen(a1);
  if ( (v21 & 1) != 0 )
    v5 = v23;
  else
    v5 = v22;
  if ( (v21 & 1) != 0 )
    v6 = *(_QWORD *)&v22[7];
  else
    v6 = (unsigned __int64)v21 >> 1;
  v19 = a1;
  v20 = v4;
  v13 = (__int64)v5;
  v14 = v6;
  sub_173B820(&v13, &v19);
  if ( (v17[0] & 1) != 0 )
    v7 = v18;
  else
    v7 = (char *)v17 + 1;
  v8 = open(v7, 0);
  if ( v8 == -1 )
    goto LABEL_25;
  if ( fstat(v8, (struct stat *)&v13) == -1 )
  {
    close(v8);
    goto LABEL_25;
  }
  v9 = len[0];
  mapped_metadata = mmap(0LL, len[0], 3, 2, v8, 0LL);
  close(v8);
  if ( mapped_metadata == (_BYTE *)-1LL )
  {
LABEL_25:
    mapped_metadata = 0LL;
    goto UNENCRYPTED;
  }
  if ( v9 >= 1 )
  {
    v11 = 0LL;
    do
    {
      mapped_metadata[v11] ^= METADATA_KEY[v11 & 0x7F];
      ++v11;
    }
    while ( v9 != (_DWORD)v11 );
  }
UNENCRYPTED:
  if ( (v17[0] & 1) != 0 )
    operator delete(v18);
  if ( (v21 & 1) != 0 )
    operator delete(v23);
  return mapped_metadata;
}
```

对比`il2cpp`的[官方实现](https://github.com/dreamanlan/il2cpp_ref/blob/master/libil2cpp/vm/MetadataLoader.cpp)：

```c++
void* MetadataLoader::LoadMetadataFile(const char* fileName)
{
    std::string resourcesDirectory = utils::PathUtils::Combine(utils::Runtime::GetDataDir(), utils::StringView<char>("Metadata"));

    std::string resourceFilePath = utils::PathUtils::Combine(resourcesDirectory, utils::StringView<char>(fileName, strlen(fileName)));

    int error = 0;
    FileHandle* handle = File::Open(resourceFilePath, kFileModeOpen, kFileAccessRead, kFileShareRead, kFileOptionsNone, &error);
    if (error != 0)
        return NULL;

    void* fileBuffer = utils::MemoryMappedFile::Map(handle);

    File::Close(handle, &error);
    if (error != 0)
    {
        utils::MemoryMappedFile::Unmap(fileBuffer);
        fileBuffer = NULL;
        return NULL;
    }

    return fileBuffer;
}
```

**显然`mmap`之后多出来那一块xor就是*解密*过程.**..SEGA这有点太糊弄了

解密脚本如下

```python
key = bytearray([
  0xC3, 0x2B, 0x75, 0xB9, 0xAF, 0x84, 0x3C, 0x1F, 0x2E, 0xFB, 
  0xBF, 0x6C, 0x63, 0x19, 0x70, 0xE4, 0xF0, 0x92, 0xA3, 0x3E, 
  0xD1, 0x5C, 0x30, 0x0A, 0xCB, 0x9B, 0x04, 0xF8, 0x16, 0xC7, 
  0x91, 0x4A, 0x8D, 0xAE, 0xFA, 0xBA, 0x7E, 0x71, 0x65, 0x53, 
  0xAF, 0x98, 0x2E, 0xC2, 0xC0, 0xC6, 0xA3, 0x81, 0x74, 0xD4, 
  0xA3, 0x2C, 0x3F, 0xC2, 0x97, 0x66, 0xFB, 0x6B, 0xEE, 0x14, 
  0x80, 0x43, 0x09, 0x67, 0x69, 0x75, 0xDE, 0xB4, 0x1F, 0xB5, 
  0x65, 0x7E, 0x2D, 0x50, 0x8E, 0x38, 0x2E, 0x6D, 0x4A, 0x05, 
  0xF7, 0x82, 0x84, 0x41, 0x23, 0x64, 0x0A, 0xCB, 0x16, 0x93, 
  0xBE, 0x13, 0x83, 0x50, 0xD2, 0x6C, 0x8F, 0xC7, 0x58, 0x4A, 
  0xE7, 0xEE, 0x62, 0xBE, 0x6F, 0x25, 0xFE, 0xEF, 0x33, 0x5E, 
  0x38, 0x8D, 0x21, 0xE8, 0x1C, 0xFE, 0xBE, 0xC7, 0x43, 0x05, 
  0x6A, 0x13, 0x9D, 0x8B, 0xF6, 0x52, 0xFA, 0xDC
])
with open('global-metadata.dat','rb') as E:
    with open('global-metadata-decrypt.dat','wb') as D:
        data = bytearray(E.read())
        for i in range(0, len(data)):
            data[i] ^= key[i & 0x7f]
        D.write(data)
```

重新用https://github.com/Perfare/Il2CppDumper，这次dump过程就能进行下去了

![image-20231228183207342](/assets/image-20231228183207342.png)

**注：**考虑到 `ERROR: This file may be protected.` 由`JNI_OnLoad`符号触发并且dump成功，暂且忽略...

附[Il2CppDumper 检查流程](https://github.com/Perfare/Il2CppDumper/blob/master/Il2CppDumper/ExecutableFormats/Elf64.cs)：

```c#
  private bool CheckProtection()
        {
            try
            {
                //.init_proc
                if (dynamicSection.Any(x => x.d_tag == DT_INIT))
                {
                    Console.WriteLine("WARNING: find .init_proc");
                    return true;
                }
                //JNI_OnLoad
                ulong dynstrOffset = MapVATR(dynamicSection.First(x => x.d_tag == DT_STRTAB).d_un);
                foreach (var symbol in symbolTable)
                {
                    var name = ReadStringToNull(dynstrOffset + symbol.st_name);
                    switch (name)
                    {
                        case "JNI_OnLoad":
                            Console.WriteLine("WARNING: find JNI_OnLoad");
                            return true;
                    }
                }
                if (sectionTable != null && sectionTable.Any(x => x.sh_type == SHT_LOUSER))
                {
                    Console.WriteLine("WARNING: find SHT_LOUSER section");
                    return true;
                }
            }
            catch
            {
                // ignored
            }
            return false;
        }
```



### 2.提取 libil2cpp.so

我的主力机（Zenfone9）BL暂时解不了*（阿苏斯!!!!!!!)*，这里只能投靠模拟器解决

我用了https://github.com/MustardChef/WSABuilds，很方便地提供了集成Magisk的方案

加上相应模块，首先尝试用https://github.com/vfsfitvnm/frida-il2cpp-bridge提取.so，不过libil2cpp.so并不能被查到

*注：貌似是arm64转译的缘故：https://github.com/frida/frida/issues/2366*

https://github.com/Perfare/Zygisk-Il2CppDumper能用，但不能提取struct信息，对分析帮助不大。

不过https://github.com/Perfare/Il2CppDumper也可以读memory dump.这里记录下最后成功提取的方法

- DUMP 运行时 `libil2cpp.so`

  *注：不用gg也可行：memory map 可以直接从 /proc/[pid]/maps 里读到，之后可手动提取

  ![image-20231228201100934](/assets/image-20231228201100934.png)

- 到lib基址，直接DUMP

![image-20231228201144812](/assets/image-20231228201144812.png)

- 提取dump文件

  ![image-20231228201739702](/assets/image-20231228201739702.png)

  magic正确，继续...

- 重复之前步骤

  ![image-20231228201403332](/assets/image-20231228201403332.png)

可以发现这次没有`ERROR: This file may be protected.`

[修正以后拖进IDA](https://github.com/Perfare/Il2CppDumper/issues/685)

![image-20231229112048891](/assets/image-20231229112048891.png)

ok！睡个觉再看看分析完没有(

### 3. dump检查

![image-20231229105840353](/assets/image-20231229105840353.png)

不幸，.net部分被[beebyte](https://www.beebyte.co.uk/)干过

不过这个混淆器貌似只会更名方法/成员名，可见class部分名称仍然完好

![image-20231229110029895](/assets/image-20231229110029895.png)

信息还是很多的！开始看看API业务逻辑吧

### 4. API 解密，first look

这里抓包工具用的是[Reqable](https://reqable.com/zh-CN/),HttpCanary的续作；CA验证可以通过https://github.com/NVISOsecurity/MagiskTrustUserCerts轻松绕过，这里不多说了

![image-20231229113505997](/assets/image-20231229113505997.png)

果然报文不是人能看懂的orz

不过按日厂尿性，这里一般都只会加1~2层对称加密(i.e. AES)；同时请求头内也没有任何关于[密钥交换](https://zh.wikipedia.org/wiki/%E8%BF%AA%E8%8F%B2-%E8%B5%AB%E7%88%BE%E6%9B%BC%E5%AF%86%E9%91%B0%E4%BA%A4%E6%8F%9B)的信息。

假设很可能成立！接下来就可以找key/iv了。

![image-20231229140355912](/assets/image-20231229140355912.png)

![image-20231229140222478](/assets/image-20231229140222478.png)

![image-20231229140251468](/assets/image-20231229140251468.png)

几经搜索，发现`APIManager`的ctor有使用静态key/iv的嫌疑

接下来的工作就是提取这里的key了...静态分析估计会非常麻烦

动态手段提取这里的key/iv*应该*不会太难受？

但愿如此，不过WSA里上调试器死活打不上bp，等搞到真机再试吧（

to be continued...
