---
author: mos9527
lastmod: 2023-12-28T18:02:56.556911+08:00
title: Project SEKAI 逆向（1）： 文件解密及API初步静态分析
tags: ["逆向","pjsk","project sekai","miku"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Project SEKAI 逆向（1）： 文件解密及API初步静态分析

- 分析variant：ColorfulStage 2.4.1 （Google Play 美服）

### 1. 解密 metadata

- 利用 [Il2CppDumper](https://github.com/Perfare/Il2CppDumper) 对apk中提取出来的`global-metadata.data`和`il2cpp.so`直接分析，可见至少`metadata`有混淆

![image-20231228181726715](/assets/image-20231228181726715.png)

- IDA静态分析`libunity.so`, 定位到export的`il2cpp_init`；没有发现有关混淆的处理

![image-20231228182008675](/assets/image-20231228182008675.png)

- 考虑直接分析`il2cpp.so`，定位到`global-metadata.dat`有关流程

![image-20231228182229242](/assets/image-20231228182229242.png)

从这里的xref可以很轻松的摸到Il2Cpp的metadata加载流程

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

对比 Unity 的默认实现 （https://github.com/mos9527/il2cpp-27/blob/main/libil2cpp/vm/MetadataLoader.cpp)：

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

可见该伪代码块涉及到混淆流程：

```c++
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
```

最后，解密脚本如下：

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

处理后再次Dump, `metadata`已经能够顺利加载

![image-20231228183207342](/assets/image-20231228183207342.png)

**注：**由 `ERROR: This file may be protected.` ，后续还将对`il2cpp.so`继续处理；请看下文

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

#### 准备

手上没有Root的安卓实体机，这里就用WSA了

https://github.com/MustardChef/WSABuilds很方便地提供了集成Magisk的方案

- Dump 运行时 `libil2cpp.so`

  ![image-20231228201100934](/assets/image-20231228201100934.png)

  从 /proc/[pid]/maps 里读到基址后可借助 GameGuardian 提取

![image-20231228201144812](/assets/image-20231228201144812.png)

![image-20231228201739702](/assets/image-20231228201739702.png)

- 重复之前步骤

  ![image-20231228201403332](/assets/image-20231228201403332.png)

可以发现这次没有`ERROR: This file may be protected.`

- 修补后载入 IDA，[详见该 Issue](https://github.com/Perfare/Il2CppDumper/issues/685)

![image-20231229112048891](/assets/image-20231229112048891.png)

### 3. Dump的部分发现

这里的混淆器是[BeeByte](https://www.beebyte.co.uk/)；虽然只是更名混淆，但要拿这些符号分析业务逻辑的话还是很头疼

![image-20231229105840353](/assets/image-20231229105840353.png)

- 貌似Class部分名称仍然完好

![image-20231229110029895](/assets/image-20231229110029895.png)

### 4. API 解密，first look

抓包工具：[Reqable](https://reqable.com/zh-CN/),

CA验证可以通过https://github.com/NVISOsecurity/MagiskTrustUserCerts轻松绕过，这里不多说了

![image-20231229113505997](/assets/image-20231229113505997.png)

- 找到疑似解密业务的逻辑如下

![image-20231229140355912](/assets/image-20231229140355912.png)

![image-20231229140222478](/assets/image-20231229140222478.png)

![image-20231229140251468](/assets/image-20231229140251468.png)

Bingo! 

想在下一次用动态调试 (i.e. Frida) 拿Key/IV，还请见后文

***SEE YOU SPACE COWBOY...***

### References

https://katyscode.wordpress.com/2021/02/23/il2cpp-finding-obfuscated-global-metadata/

