---
author: mos9527
lastmod: 2025-01-13T21:39:54.798422
title: Project SEKAI Reverse: notes archive (20240105)
tags: ["reverse direction","Unity","UnityPy","Frida","PJSK","Project SEKAI","Blender","CG","3D","NPR","Python"]
categories: ["PJSK", "reverse direction", "collection"]
ShowToc: true
TocOpen: true
typora-root-url: ../../../static
---

# Preface

*Cutoff: 24/01/05 Revision: 24/09/26*

Actually, analytics didn't really take off until mostly after that...but couldn't really find the time to write a post orz

If you are interested in the latest knowledge or would like to participate in the research, please also refer to the following Repo and its Wiki:

- https://github.com/mos9527/sssekai
- https://github.com/mos9527/sssekai_blender_io

## Project SEKAI Reverse (1): File Decryption and Preliminary Static Analysis of APIs

- Analyzing Variant: ColorfulStage 2.4.1 (Google Play US)

### 1. decrypt metadata

- utilization [Il2CppDumper](https://github.com/Perfare/Il2CppDumper) To the apk extracted from the`global-metadata.data`and`il2cpp.so`Directly analyzed, it can be seen that at least`metadata`confusing

![image-20231228181726715](/image-archive-20240105/image-20231228181726715.png)

- IDA Static Analysis`libunity.so`, Locate the export`il2cpp_init`；No obfuscated treatments were identified

![image-20231228182008675](/image-archive-20240105/image-20231228182008675.png)

- Consider direct analysis`il2cpp.so`，locate`global-metadata.dat`Processes

![image-20231228182229242](/image-archive-20240105/image-20231228182229242.png)

The Il2Cpp metadata loading process can be easily touched from the xref here

*(Note: some variables have been renamed)*

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

Compare to Unity's default implementation （https://github.com/mos9527/il2cpp-27/blob/main/libil2cpp/vm/MetadataLoader.cpp)：

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

This pseudo-code block can be seen to involve an obfuscation process：

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

Finally, the decryption script is as follows:

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

Dump again after processing, `metadata` can be loaded without any problem.

![image-20231228183207342](/image-archive-20240105/image-20231228183207342.png)

**Note:** by `ERROR: This file may be protected.` , followed by continued processing of `il2cpp.so`; see below

Attachment [Il2CppDumper check flow](https://github.com/Perfare/Il2CppDumper/blob/master/Il2CppDumper/ExecutableFormats/Elf64.cs)：

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



### 2. Extract libil2cpp.so

#### Preparation

Don't have a Rooted Android physical machine on hand, so I'll use WSA here

https://github.com/MustardChef/WSABuilds很方便地提供了集成Magisk的方案

- Dump Runtime `libil2cpp.so`

  ![image-20231228201100934](image-archive-20240105/image-20231228201100934.png)

  After reading the base address from /proc/[pid]/maps, you can extract it with GameGuardian's

![image-20231228201144812](/image-archive-20240105/image-20231228201144812.png)

![image-20231228201739702](/image-archive-20240105/image-20231228201739702.png)

- Repeat the previous steps

  ![image-20231228201403332](/image-archive-20240105/image-20231228201403332.png)

It can be noticed that this time there is no `ERROR: This file may be protected.`

- Patch loaded into IDA，[see the Issue](https://github.com/Perfare/Il2CppDumper/issues/685)

![image-20231229112048891](/image-archive-20240105/image-20231229112048891.png)

### 3. Part of Dump's findings ###

The obfuscator here is[BeeByte](https://www.beebyte.co.uk/); it's just a name change obfuscation, but it's still a headache to analyze business logic with these symbols

![image-20231229105840353](/image-archive-20240105/image-20231229105840353.png)

- Looks like the Class part of the name is still intact.

![image-20231229110029895](/image-archive-20240105/image-20231229110029895.png)

### 4. API decryption, first look ###

Packet grabber: [Reqable](https://reqable.com/zh-CN/) ,

CA validation can be done at https://github.com/NVISOsecurity/MagiskTrustUserCerts Easy to bypass, not much to say here

![image-20231229113505997](/image-archive-20240105/image-20231229113505997.png)

- The logic to find the suspected decryption operation is as follows

![image-20231229140355912](/image-archive-20240105/image-20231229140355912.png)

![image-20231229140222478](/image-archive-20240105/image-20231229140222478.png)

![image-20231229140251468](/image-archive-20240105/image-20231229140251468.png)

Bingo! 

For the next time you want to use dynamic debugging (i.e. Frida) to get a Key/IV, see also later!

***SEE YOU SPACE COWBOY...***

### References

https://katyscode.wordpress.com/2021/02/23/il2cpp-finding-obfuscated-global-metadata/

## Project SEKAI Reverse (2): Frida Dynamic Debugging and Fetching API AES KEY/IV

- Analyzing Variant: World Plan 2.6.1 (Google Play Taiwan)

### 0. Version update

After all, the US database lags by several version numbers...

The version number (2.4.1->2.6.1) has changed after switching to Taiwan, and the `metadata` encryption method mentioned in the previous post has also changed.

- Again, pulling a copy of the repaired dump and analyzing it reveals:

![image-20231230182936676](/image-archive-20240105/image-20231230182936676.png)

- The `metadata` load section is no longer globally encrypted and decrypted, as can also be found by observing the binary:

![image-20231230183059925](/image-archive-20240105/image-20231230183059925.png)

2.4.0 Decrypted `metadata`

![image-20231230183133793](/image-archive-20240105/image-20231230183133793.png)

2.6.1 Extra 8 bytes in source `metadata`?

![image-20231230183234026](/image-archive-20240105/image-20231230183234026.png)

![image-20231230183305449](/image-archive-20240105/image-20231230183305449.png)

- The loading part of the discrepancy is not big, try to delete 8 bytes to analyze directly

![image-20231230183414173](/image-archive-20240105/image-20231230183414173.png)

- Stuck on reading string; go to `Il2CppGlobalMetadataHeader.stringOffset` to check it out

![image-20231230183456731](/image-archive-20240105/image-20231230183456731.png)

- Sure enough there's confusion; this part should be in plaintext, whereas the unity example should start with `mscrolib`

![image-20231230183601458](/image-archive-20240105/image-20231230183601458.png)

- The use of metadata by il2cpp in the so does not add an extra step; presumably this part is decrypted in situ at load time

### 1. metadata dynamic extraction

- The same `il2cpp.so` extraction steps as documented in the previous post, so I won't go into them here.

![image-20231230183727385](/image-archive-20240105/image-20231230183727385.png)

- Compare the apk to the dumped binary:

![image-20231230184527296](/image-archive-20240105/image-20231230184527296.png)

Sure enough, the string part is clear; keep dumping!

![image-20231230183345026](/image-archive-20240105/image-20231230183345026.png)

- Success! Bringing information into ida; and the very dramatic thing is:

![image-20231230185256794](/image-archive-20240105/image-20231230185256794.png)

![image-20231230185620868](/image-archive-20240105/image-20231230185620868.png)

**Confusion is gone.**

*It's really a very pro-customer update (I'm sure of it).*

---

### 2. frida-gadget injection

Injections on WSA hit all sorts of walls, but it's possible to modify the apk for injection on a physical machine even without a root, via `frida-gadget`.

I've tried this before by changing the dex and it didn't work; picking a .so that will be loaded at runtime to start with:

**Note:** The target lib has been changed to `libFastAES.so`, the screenshot has not been updated yet.

![ ](/image-archive-20240105/image-20231229185227359.png)

![image-20231229192615725](/image-archive-20240105/image-20231229192615725.png)

![image-20231229185311898](/image-archive-20240105/image-20231229185311898.png)

Take apktool, package it and sign it to install it on the real phone.

![image-20231229192636829](/image-archive-20240105/image-20231229192636829.png)

**Note:** My frida is running on WSL, take the Windows machine's adb as the backend in the Win/Linux machine configuration respectively as follows

```cmd
adb.exe -a -P 5555 nodaemon server
```

``` bash
export ADB_SERVER_SOCKET=tcp:192.168.0.2:5555
```

- Tested with https://github.com/vfsfitvnm/frida-il2cpp-bridge, no problems found

![image-20231230233324422](/image-archive-20240105/image-20231230233324422.png)

### 3. IL2CPP dynamically invokes runtime

Next you can call the runtime stuff directly

- Recalling the above, it looks like the API business logic only adds a set of symmetric encryption (AES128 CBC); the same is true here

![image-20231231002031783](/image-archive-20240105/image-20231231002031783.png)

![image-20231231003122139](/image-archive-20240105/image-20231231003122139.png)

![image-20231231003153856](/image-archive-20240105/image-20231231003153856.png)

![image-20231231003214559](/image-archive-20240105/image-20231231003214559.png)

**Summary**:

- `APIManager` as a singleton instance
- `APIManager.get_Crypt()` gets `Crypt`, whose `aesAlgo` is the .NET standard `AesManaged`

To summarize, simply write a script:

```typescript
import "frida-il2cpp-bridge";

Il2Cpp.perform(() => {
    Il2Cpp.domain.assemblies.forEach((i)=>{
        console.log(i.name);
    })
    const game = Il2Cpp.domain.assembly("Assembly-CSharp").image;
    const apiManager = game.class("Sekai.APIManager");
    Il2Cpp.gc.choose(apiManager).forEach((instance: Il2Cpp.Object) => {
        console.log("instance found")
        const crypt = instance.method<Il2Cpp.Object>("get_Crypt").invoke();
        const aes = crypt.field<Il2Cpp.Object>("aesAlgo");
        const key = aes.value.method("get_Key").invoke();
        const iv = aes.value.method("get_IV").invoke();
        console.log(key);
        console.log(iv);
    });
});
```

The output is as follows:

![image-20231231003558106](/image-archive-20240105/image-20231231003558106.png)

![image-20231231001737613](/image-archive-20240105/image-20231231001737613.png)

- Test Decryption

![image-20231231002825994](/image-archive-20240105/image-20231231002825994.png)

The test script is as follows:

```python
from Crypto.Cipher import AES

def unpad(data):
    padding_len = data[-1]
    return data[:-padding_len]
def decrypt_aes_cbc(data, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(data))

payload = open('payload','rb').read()
key = b"g2fcC0ZczN9MTJ61"
iv = b"msx3IV0i9XE5uYZ1"

plaintext = decrypt_aes_cbc(payload, key, iv)
print(plaintext)
```

![image-20231231002849347](/image-archive-20240105/image-20231231002849347.png)

It's not clear what the serialization format is, probably protobuf; better to keep investigating in the next post

***SEE YOU SPACE COWBOY...***

### References

https://lief-project.github.io/doc/latest/tutorials/09_frida_lief.html

https://github.com/vfsfitvnm/frida-il2cpp-bridge

## Project SEKAI Reverse (3): pjsk API Man-in-the-Middle Attack POC

- Analyzing Variant: World Plan 2.6.1 (Google Play Taiwan)

Here and il2cpp seems to have little to do; the main record of api package hijacking can be used to means

### 1. Tools

host using https://github.com/mitmproxy/mitmproxy，

After installing https://github.com/NVISOsecurity/MagiskTrustUserCerts on victim, import the CA for mitmproxy and reboot to make it the root certificate

Finally, the script used by mitmproxy: https://github.com/mos9527/sssekai/blob/main/sssekai/scripts/mitmproxy_sekai_api.py

### 2. Analysis

The previous post guessed that the api's packet used protobuf; it didn't, here it is [MessagePack](https://msgpack.org/index.html)

In this way, the data schema and the message are together; no extra digging is needed

Do a poc realtime decryption to `json` with mitmproxy and see:

![image-20231231115914176](/image-archive-20240105/image-20231231115914176.png)

### 3. MITM

*Yes, messing with this just to see what the MASTER spectrum looks like... *

---

The field for the lock MASTER seems to be here; try to modify it directly

```python
    def response(self, flow : http.HTTPFlow):
        if self.filter(flow):
            body = self.log_flow(flow.response.content, flow)
            if body:
                if 'userMusics' in body:
                    print('! Intercepted userMusics')
                    for music in body['userMusics']:
                        for stat in music['userMusicDifficultyStatuses']:
                            stat['musicDifficultyStatus'] = 'available'
                    flow.response.content = sekai_api_encrypt(packb(body))
```

option can be lit; but to no avail: it appears that live will authenticate on the server side

![image-20231231121435020](/image-archive-20240105/image-20231231121435020.png)

![image-20231231121528777](/image-archive-20240105/image-20231231121528777.png)

But the live authentication id doesn't seem to be used in starting live; there's no secondary reference to the id in the grab bag

Consider the possibility that spectral difficulty selection is only done on the client side. Then modifying the difficulty factor reported to the server might be able to bypass the

```python
    def request(self,flow: http.HTTPFlow):
        print(flow.request.host_header, flow.request.url)
        if self.filter(flow):
            body = self.log_flow(flow.request.content, flow)
            if body:
                if 'musicDifficultyId' in body:
                    print('! Intercepted Live request')
                    body['musicDifficultyId'] = 4 # Expert
                flow.request.content = sekai_api_encrypt(packb(body))
```

Starting up again:

![image-20231231123020461](/image-archive-20240105/image-20231231123020461.png)

Of course, scores will only be reported by EXPERT difficulty

---

Fewer surprises, no more MITM-related content to follow

Next... Extract the game ASSET? Before that...

***SEE YOU SPACE COWBOY...***

### References

https://msgpack.org/index.html

https://github.com/mitmproxy/mitmproxy

## Project SEKAI 逆向（4）： pjsk AssetBundle 反混淆 + PV 动画导入

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

### 1. 数据提取

pjsk资源采用热更新模式；本体运行时之外，还会有~~3~4G左右的资源~~ （**注：**不定量，见下一篇）

- 尝试从本机提取资源

![image-20231231183530650](/image-archive-20240105/image-20231231183530650.png)

![image-20231231183558159](/image-archive-20240105/image-20231231183558159.png)

没有magic `UnityFS`,考虑ab文件有混淆

### 2. 加载流程分析

- 进dnSpy直接搜assetbundle找相关Class

![image-20231231183823530](/image-archive-20240105/image-20231231183823530.png)

- 进ida看impl，可以很轻松的找到加载ab的嫌疑流程

![image-20231231183917530](/image-archive-20240105/image-20231231183917530.png)

![image-20231231183933304](/image-archive-20240105/image-20231231183933304.png)

- 最后直接调用了unity的`LoadFromStream`，`Sekai.AssetBundleStream`实现了这样的Stream：

![image-20231231184111015](/image-archive-20240105/image-20231231184111015.png)

![image-20231231184246728](/image-archive-20240105/image-20231231184246728.png)

可以注意到

- 加载时根据 `_isInverted` flag 决定是否进行反混淆操作
- 如果有，则先跳过4bytes,之后5bytes按位取反
- 最后移交`InvertedBytesAB`继续处理
  - 注意到`n00`应为128，`v20`为读取offset

- 这里考虑offset=0情况，那么仅前128字节需要处理

跟进`InvertedBytesAB`

![image-20231231184647711](/image-archive-20240105/image-20231231184647711.png)

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

![image-20231231192311049](/image-archive-20240105/image-20231231192311049.png)

- 文件处理完后，就可以靠https://github.com/Perfare/AssetStudio查看资源了：

![image-20231231192416677](/image-archive-20240105/image-20231231192416677.png)

- 不过版本号很好找，这里是`2020.3.21f1`：

![image-20231231192541641](/image-archive-20240105/image-20231231192541641.png)

- 加载可行，如图：

![image-20231231192616533](/image-archive-20240105/image-20231231192616533.png)

### 4. AssetBundleInfo?

在数据目录里发现了这个文件，同时在`Sekai_AssetBundleManager__LoadClientAssetBundleInfo`中：

![image-20231231194342801](/image-archive-20240105/image-20231231194342801.png)

用的是和API一样的密钥和封包手段，解开看看

**注：** 工具移步 https://github.com/mos9527/sssekai；内部解密流程在文章中都有描述

```bash
python -m sssekai apidecrypt .\AssetBundleInfo .\AssetBundleInfo.json
```

![image-20231231202455181](/image-archive-20240105/image-20231231202455181.png)

### 5. 资源使用？

- 角色模型数很少

![image-20231231203837242](/image-archive-20240105/image-20231231203837242.png)

- 猜测这里的资源被热加载；在blender直接看看已经有的mesh吧：

  bind pose有问题，修正FBX导出设置可以解决；不过暂且不往这个方向深究

![image-20231231204536443](/image-archive-20240105/image-20231231204536443.png)

- 同时也许可以试试导入 Unity？

https://github.com/AssetRipper/AssetRipper/ 可以做到这一点，尝试如下：

![image-20231231212152781](/image-archive-20240105/image-20231231212152781.png)

![image-20231231212236240](/image-archive-20240105/image-20231231212236240.png)

![image-20231231212822730](/image-archive-20240105/image-20231231212822730.png)

- 拖进 Editor

![image-20240101141156185](/image-archive-20240105/image-20240101141156185.png)

- 注意shader并没有被拉出来，暂时用standard替补

![image-20240101152353581](/image-archive-20240105/image-20240101152353581.png)

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

![image-20240101141256456](/image-archive-20240105/image-20240101141256456.png)

- 注意到blendshape/morph名字对不上

![image-20240101141815895](/image-archive-20240105/image-20240101141815895.png)

![image-20240101141909497](/image-archive-20240105/image-20240101141909497.png)

爬了下issue：这里的数字是名称的crc32（见 https://github.com/AssetRipper/AssetRipper/issues/954）

![image-20240101142406334](/image-archive-20240105/image-20240101142406334.png)

![image-20240101142422934](/image-archive-20240105/image-20240101142422934.png)

- 拿blendshape名字做个map修复后，动画key正常

![image-20240101150057515](/image-archive-20240105/image-20240101150057515.png)

- 加上timeline后的播放效果

![Animation](/image-archive-20240105/Animation.gif)

不知道什么时候写之后的，暂时画几个饼：

- 资源导入Blender + toon shader 复刻
- 资源导入 [Foundation](https://github.com/mos9527/Foundation/tree/master/Source) 
- 脱离游戏解析+下载资源

***SEE YOU SPACE COWBOY...***

### References

https://github.com/AssetRipper/AssetRipper/

https://github.com/AssetRipper/AssetRipper/issues/954

https://github.com/mos9527/Foundation

## Project SEKAI 逆向（5）： AssetBundle 脱机 + USM 提取

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

### 1. AssetBundleInfo

前文提及的`AssetBundleInfo`是从设备提取出来的；假设是已经加载过的所有资源的缓存的话：

- 在刚刚完成下载的设备上提取该文件时，该文件 4MB

![image-20240101204313069](/image-archive-20240105/image-20240101204313069.png)

- 但是在初始化后重现抓包时发现的该文件为 **13MB**

```bash
curl -X GET 'https://184.26.43.87/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online/android21/AssetBundleInfo.json' -H 'Host: lf16-mkovscdn-sg.bytedgame.com' -H 'User-Agent: UnityPlayer/2020.3.32f1 (UnityWebRequest/1.0, libcurl/7.80.0-DEV)' -H 'Accept-Encoding: deflate, gzip' -H 'X-Unity-Version: 2020.3.32f1'
```

![image-20240101204525117](/image-archive-20240105/image-20240101204525117.png)

- 推测设备上文件为已缓存资源库，而这里的即为全量资源集合；尝试dump

```bash
 sssekai apidecrypt .\assetbundleinfo .\assetbundleinfo.json
```

- 查身体模型数看看吧

![image-20240101204751167](/image-archive-20240105/image-20240101204751167.png)

- 此外，这里的数据还会多几个field

新数据库：

```json
        "live_pv/model/character/body/21/0001/ladies_s": {
            "bundleName": "live_pv/model/character/body/21/0001/ladies_s",
            "cacheFileName": "db0ad5ee5cc11c50613e7a9a1abc4c55",
            "cacheDirectoryName": "33a2",
            "hash": "28b258e96108e44578028d36ec1a1565",
            "category": "Live_pv",
            "crc": 2544770552,
            "fileSize": 588586,
            "dependencies": [
                "android1/shader/live"
            ],
            "paths": null,
            "isBuiltin": false,
            "md5Hash": "f9ac19a16b2493fb3f6f0438ada7e269",
            "downloadPath": "android1/live_pv/model/character/body/21/0001/ladies_s"
        },
```

设备数据库：

```json
        "live_pv/model/character/body/21/0001/ladies_s": {
            "bundleName": "live_pv/model/character/body/21/0001/ladies_s",
            "cacheFileName": "db0ad5ee5cc11c50613e7a9a1abc4c55",
            "cacheDirectoryName": "33a2",
            "hash": "28b258e96108e44578028d36ec1a1565",
            "category": "Live_pv",
            "crc": 2544770552,
            "fileSize": 588586,
            "dependencies": [
                "android1/shader/live"
            ],
            "paths": null,
            "isBuiltin": false,
            "md5Hash": "",
            "downloadPath": ""
        },
```

多出的`downloadPath`可以利用，继续吧...

### 2. CDN？

- 启动下载后，能抓到一堆这种包：

```bash
curl -X GET 'https://184.26.43.74/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online/android1/actionset/group1?t=20240101203510' -H 'Host: lf16-mkovscdn-sg.bytedgame.com' -H 'User-Agent: UnityPlayer/2020.3.32f1 (UnityWebRequest/1.0, libcurl/7.80.0-DEV)' -H 'Accept-Encoding: deflate, gzip' -H 'X-Unity-Version: 2020.3.32f1'
```

`downloadPath`字段在这里出现了；看起来`https://184.26.43.74/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online` ` 是这里的AB的根路径

而`184.26.43.74`就是cdn了，毕竟

![image-20240101205501465](/image-archive-20240105/image-20240101205501465.png)

- cdn的地址看起来是内嵌的；在dump出来的strings中：

![image-20240101210240573](/image-archive-20240105/image-20240101210240573.png)

### 3. 热更新 Cache

考虑pjsk更新频率大，每次重新下所有数据不是很高效

做一个本地cache动机充分；细节就不在这里说了，请看 https://github.com/mos9527/sssekai/blob/main/sssekai/abcache/__init__.py

尝试拉取全部资源，貌似需要*27GB*

![image-20240102003435800](/image-archive-20240105/image-20240102003435800.png)

### 4. 文件一览

- 在 WinDirStat 中查看分布

![image-20240102095200320](/image-archive-20240105/image-20240102095200320.png)

- 动画资源：

![image-20240102095331527](/image-archive-20240105/image-20240102095331527.png)

- VO

![image-20240102095619789](/image-archive-20240105/image-20240102095619789.png)

![image-20240102095636123](/image-archive-20240105/image-20240102095636123.png)

其它的话，貌似是音视频文件居多

揭开后可以发现封包格式是[CriWare](https://www.criware.com/en/)中间件格式（i.e. USM视频流，HCA音频流）

### 5. USM 提取

*动机：应该很简单orz*

---

![image-20240102112727738](/image-archive-20240105/image-20240102112727738.png)

- 没有 Magic `CRID`

  回到IDA，看起来USM资源并不是直接从assetbundle中提取；中间有缓存到文件系统的流程

![image-20240102114924491](/image-archive-20240105/image-20240102114924491.png)

![image-20240102113153597](/image-archive-20240105/image-20240102113153597.png)

- 果然，在`/sdcard/Android/data/[...]/cache/movies`下有这样的文件

![image-20240102115142365](/image-archive-20240105/image-20240102115142365.png)

![image-20240102115207418](/image-archive-20240105/image-20240102115207418.png)

- 而且用[WannaCri](https://github.com/donmai-me/WannaCRI)可以直接demux，没有额外密钥

![image-20240102115303855](/image-archive-20240105/image-20240102115303855.png)

- 回顾asset中USM文件

![image-20240102115518322](/image-archive-20240105/image-20240102115518322.png)

- 利用`MovieBundleBuildData`猜测可以拼接出源文件

![image-20240102125531642](/image-archive-20240105/image-20240102125531642.png)

**脚本：**https://github.com/mos9527/sssekai/blob/main/sssekai/entrypoint/usmdemux.py

![image-20240102135738112](/image-archive-20240105/image-20240102135738112.png)

***SEE YOU SPACE COWBOY...***

### References

https://github.com/mos9527/sssekai

https://github.com/donmai-me/WannaCRI

## Project SEKAI 逆向（6）：Live2D 资源

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

### 1. Live2D 模型

![image-20240102205059463](/image-archive-20240105/image-20240102205059463.png)

- 所有live2d资源都可以在 `[abcache]/live2d/`下找到；包括模型及动画

首先，`.moc3`,`.model3`,`.physics3`资源都可以直接利用[Live2D Cubism Editor](https://www.live2d.com/en/cubism/download/editor/)直接打开

而模型材质需要额外更名；这些信息都在`BuildModelData`中

![image-20240102205701929](/image-archive-20240105/image-20240102205701929.png)

- 补全后即可导入，效果如图

![image-20240102205542299](/image-archive-20240105/image-20240102205542299.png)

### 2. 动画 Key 预处理

- 可惜动画并不是`.motion3`格式

  封包中有的是Unity自己的Animation Clip

  在提取资源时，所有的动画key只能读到对应key string的CRC32 hash；导出/操作必须知道string-hash关系

![image-20240102210045486](/image-archive-20240105/image-20240102210045486.png)

- 这些string在`moc3`以外的文件中未知：当然，碰撞出string也不现实；猜想string和Live2D参数有关

![image-20240102210113370](/image-archive-20240105/image-20240102210113370.png)

尝试搜索无果

![image-20240102210134670](/image-archive-20240105/image-20240102210134670.png)

- 幸运的是Live2D Unity SDK可以免费取得，而且附带样例

  还记得前文处理BlendShape时，可以知道`AnimationClip`的源`.anim`会有path的源string，而不是crc

![image-20240102210341040](/image-archive-20240105/image-20240102210341040.png)

尝试加入前缀

![image-20240102210356955](/image-archive-20240105/image-20240102210356955.png)

![image-20240102210406498](/image-archive-20240105/image-20240102210406498.png)

可以定位；下面介绍如何构建CRC表，完成crc-string map

### 3. moc3 反序列化 + CRC打表

- 每次读取都从`moc3`文件构造应该可行；不过考虑到有导入纯动画的需求，显然一个常量map是需要的

- 故需要能读取`moc3`中所有参数名；参照https://raw.githubusercontent.com/OpenL2D/moc3ingbird/master/src/moc3.hexpat

  在 ImHex 中可见：

![image-20240103090132409](/image-archive-20240105/image-20240103090132409.png)

- 提取参数名脚本如下：

```python
from typing import BinaryIO
from struct import unpack

## https://github.com/OpenL2D/moc3ingbird/blob/master/src/moc3.hexpat
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

![image-20240102225301658](/image-archive-20240105/image-20240102225301658.png)

### 4. AnimationClip 转换

Live2D有自己私有的动画格式`motion3`，幸运的是[UnityLive2DExtractor](https://github.com/Perfare/UnityLive2DExtractor)已做了相当多的解析实现，可供参考

由于上文介绍的细节出入，对PJSK的转换并不能直接使用这个工具

索性在[sssekai](https://github.com/mos9527/sssekai)重现；细节**非常**繁琐，再次不多说；有兴趣的话还请参考源码

- 使用例：将转化所有找到的`AnimationClip`为`.motion3.json`

```bash
sssekai live2dextract c:\Users\mos9527\.sssekai\abcache\live2d\motion\21miku_motion_base .
```

- 效果如图

![sssekai-live2d-anim-import-demo](/image-archive-20240105/sssekai-live2d-anim-import-demo.gif)

***SEE YOU SPACE COWBOY...***

### References

https://github.com/AssetRipper/AssetRipper

https://github.com/OpenL2D/moc3ingbird

https://github.com/Perfare/UnityLive2DExtractor

## Project SEKAI 逆向（7）：3D 模型

### 1. 文件结构

- 目前发现的 3D 模型基本都在 `[ab cache]/live_pv/model/` 下

![image-20240105080707360](/image-archive-20240105/image-20240105080707360.png)

![image-20240105080841622](/image-archive-20240105/image-20240105080841622.png)

初步观察：

- (1) Body 即目标模型；当然，作为skinned mesh，而且带有blend shapes，处理细节会很多；后面继续讲
- (2) 处的 `MonoBehavior` 就其名字猜测是碰撞盒
- (3) 处的几个 `Texture2D` 则作为texture map

### 2. 模型收集

利用`sssekai`取得数据的流程在（5）中已有描述，这里不再多说

首先整理下根据mesh发现的**数据需求**

- (1) **Static Mesh**

pjsk发现的所有mesh在相应assetbundle中会有1个或更多`GameObject`的ref；对于这些ref，static mesh会出现在`m_MeshRenderer`之中

其他细节暂且不说；因为做 Skinned Mesh 导入时都是我们要处理的东西

- (2) **Skinned Mesh**

不同于static mesh,这些ref会出现在`m_SkinnedMeshRenderer`之中

同时，我们也会需要**骨骼结构的信息**；bone  weight以外，也需要bone path（后面会用来反向hash）和transform

- (3) **Blend Shapes**

  这些可以出现在static/skinned mesh之中；如果存在，我们也会需要blend shape名字的hash，理由和bone path一致

  加之，Unity存在aseetbundle中动画path也都是crc，blendshape不是例外

**总结:**

- (1) 所以对于static mesh,搜集对应`GameObject`即可

- (2) 对于skinned mesh，同时也需要构造bone hierarchy（就是个单根有向无环图啦），并且整理vertex权重；

  则需要收集的，反而只是bone的transform而已；transform有子/父节点信息，也有拥有transform的`GameObject`的ref

- (3) 的数据，在(1)(2)中都会有

### 3. 模型导入

当然，这里就不考虑将模型转化为中间格式了（i.e. FBX,GLTF）

利用Blender Python，可以直接给这些素材写个importer

实现细节上，有几个值得注意的地方：

- Unity读到的mesh是triangle list

- Blender使用右手系，Unity/Direct3D使用左手系

| 坐标系  | 前   | 上   | 左   |
| ------- | ---- | ---- | ---- |
| Unity   | Z    | Y    | X    |
| Blender | -Y   | Z    | -X   |

  - 意味着对向量需要如下转化

    $\vec{V_{blender}}(X,Y,Z) = \vec(-V_{unity}.X,-V_{unity}.Z,V_{unity}.Y)$

  - 对四元数XYZ部分

    $\vec{Q_{blender}}(W,X,Y,Z) = \overline{\vec(V_{unity}.W,-V_{unity}.X,-V_{unity}.Z,V_{unity}.Y)}$

- Unity存储vector类型数据可能以2,3,4或其他个数浮点数读取，而vector不会额外封包，需要从flat float array中读取

  意味着需要这样的处理

  ```python
         vtxFloats = int(len(data.m_Vertices) / data.m_VertexCount)
         vert = bm.verts.new(swizzle_vector3(
              data.m_Vertices[vtx * vtxFloats], # x,y,z
              data.m_Vertices[vtx * vtxFloats + 1],
              data.m_Vertices[vtx * vtxFloats + 2]            
          ))
  ```

  嗯。这里的`vtxFloats`就有可能是$4$. 虽然$w$项并用不到

- 对于BlendShape, blender并不支持用他们修改法线或uv;这些信息只能丢掉

- **Blender的BlendShape名字不能超过64字，否则名称会被截取**

- 对于bone,他们会以`Transform`的方式呈现；但在模型（和动画文件）中，他们只会以`Scene`中这些**transform的完整路径的hash存储**

- 然后，**Blender的Vertex Group(bone weight group)同样也不能有64+长名字**

- 对于vertex color，blender的`vertex_colors`layer在4.0已被弃用；不过可以放在**Color Atrributes**

**注：**Blender中对写脚本帮助很大的一个小功能

![image-20240104202236513](/image-archive-20240105/image-20240104202236513.png)

![image-20240105085540376](/image-archive-20240105/image-20240105085540376.png)

### 4. Shaders!

`Texture2D`和其他meta信息导入后，接下来就是做shader了

- 手头有的纹理资源如下：

1. `tex_[...]_C`

   Base **C**olor Map，没什么好说的

![image-20240105081336340](/image-archive-20240105/image-20240105081336340.png)

2. `tex_[...]_S`

   **S**hadowed Color Map（乱猜

   - NPR渲染中常用的阈值Map；为节省性能（和细节质量），引擎也许并不会绘制真正的**Shadow Map**

   - 在很多 NPR Shader中，你会见到这样的逻辑：

   ```glsl
   if (dot(N, L) > threshold) {
   	diffuse = Sample(ColorMap, uv);
   } else {
   	diffuse = Sample(ShadowColorMap, uv);
   }
   ```

即：对NdotL作阈值处理，光线亮（NdotL更大）采用原map，光线暗/无法照明（NdotL更小或为负）采用阴影map

![image-20240105081322556](/image-archive-20240105/image-20240105081322556.png)

3. `tex_[...]_H`

   **H**ightlight Map

   - ~~注意到`Format`出于某种原因竟然是未压缩的`RGB565`;同时,$R$通道恒为$0$,$B$通道恒为$132$，只有G通道有带意义的信息~~
   - **UPD (20240907):** 据指正，$R$通道定义肤色,$B$定义NPR阴影阈值，$G$通道标记了高光部分
   - **UPD (20240907):** 同时，在三周年（JP）更新后,$a$通道指定了光泽度

![image-20240105081327608](/image-archive-20240105/image-20240105081327608.png)

4. Vertex Color

   - [虽然不是]()texture map，但是放这里讲会合适不少

![image-20240105180210479](/image-archive-20240105/image-20240105180210479.png)

- 这里只有RG通道有信息,猜测：

  - $R$通道决定是否接受描边

  - $G$通道决定高光强度


### 5. Shader 实现

1. 阴影 / Diffuse

![image-20240105180740265](/image-archive-20240105/image-20240105180740265.png)

**注：** BSDF应为Diffuse BSDF,截图暂未更新

这里实现的即为上文所述的阈值阴影，不多说了

2. Specular 

直接利用Specular BSDF的输出和前文所提到的weight，mix到输出即可

3. Emissive

用`_H`材质的$G$通道叠加，node如图

![image-20240105183024248](/image-archive-20240105/image-20240105183024248.png)

至此Shader部分介绍完毕，效果如图

![image-20240105183301304](/image-archive-20240105/image-20240105183301304.png)

![image-20240105183407294](/image-archive-20240105/image-20240105183407294.png)

### 6.描边

游戏使用了经典的shell shading技术

- 细节上，重现如图效果的话... (PV: [愛して愛して愛して](https://www.bilibili.com/video/BV1cP4y1P7TM/)))

![image-20240105183827963](/image-archive-20240105/image-20240105183827963.png)

- 可见$1$区域带明显描边而$2$区域没有，观察vertex color：

![image-20240105183943570](/image-archive-20240105/image-20240105183943570.png)

![image-20240105184014817](/image-archive-20240105/image-20240105184014817.png)

这和之前对描边做的猜测是一致的; $R$​值决定是否描边

显然在 Blender 中使用 Geometry Node 可以很轻松地实现这个效果

---


***SEE YOU SPACE COWBOY...***

### References

https://github.com/mos9527/sssekai_blender_io 👈 插件在这

https://github.com/KH40-khoast40/Shadekai

https://github.com/KhronosGroup/glTF-Blender-IO

https://github.com/theturboturnip/yk_gmd_io

https://github.com/SutandoTsukai181/yakuza-gmt-blender

https://github.com/UuuNyaa/blender_mmd_tools