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

Compare to Unity's default implementation （https://github.com/mos9527/il2cpp-27/blob/main/libil2cpp/vm/MetadataLoader.cpp) ：

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

## Project SEKAI Reverse (4): pjsk AssetBundle Anti-Obfuscation + PV Animation Import

- Analyzing Variant: World Plan 2.6.1 (Google Play Taiwan)

### 1. Data extraction

The pjsk resources are in hot update mode; there will be ~~3~4G or so in addition to the ontology runtime~~ (**Note: **Variable amount, see next post)

- Trying to extract resources from the local machine

![image-20231231183530650](/image-archive-20240105/image-20231231183530650.png)

![image-20231231183558159](/image-archive-20240105/image-20231231183558159.png)

There is no magic `UnityFS`, consider the ab file confusing

### 2. Load flow analysis

- Go to dnSpy and search for assetbundle to find the relevant class.

![image-20231231183823530](/image-archive-20240105/image-20231231183823530.png)

- Going into ida and looking at the impl, you can easily find the suspect process of loading the ab

![image-20231231183917530](/image-archive-20240105/image-20231231183917530.png)

![image-20231231183933304](/image-archive-20240105/image-20231231183933304.png)

- It ends up calling unity's `LoadFromStream` directly, and `Sekai.AssetBundleStream` implements such a Stream:

![image-20231231184111015](/image-archive-20240105/image-20231231184111015.png)

![image-20231231184246728](/image-archive-20240105/image-20231231184246728.png)

It may be noted that

- The `_isInverted` flag determines whether or not to perform an anti-obfuscation operation when loading.
- If yes, skip 4bytes first, then invert 5bytes byte by bit.
- Final handover to `InvertedBytesAB` for further processing
  - Notice that `n00` should be 128 and `v20` is the read offset

- Consider the offset=0 case, then only the first 128 bytes need to be processed

Follow-up to `InvertedBytesAB'

![image-20231231184647711](/image-archive-20240105/image-20231231184647711.png)

It can be seen that here, i.e. **after skipping 4bytes, every 8bytes, invert the first 5bytes**

In summary, the decryption process is analyzed; the script is attached:

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

### 3. Extraction of resources

![image-20231231192311049](/image-archive-20240105/image-20231231192311049.png)

- Once the file is processed, you can rely on https://github.com/Perfare/AssetStudio查看资源了:

![image-20231231192416677](/image-archive-20240105/image-20231231192416677.png)

- The version number is easy to find though, here it is `2020.3.21f1`:

![image-20231231192541641](/image-archive-20240105/image-20231231192541641.png)

- Loading works, as shown:

![image-20231231192616533](/image-archive-20240105/image-20231231192616533.png)

### 4. AssetBundleInfo?

Found this file in the data directory along with `Sekai_AssetBundleManager__LoadClientAssetBundleInfo`:

![image-20231231194342801](/image-archive-20240105/image-20231231194342801.png)

Using the same key and packetized means as the API, unpack it and see

**Note:** Tools moved to https://github.com/mos9527/sssekai; the internal decryption process is described in the article

```bash
python -m sssekai apidecrypt .\AssetBundleInfo .\AssetBundleInfo.json
```

![image-20231231202455181](/image-archive-20240105/image-20231231202455181.png)

### 5. Resource utilization?

- Low number of character models

![image-20231231203837242](/image-archive-20240105/image-20231231203837242.png)

- Guessing that the resources here are being hot loaded; take a look at the mesh that's already there in blender directly:

  There is a problem with the bind pose, which can be solved by fixing the FBX export settings; but let's not go deeper in that direction for now.

![image-20231231204536443](/image-archive-20240105/image-20231231204536443.png)

- Also maybe try importing Unity?

https://github.com/AssetRipper/AssetRipper/ This can be done by trying the following:

![image-20231231212152781](/image-archive-20240105/image-20231231212152781.png)

![image-20231231212236240](/image-archive-20240105/image-20231231212236240.png)

![image-20231231212822730](/image-archive-20240105/image-20231231212822730.png)

- Drag and drop into Editor

![image-20240101141156185](/image-archive-20240105/image-20240101141156185.png)

- Note that shader isn't being pulled out and is being replaced with standard for now

![image-20240101152353581](/image-archive-20240105/image-20240101152353581.png)

- Separate face/body mesh; need to bind face root bone (Neck) to body (Neck)

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

- Notice that the blendshape/morph names don't match.

![image-20240101141815895](/image-archive-20240105/image-20240101141815895.png)

![image-20240101141909497](/image-archive-20240105/image-20240101141909497.png)

Crawling down the issue: the number here is the crc32 of the name (see https://github.com/AssetRipper/AssetRipper/issues/954)

![image-20240101142406334](/image-archive-20240105/image-20240101142406334.png)

![image-20240101142422934](/image-archive-20240105/image-20240101142422934.png)

- After taking the blendshape name and doing a map fix, the animation key works fine

![image-20240101150057515](/image-archive-20240105/image-20240101150057515.png)

- Playback with timeline

![Animation](/image-archive-20240105/Animation.gif)

I don't know when I'm going to write the afterward, so I'll paint a few pies for now:

- Resource import Blender + toon shader replica
- Resource Import [Foundation](https://github.com/mos9527/Foundation/tree/master/Source) 
- Disengagement Game Explanation + Download Resources

***SEE YOU SPACE COWBOY...***

### References

https://github.com/AssetRipper/AssetRipper/

https://github.com/AssetRipper/AssetRipper/issues/954

https://github.com/mos9527/Foundation

## Project SEKAI Reverse (5): AssetBundle offline + USM extraction

- Analyzing Variant: World Plan 2.6.1 (Google Play Taiwan)

### 1. AssetBundleInfo

The `AssetBundleInfo` mentioned earlier is extracted from the device; assuming it is a cache of all resources that have been loaded:

- When extracting the file on a device that has just completed the download, the file 4MB

![image-20240101204313069](/image-archive-20240105/image-20240101204313069.png)

- However, the file found when reproducing the packet capture after initialization is **13MB**

```bash
curl -X GET 'https://184.26.43.87/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online/android21/AssetBundleInfo.json' -H 'Host: lf16-mkovscdn-sg.bytedgame.com' -H 'User-Agent: UnityPlayer/2020.3.32f1 (UnityWebRequest/1.0, libcurl/7.80.0-DEV)' -H 'Accept-Encoding: deflate, gzip' -H 'X-Unity-Version: 2020.3.32f1'
```

![image-20240101204525117](/image-archive-20240105/image-20240101204525117.png)

- It is assumed that the files on the device are cached repositories, and the ones here are the full set of resources; try dumping them

```bash
 sssekai apidecrypt .\assetbundleinfo .\assetbundleinfo.json
```

- Let's check the body model number

![image-20240101204751167](/image-archive-20240105/image-20240101204751167.png)

- In addition, the data here will have a few more fields

New database:

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

Equipment database:

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

The extra `downloadPath` can be utilized, go ahead...

### 2. CDN？

- After initiating the download, you can grab a bunch of these packages:

```bash
curl -X GET 'https://184.26.43.74/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online/android1/actionset/group1?t=20240101203510' -H 'Host: lf16-mkovscdn-sg.bytedgame.com' -H 'User-Agent: UnityPlayer/2020.3.32f1 (UnityWebRequest/1.0, libcurl/7.80.0-DEV)' -H 'Accept-Encoding: deflate, gzip' -H 'X-Unity-Version: 2020.3.32f1'
```

The `downloadPath` field comes up here; it looks like `https://184.26.43.74/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online` ` is the root path of AB here

And `184.26.43.74` is the cdn, after all

![image-20240101205501465](/image-archive-20240105/image-20240101205501465.png)

- The cdn address appears to be embedded; in the strings that come out of the dump:

![image-20240101210240573](/image-archive-20240105/image-20240101210240573.png)

### 3. Hot Updates Cache

Considering the frequency of pjsk updates, it's not very efficient to re-download all of the data each time

Doing a local cache is motivated sufficiently; the details will not be covered here, see https://github.com/mos9527/sssekai/blob/main/sssekai/abcache/__init__.py

Trying to pull the full resource, it looks like it takes *27GB*

![image-20240102003435800](/image-archive-20240105/image-20240102003435800.png)

### 4. List of documents

- View the distribution in WinDirStat

![image-20240102095200320](/image-archive-20240105/image-20240102095200320.png)

- Animation Resources:

![image-20240102095331527](/image-archive-20240105/image-20240102095331527.png)

- VO

![image-20240102095619789](/image-archive-20240105/image-20240102095619789.png)

![image-20240102095636123](/image-archive-20240105/image-20240102095636123.png)

Otherwise, it seems to be mostly audio/video files

Uncovering reveals that the packet format is [CriWare](https://www.criware.com/en/) middleware format (i.e. USM video stream, HCA audio stream)

### 5. USM extraction

*Motivation: it should be simple orz*

---

![image-20240102112727738](/image-archive-20240105/image-20240102112727738.png)

- No Magic `CRID`

  Going back to IDA, it looks like the USM resources aren't pulled directly from the assetbundle; there's a cache-to-filesystem process in between

![image-20240102114924491](/image-archive-20240105/image-20240102114924491.png)

![image-20240102113153597](/image-archive-20240105/image-20240102113153597.png)

- Sure enough, under `/sdcard/Android/data/[...]/cache/movies` there is this file

![image-20240102115142365](/image-archive-20240105/image-20240102115142365.png)

![image-20240102115207418](/image-archive-20240105/image-20240102115207418.png)

- And with [WannaCri](https://github.com/donmai-me/WannaCRI) you can just demux, no extra keys!

![image-20240102115303855](/image-archive-20240105/image-20240102115303855.png)

- Review of USM files in asset

![image-20240102115518322](/image-archive-20240105/image-20240102115518322.png)

- Using `MovieBundleBuildData` it is guessed that the source file can be stitched together

![image-20240102125531642](/image-archive-20240105/image-20240102125531642.png)

**Script:** https://github.com/mos9527/sssekai/blob/main/sssekai/entrypoint/usmdemux.py

![image-20240102135738112](/image-archive-20240105/image-20240102135738112.png)

***SEE YOU SPACE COWBOY...***

### References

https://github.com/mos9527/sssekai

https://github.com/donmai-me/WannaCRI

## Project SEKAI Reverse (6): Live2D Resources

- Analyzing Variant: World Plan 2.6.1 (Google Play Taiwan)

### 1. Live2D model

![image-20240102205059463](/image-archive-20240105/image-20240102205059463.png)

- All live2d resources can be found under `[abcache]/live2d/`; including models and animations.

First of all, `.moc3`, `.model3`, and `.physics3` resources can be opened directly with [Live2D Cubism Editor](https://www.live2d.com/en/cubism/download/editor/)

And the model material needs to be renamed additionally; this information is in the `BuildModelData`

![image-20240102205701929](/image-archive-20240105/image-20240102205701929.png)

- Completion can be imported, the effect is as shown in the figure

![image-20240102205542299](/image-archive-20240105/image-20240102205542299.png)

### 2. Animated Key Preprocessing

- Unfortunately the animation is not in `.motion3` format

  What's in the package is Unity's own Animation Clip

  When extracting resources, all animated keys can only read the CRC32 hash of the corresponding key string; the export/operation must know the string-hash relationship

![image-20240102210045486](/image-archive-20240105/image-20240102210045486.png)

- These strings are unknown in files other than `moc3`: of course, it's not practical to collide out the strings; guessing that the strings have something to do with the Live2D parameter

![image-20240102210113370](/image-archive-20240105/image-20240102210113370.png)

Tried searching to no avail

![image-20240102210134670](/image-archive-20240105/image-20240102210134670.png)

- Luckily the Live2D Unity SDK is available for free and comes with samples!

  Remember from the previous article dealing with BlendShape, you can tell that `AnimationClip`'s source `.anim` will have the source string of path, not crc

![image-20240102210341040](/image-archive-20240105/image-20240102210341040.png)

Try adding a prefix

![image-20240102210356955](/image-archive-20240105/image-20240102210356955.png)

![image-20240102210406498](/image-archive-20240105/image-20240102210406498.png)

can be located; the following describes how to construct a CRC table to complete the crc-string map

### 3. moc3 deserialization + CRC hit list

- It should be feasible to construct from the `moc3` file for each read; however, given the need to import pure animations, it is clear that a constant map is required

- Therefore, it is necessary to be able to read all parameter names in `moc3`; see https://raw.githubusercontent.com/OpenL2D/moc3ingbird/master/src/moc3.hexpat.

  Visible in ImHex:

![image-20240103090132409](/image-archive-20240105/image-20240103090132409.png)

- The script for extracting parameter names is as follows:

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

- After that, constructing the CRC table is simple

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

- The export results are as follows:

![image-20240102225301658](/image-archive-20240105/image-20240102225301658.png)

### 4. AnimationClip Conversion

Live2D has its own private animation format `motion3`, fortunately [UnityLive2DExtractor](https://github.com/Perfare/UnityLive2DExtractor) has done quite a bit of parsing to implement this for reference

Due to the discrepancies in the details described above, the conversion of PJSK cannot be done directly using this tool.

Solely reproduced at [sssekai](https://github.com/mos9527/sssekai); details **very** tedious, again not much to say; see also the source code if you're interested

- Usage example: will transform all found `AnimationClip` to `.motion3.json`.

```bash
sssekai live2dextract c:\Users\mos9527\.sssekai\abcache\live2d\motion\21miku_motion_base .
```

- The effect is as shown in the picture.

![sssekai-live2d-anim-import-demo](/image-archive-20240105/sssekai-live2d-anim-import-demo.gif)

***SEE YOU SPACE COWBOY...***

### References

https://github.com/AssetRipper/AssetRipper

https://github.com/OpenL2D/moc3ingbird

https://github.com/Perfare/UnityLive2DExtractor

## Project SEKAI Reverse (7): 3D Modeling

### 1. Structure of the document

- The 3D models found so far are basically under `[ab cache]/live_pv/model/`

![image-20240105080707360](/image-archive-20240105/image-20240105080707360.png)

![image-20240105080841622](/image-archive-20240105/image-20240105080841622.png)

Preliminary Observations:

- (1) Body is the target model; of course, as a skinned mesh and with blend shapes, there is a lot of detail to deal with; more on that later!
- (2) The `MonoBehavior` at `MonoBehavior` is presumably a collision box based on its name
- (3) Several `Texture2D`s are used as texture maps

### 2. Model collection

The process of obtaining data using `sssekai` is described in (5) and will not be repeated here.

First organize the **data requirements** found based on mesh

- (1) **Static Mesh**

All meshes found by pjsk will have 1 or more `GameObject` refs in the corresponding assetbundle; for these refs, the static mesh will appear in the `m_MeshRenderer`

Other details aside; because doing a Skinned Mesh import is all stuff we have to deal with

- (2) **Skinned Mesh**

Unlike static mesh, these refs appear in the `m_SkinnedMeshRenderer`

Also, we will need **information about the bone structure**; in addition to the bone weight, we will also need the bone path (which will be used for reverse hash later) and the transformation

- (3) **Blend Shapes**

  These can appear in static/skinned mesh; if they do, we will also need the hash of the blend shape name for the same reason as the bone path

  Plus, Unity exists aseetbundle in which the animation path is also all crc, blendshape is not the exception

**Summary:**

- (1) So for static mesh, just collect the `GameObject`

- (2) For skinned mesh, it is also necessary to construct a bone hierarchy (which is a single rooted directed acyclic graph) and organize the vertex weights;

  Instead, all that needs to be collected is the bone's transform; the transform has the child/parent node information, as well as the ref of the `GameObject` that owns the transform.

- (3) of the data in (1)(2) will both have the

### 3. Model import

Of course, the conversion of the model to an intermediate format is not considered here (i.e. FBX,GLTF)

With Blender Python, it's possible to write an importer directly to these clips

There are a few noteworthy aspects of the implementation details:

- Unity reads the mesh as a triangle list

- Blender uses a right-handed system, Unity/Direct3D use a left-handed system

| Coordinate System  | Forward   | Up   | Left   |
| ------- | ---- | ---- | ---- |
| Unity   | Z    | Y    | X    |
| Blender | -Y   | Z    | -X   |

  - means that the following transformations are needed for vectors

    $\vec{V_{blender}}(X,Y,Z) = \vec(-V_{unity}.X,-V_{unity}.Z,V_{unity}.Y)$

  - For the XYZ part of the quaternion

    $\vec{Q_{blender}}(W,X,Y,Z) = \overline{\vec(V_{unity}.W,-V_{unity}.X,-V_{unity}.Z,V_{unity}.Y)}$

- Unity stores vector type data which may be read as 2,3,4 or other number of floats, whereas vectors are not additionally packetized and need to be read from a flat float array.

  Meaning it needs to be handled like this

  ```python
         vtxFloats = int(len(data.m_Vertices) / data.m_VertexCount)
         vert = bm.verts.new(swizzle_vector3(
              data.m_Vertices[vtx * vtxFloats], # x,y,z
              data.m_Vertices[vtx * vtxFloats + 1],
              data.m_Vertices[vtx * vtxFloats + 2]            
          ))
  ```

  Hmm. Here the `vtxFloats` could be $4$. Although the $w$ term is not used

- For BlendShape, blender doesn't support modifying normals or uv's with them; this information can only be thrown away

- **Blender's BlendShape name can't be more than 64 characters or the name will be truncated**

- For the bone, they will be rendered as `Transform`; but in the model (and animation files), they will only be stored as a hash of the full path of these **transforms in the `Scene` **

- Then, **Blender's Vertex Group (bone weight group) likewise can't have 64+ long names**

- For vertex color, blender's `vertex_colors` layer is deprecated in 4.0; however, it can be placed in **Color Attributes**.

**Note:** A small feature in Blender that helps a lot in writing scripts

![image-20240104202236513](/image-archive-20240105/image-20240104202236513.png)

![image-20240105085540376](/image-archive-20240105/image-20240105085540376.png)

### 4. Shaders!

After `Texture2D` and other meta information is imported, the next step is to make the shader

- Texture resources on hand are listed below:

1. `tex_[...]_C`

   Base **C**olor Map, nothing to it

![image-20240105081336340](/image-archive-20240105/image-20240105081336340.png)

2. `tex_[...]_S`

   **S**hadowed Color Map (wild guess)

   - Threshold Map commonly used in NPR rendering; to save performance (and detail quality), the engine may not draw a true **Shadow Map**

   - You'll see this logic in many NPR Shaders:

   ```glsl
   if (dot(N, L) > threshold) {
   	diffuse = Sample(ColorMap, uv);
   } else {
   	diffuse = Sample(ShadowColorMap, uv);
   }
   ```

I.e., NdotL is thresholded, with the original map used for bright light (larger NdotL) and the shadow map used for dark light/unable to illuminate (smaller or negative NdotL)

![image-20240105081322556](/image-archive-20240105/image-20240105081322556.png)

3. `tex_[...]_H`

   **H**ightlight Map

   - ~~Note that `Format` is for some reason an uncompressed `RGB565`; at the same time, the $R$-channel is always $0$, the $B$-channel is always $132$, and only the G-channel has any meaningful information ~~
   - **UPD (20240907):** Corrected, $R$ channel defines skin color, $B$ defines NPR shadow threshold, and $G$ channel marks highlights.
   - **UPD (20240907):** Meanwhile, after the third anniversary (JP) update, the $a$-channel specifies glossiness

![image-20240105081327608](/image-archive-20240105/image-20240105081327608.png)

4. Vertex Color

   - [It's not]() texture map, but it would fit in here quite well

![image-20240105180210479](/image-archive-20240105/image-20240105180210479.png)

- Only the RG channel has information here, guess:

  - The $R$ channel decides whether to accept the stroke

  - The $G$ channel determines the high light intensity


### 5. Shader implementation

1. Shadow / Diffuse

![image-20240105180740265](/image-archive-20240105/image-20240105180740265.png)

**Note:** BSDF should be Diffuse BSDF, screenshot not yet updated

What is implemented here is the threshold shading described above, without further ado

2. Specular 

Directly using the output of Specular BSDF and the previously mentioned weight, mix to the output is sufficient

3. Emissive

Superimposed with the $G$ channel of the `_H` material, node as in the picture

![image-20240105183024248](/image-archive-20240105/image-20240105183024248.png)

At this point the Shader part of the introduction is complete, the effect is shown in Figure

![image-20240105183301304](/image-archive-20240105/image-20240105183301304.png)

![image-20240105183407294](/image-archive-20240105/image-20240105183407294.png)

### 6. Stroke

The game uses the classic shell shading technique

- Details to reproduce the effect as shown if... (PV: [愛して愛して愛して愛して](https://www.bilibili.com/video/BV1cP4y1P7TM/)))

![image-20240105183827963](/image-archive-20240105/image-20240105183827963.png)

- It can be seen that the $1$ region has a distinct stroke while the $2$ region does not, observe vertex color:

![image-20240105183943570](/image-archive-20240105/image-20240105183943570.png)

![image-20240105184014817](/image-archive-20240105/image-20240105184014817.png)

This is consistent with the previous guess made about the stroke; the $R$ value determines whether or not the stroke is made

Obviously it's easy to achieve this effect in Blender using Geometry Node

---


***SEE YOU SPACE COWBOY...***

### References

https://github.com/mos9527/sssekai_blender_io 👈 The plugin is here

https://github.com/KH40-khoast40/Shadekai

https://github.com/KhronosGroup/glTF-Blender-IO

https://github.com/theturboturnip/yk_gmd_io

https://github.com/SutandoTsukai181/yakuza-gmt-blender

https://github.com/UuuNyaa/blender_mmd_tools