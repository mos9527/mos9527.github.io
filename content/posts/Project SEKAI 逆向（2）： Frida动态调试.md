---
author: mos9527
lastmod: 2023-12-29T18:45:50.537299+08:00
title: Project SEKAI 逆向（2）： Frida动态调试及取API AES KEY/IV
tags: ["逆向","frida","pjsk","api","project sekai","miku"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Project SEKAI 逆向（2）： Frida动态调试及取API AES KEY/IV

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

### 0. 版本更新

毕竟美服的数据库滞后好几个版本号...

换到台服之后版本号(2.4.1->2.6.1)有变化，同时上一篇所说的`metadata`加密手段也换了

- 同样，拉取一份修复过的dump分析，可见:

![image-20231230182936676](/assets/image-20231230182936676.png)

- `metadata`加载部分不再进行全局加解密，观察二进制也可以发现：

![image-20231230183059925](/assets/image-20231230183059925.png)

2.4.0解密后`metadata`

![image-20231230183133793](/assets/image-20231230183133793.png)

2.6.1源`metadata`多出来8字节？

![image-20231230183234026](/assets/image-20231230183234026.png)

![image-20231230183305449](/assets/image-20231230183305449.png)

- 加载部分出入不大，尝试删掉8字节直接分析

![image-20231230183414173](/assets/image-20231230183414173.png)

- 卡在读string这一块；到`Il2CppGlobalMetadataHeader.stringOffset`检查一下

![image-20231230183456731](/assets/image-20231230183456731.png)

- 果然有混淆；这部分应该是明文，而unity的例子中开始应该是`mscrolib`

![image-20231230183601458](/assets/image-20231230183601458.png)

- 而so中il2cpp对metadata的使用却没加多余步骤；推测这部分在加载时进行了原地解密

### 1. metadata 动态提取

- 和上一篇记录的`il2cpp.so`提取步骤如出一辙，这里不再多说

![image-20231230183727385](/assets/image-20231230183727385.png)

- 比对apk的和dump出来的二进制：

![image-20231230184527296](/assets/image-20231230184527296.png)

果然string部分明朗了；继续dump

![image-20231230183345026](/assets/image-20231230183345026.png)

- 成功！带信息进ida；而非常戏剧化的事情是：

![image-20231230185256794](/assets/image-20231230185256794.png)

![image-20231230185620868](/assets/image-20231230185620868.png)

**混淆没了。**

*果然是非常pro-customer的版本更新啊（确信）*

---

### 2. frida-gadget 注入

WSA上注入各种碰壁，不过实体机上即使无Root也可以通过`frida-gadget`修改apk注入

之前用过改dex的方式尝试过，并不成功;挑一个运行时会加载的.so下手：

**注：** 后面目标lib换成了`libFastAES.so`，截图尚未更新

![ ](/assets/image-20231229185227359.png)

![image-20231229192615725](/assets/image-20231229192615725.png)

![image-20231229185311898](/assets/image-20231229185311898.png)

拿apktool打包后签名即可安装到真机

![image-20231229192636829](/assets/image-20231229192636829.png)

**注：** 我的frida跑在WSL上，拿 Windows 机器的adb做后端在 Win/Linux 机器配置分别如下

```cmd
adb.exe -a -P 5555 nodaemon server
```

``` bash
export ADB_SERVER_SOCKET=tcp:192.168.0.2:5555
```

- 用 https://github.com/vfsfitvnm/frida-il2cpp-bridge 测试，未发现问题

![image-20231230233324422](/assets/image-20231230233324422.png)

### 3. IL2CPP 动态调用 runtime

接下来就可以直接调用runtime的东西了

- 回顾上文，貌似API业务逻辑只加了一套对称加密（AES128 CBC）；这里同样如此

![image-20231231002031783](/assets/image-20231231002031783.png)

![image-20231231003122139](/assets/image-20231231003122139.png)

![image-20231231003153856](/assets/image-20231231003153856.png)

![image-20231231003214559](/assets/image-20231231003214559.png)

**总结**：

- `APIManager`为单例实例
- `APIManager.get_Crypt()`取得`Crypt`，其`aesAlgo`即为.NET标准的`AesManaged`

综上，简单写一个脚本：

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

输出如下：

![image-20231231003558106](/assets/image-20231231003558106.png)

![image-20231231001737613](/assets/image-20231231001737613.png)

- 测试解密

![image-20231231002825994](/assets/image-20231231002825994.png)

测试脚本如下：

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

![image-20231231002849347](/assets/image-20231231002849347.png)

不清楚序列化格式是什么，可能是protobuf；还是在下一篇继续调查吧

***SEE YOU SPACE COWBOY...***

### References

https://lief-project.github.io/doc/latest/tutorials/09_frida_lief.html

https://github.com/vfsfitvnm/frida-il2cpp-bridge
