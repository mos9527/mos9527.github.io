---
author: mos9527
title: IL2CPP 逆向（二）： Frida动态调试及取API AES KEY/IV
tags: ["逆向","frida","pjsk","api","project sekai","miku"]
categories: ["逆向"]
series: ["Project SEKAI 逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ./..\..\static
---

# IL2CPP 逆向（二）： Frida动态调试及取API AES KEY/IV

### 0. 版本更新！

*（原来没人玩play版orz*

换到台版之后版本号(2.4.1->2.6.1)有变化，同时上一篇所说的metadata加密手段也换了

同样，拉取一份修复过的dump分析，可见:

![image-20231230182936676](/assets/image-20231230182936676.png)

metadata加载部分不再进行全局加解密，观察二进制也可以发现：

![image-20231230183059925](/assets/image-20231230183059925.png)

2.4.0解密后metadata

![image-20231230183133793](/assets/image-20231230183133793.png)

2.6.1源metadata...多出来8字节？

![image-20231230183234026](/assets/image-20231230183234026.png)

![image-20231230183305449](/assets/image-20231230183305449.png)

加载部分出入不大，尝试删掉8字节直接分析

![image-20231230183414173](/assets/image-20231230183414173.png)

死在读string这一块；到`Il2CppGlobalMetadataHeader.stringOffset`检查一下

![image-20231230183456731](/assets/image-20231230183456731.png)

果然有混淆；这部分应该是明文，而unity的例子中开始应该是`mscrolib`

![image-20231230183601458](/assets/image-20231230183601458.png)

而so中il2cpp对metadata的使用却没加多余步骤；这部分在加载时很可能进行了inplace解密

毕竟暂时不考虑静态篡改，那就直接dump吧！

![image-20231230183727385](/assets/image-20231230183727385.png)

比对

![image-20231230184527296](/assets/image-20231230184527296.png)

果然有且仅有string部分明朗了；继续dump

![image-20231230183345026](/assets/image-20231230183345026.png)

成功！带信息进ida

![image-20231230185256794](/assets/image-20231230185256794.png)

![image-20231230185620868](/assets/image-20231230185620868.png)

我去~~*（初音未来）*~~？？？

方法名部分混淆没了？？？

*~~爱死你了sega!!~~*接下来就好办了！

---

### 1. frida-gadet 注入

看来是找不到有magisk的实体机了..不过不用root也可以调

之前用过改dex的方式尝试过，并不成功;这里通过[尝试修改elf](https://lief-project.github.io/doc/latest/tutorials/09_frida_lief.html)进行

挑一个运行时会加载的.so下手；~~这里选择了libmain.so~~

**注：**だめ！ 改了以后并不能启动；后面目标lib换成了`libFastAES.so`，截图尚未更新..

![ ](/assets/image-20231229185227359.png)

![image-20231229192615725](/assets/image-20231229192615725.png)

![image-20231229185311898](/assets/image-20231229185311898.png)

拿apktool打包后签名即可安装到真机

![image-20231229192636829](/assets/image-20231229192636829.png)

在 WSL 上直接拿 Windows 机器的adb做后端

```cmd
adb.exe -a -P 5555 nodaemon server
```

``` bash
export ADB_SERVER_SOCKET=tcp:192.168.0.2:5555
```

检查一下！

![image-20231230233324422](/assets/image-20231230233324422.png)

dump可行！不过这里并不需要...

### 2. IL2CPP 动态调用 runtime

接下来就可以用 https://github.com/vfsfitvnm/frida-il2cpp-bridge 直接调用runtime的东西了

在此之前，看看API请求的业务流程吧

最重要的一块也许在这里：

![image-20231231002031783](/assets/image-20231231002031783.png)

嗯！还真是就且仅就一套AES加解密？

好吧，找key是一定要做的了...

![image-20231231003122139](/assets/image-20231231003122139.png)

![image-20231231003153856](/assets/image-20231231003153856.png)

![image-20231231003214559](/assets/image-20231231003214559.png)

啊，非常明朗！

ok. 接下来又可以直接跟runtime交互了；所幸`APIManager`是个singleton.

单例的话，直接扫heap就行！

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

ok. 变成字串看看...

![image-20231231001737613](/assets/image-20231231001737613.png)

🤔又是ascii字串？~~*是为方便了我复制粘贴吧*~~

最后抓个包验证一下：

*(byted? bytedance??? 好像logcat里确实看得到他们的东西...*

![image-20231231002825994](/assets/image-20231231002825994.png)

非常简单的解密脚本

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

ok. 有明文了！貌似request/response都是一个密码系统

![image-20231231002849347](/assets/image-20231231002849347.png)

不过内容格式不是json...看起来很像是protobuf？

不过12:40了啊 *（这么早？？？* 数据体分析明天再看吧

fin~
