---
author: mos9527
lastmod: 2023-12-31T09:52:45.779177+08:00
title: Project SEKAI 逆向（3）： pjsk API中间人攻击 POC
tags: ["逆向","mitm", "pjsk","project sekai","miku"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Project SEKAI 逆向（3）： pjsk API中间人攻击 POC

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

这里和il2cpp关系貌似不大；主要记录一下api包劫持可以用的手段

### 1. 工具

host使用https://github.com/mitmproxy/mitmproxy，

victim上安装https://github.com/NVISOsecurity/MagiskTrustUserCerts后，导入mitmproxy的CA，重启就能把它变成根证书

最后，mitmproxy所用脚本：https://github.com/mos9527/sssekai/blob/main/sssekai/scripts/mitmproxy_sekai_api.py

### 2. 分析

上一篇猜测api的封包用了protobuf；并不然，这里是[MessagePack](https://msgpack.org/index.html)

如此，数据schema和报文在一起；不用额外挖了

用mitmproxy做个poc实时解密转`json`看看：

![image-20231231115914176](/assets/image-20231231115914176.png)

### 3. MITM

*没错，搞这个只是想看看MASTER谱面长什么样...*

---

锁MASTER的字段貌似在这里；尝试直接修改

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

选项可以点亮；但无果：貌似live会在服务端鉴权

![image-20231231121435020](/assets/image-20231231121435020.png)

![image-20231231121528777](/assets/image-20231231121528777.png)

不过live鉴权id貌似不会在开始live中使用；抓包中没有对id的二次引用

考虑可能谱面难度选择只在客户端进行，那么修改上报服务器的难度系数也许能够绕过

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

再次启动：

![image-20231231123020461](/assets/image-20231231123020461.png)

当然，分数上报只会按expert难度进行

---

意外较少，后续应该不会再玩 MITM 相关内容了

接下来...提取游戏asset？在此之前...

***SEE YOU SPACE COWBOY...***

### References

https://msgpack.org/index.html

https://github.com/mitmproxy/mitmproxy

