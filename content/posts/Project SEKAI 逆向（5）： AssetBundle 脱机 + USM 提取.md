---
author: mos9527
title: Project SEKAI 逆向（5）： AssetBundle 脱机 + USM 提取
tags: ["逆向","unity","pjsk","api","project sekai","miku","unity","criware"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ./..\..\static
---

# Project SEKAI 逆向（5）： AssetBundle 脱机 + USM 提取

# 1. 索引

前文提及的`AssetBundleInfo`是从设备提取出来的；假设是已经加载过的所有资源的缓存的话：

（在模拟器）提取该文件时，并没有真正开始除教程Tell Your World以外的任何PV;上一篇所述情况理所应当

![image-20240101204313069](/assets/image-20240101204313069.png)

嗯，只有4MB？

但是在初始化后重现抓包时找到了这玩意

```bash
curl -X GET 'https://184.26.43.87/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online/android21/AssetBundleInfo.json' -H 'Host: lf16-mkovscdn-sg.bytedgame.com' -H 'User-Agent: UnityPlayer/2020.3.32f1 (UnityWebRequest/1.0, libcurl/7.80.0-DEV)' -H 'Accept-Encoding: deflate, gzip' -H 'X-Unity-Version: 2020.3.32f1'
```

![image-20240101204525117](/assets/image-20240101204525117.png)

啊！13MB！

看起来这里是**所有资源**的索引了！解密看看

```bash
 sssekai.exe apidecrypt .\assetbundleinfo .\assetbundleinfo.json
```

查身体模型数...

![image-20240101204751167](/assets/image-20240101204751167.png)

嗯，在这里挖不错了

此外，这里的数据还会多几个field

这是新的

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

这是设备上的

```
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

哦！多出的`downloadPath`可以利用，继续吧...

### 2. CDN？

启动下载后，能抓到一堆这种包：

```bash
curl -X GET 'https://184.26.43.74/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online/android1/actionset/group1?t=20240101203510' -H 'Host: lf16-mkovscdn-sg.bytedgame.com' -H 'User-Agent: UnityPlayer/2020.3.32f1 (UnityWebRequest/1.0, libcurl/7.80.0-DEV)' -H 'Accept-Encoding: deflate, gzip' -H 'X-Unity-Version: 2020.3.32f1'
```

`downloadPath`字段在这里出现了；看起来`https://184.26.43.74/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online` ` 是这里的AB的根路径

而`184.26.43.74`就是cdn了，毕竟

![image-20240101205501465](/assets/image-20240101205501465.png)

cdn的地址看起来是内嵌的；在dump出来的strings中找到了

![image-20240101210240573](/assets/image-20240101210240573.png)

看来可以把这段当常量看待

### 3. 热更新 Cache

考虑pjsk更新量多频率大，每次重新下一边所有数据不是很高效（流量也吃不起啊喂

做一个本地cache很有必要；细节就不在这里说了，请看 https://github.com/mos9527/sssekai/blob/main/sssekai/abcache/__init__.py

尝试拉取全部资源：

![image-20240102003435800](/assets/image-20240102003435800.png)

27GB？？

还好大陆也能裸连orz；下完后看看文件分布吧

### 4. 文件一览

![image-20240102095200320](/assets/image-20240102095200320.png)

![image-20240102095331527](/assets/image-20240102095331527.png)

嗯，3d模型+动画的数据量还是很客观的

(sega不介意我用用吧？)

![image-20240102095619789](/assets/image-20240102095619789.png)

![image-20240102095636123](/assets/image-20240102095636123.png)

其它的话，貌似是音视频文件居多

揭开后可以发现封包格式是日厂喜闻乐见的[CriWare](https://www.criware.com/en/)中间件格式（i.e. USM视频流，HCA音频流）

（话说这东西除了霓虹人还有谁用吗...啊对了，⚪神！

### 5. USM？

![image-20240102112727738](/assets/image-20240102112727738.png)

嗯？连usm头`CRID`都没了

看起来USM资源并不是直接从assetbundle中提取；中间有缓存到文件系统的流程

![image-20240102114924491](/assets/image-20240102114924491.png)

![image-20240102113153597](/assets/image-20240102113153597.png)

果然，在`/sdcard/Android/data/[...]/cache/movies`下有这样的文件

![image-20240102115142365](/assets/image-20240102115142365.png)

哦？这里就是明文了

![image-20240102115207418](/assets/image-20240102115207418.png)

而且用wannacri可以直接demux

![image-20240102115303855](/assets/image-20240102115303855.png)

不过回头看这个`001`文件

![image-20240102115518322](/assets/image-20240102115518322.png)

嘶，原来如此；看起来这些文件需要衔接

![image-20240102125531642](/assets/image-20240102125531642.png)

衔接顺序在`MovieBundleBuildData`可以找到

细节比较繁琐，这里就只放链接了

https://github.com/mos9527/sssekai/blob/main/sssekai/entrypoint/usmdemux.py

使用例：

![image-20240102135738112](/assets/image-20240102135738112.png)

~fin