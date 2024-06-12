---
author: mos9527
lastmod: 2024-01-01T20:39:55.836773+08:00
title: Project SEKAI 逆向（5）： AssetBundle 脱机 + USM 提取
tags: ["逆向","unity","pjsk","api","project sekai","miku","unity","criware"]
categories: ["Project SEKAI 逆向", "逆向"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Project SEKAI 逆向（5）： AssetBundle 脱机 + USM 提取

- 分析variant：世界計劃 2.6.1 （Google Play 台服）

### 1. AssetBundleInfo

前文提及的`AssetBundleInfo`是从设备提取出来的；假设是已经加载过的所有资源的缓存的话：

- 在刚刚完成下载的设备上提取该文件时，该文件 4MB

![image-20240101204313069](/assets/image-20240101204313069.png)

- 但是在初始化后重现抓包时发现的该文件为 **13MB**

```bash
curl -X GET 'https://184.26.43.87/obj/sf-game-alisg/gdl_app_5245/AssetBundle/2.6.0/Release/online/android21/AssetBundleInfo.json' -H 'Host: lf16-mkovscdn-sg.bytedgame.com' -H 'User-Agent: UnityPlayer/2020.3.32f1 (UnityWebRequest/1.0, libcurl/7.80.0-DEV)' -H 'Accept-Encoding: deflate, gzip' -H 'X-Unity-Version: 2020.3.32f1'
```

![image-20240101204525117](/assets/image-20240101204525117.png)

- 推测设备上文件为已缓存资源库，而这里的即为全量资源集合；尝试dump

```bash
 sssekai apidecrypt .\assetbundleinfo .\assetbundleinfo.json
```

- 查身体模型数看看吧

![image-20240101204751167](/assets/image-20240101204751167.png)

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

![image-20240101205501465](/assets/image-20240101205501465.png)

- cdn的地址看起来是内嵌的；在dump出来的strings中：

![image-20240101210240573](/assets/image-20240101210240573.png)

### 3. 热更新 Cache

考虑pjsk更新频率大，每次重新下所有数据不是很高效

做一个本地cache动机充分；细节就不在这里说了，请看 https://github.com/mos9527/sssekai/blob/main/sssekai/abcache/__init__.py

尝试拉取全部资源，貌似需要*27GB*

![image-20240102003435800](/assets/image-20240102003435800.png)

### 4. 文件一览

- 在 WinDirStat 中查看分布

![image-20240102095200320](/assets/image-20240102095200320.png)

- 动画资源：

![image-20240102095331527](/assets/image-20240102095331527.png)

- VO

![image-20240102095619789](/assets/image-20240102095619789.png)

![image-20240102095636123](/assets/image-20240102095636123.png)

其它的话，貌似是音视频文件居多

揭开后可以发现封包格式是[CriWare](https://www.criware.com/en/)中间件格式（i.e. USM视频流，HCA音频流）

### 5. USM 提取

*动机：应该很简单orz*

---

![image-20240102112727738](/assets/image-20240102112727738.png)

- 没有 Magic `CRID`

  回到IDA，看起来USM资源并不是直接从assetbundle中提取；中间有缓存到文件系统的流程

![image-20240102114924491](/assets/image-20240102114924491.png)

![image-20240102113153597](/assets/image-20240102113153597.png)

- 果然，在`/sdcard/Android/data/[...]/cache/movies`下有这样的文件

![image-20240102115142365](/assets/image-20240102115142365.png)

![image-20240102115207418](/assets/image-20240102115207418.png)

- 而且用[WannaCri](https://github.com/donmai-me/WannaCRI)可以直接demux，没有额外密钥

![image-20240102115303855](/assets/image-20240102115303855.png)

- 回顾asset中USM文件

![image-20240102115518322](/assets/image-20240102115518322.png)

- 利用`MovieBundleBuildData`猜测可以拼接出源文件

![image-20240102125531642](/assets/image-20240102125531642.png)

**脚本：**https://github.com/mos9527/sssekai/blob/main/sssekai/entrypoint/usmdemux.py

![image-20240102135738112](/assets/image-20240102135738112.png)

***SEE YOU SPACE COWBOY...***

### References

https://github.com/mos9527/sssekai

https://github.com/donmai-me/WannaCRI