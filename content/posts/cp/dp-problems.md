---
author: mos9527
lastmod: 2025-05-09T19:37:44.544331
title: 算竞笔记 - 动态规划专题
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

​	

# Preface

“标题说是专题其实只是题单”系列（1/1）

## 树型DP

### 2070D - Tree Jumps

https://codeforces.com/problemset/problem/2070/D

**Idea:** 从深度上至下记录$dp[u]$,$u$为路径结尾点，$\sum dp[i]$即可能数量

## 双指针

### 2028C - Alice's Adventures in Cutting Cake

https://codeforces.com/problemset/problem/2028/C

**Idea:** $dp_1[u]$记录选前$u$个可以满足的最大怪物数量，双指针维护；同时，逆序后可易求后缀$dp_2[v]$记录后$v$个可以满足的最大怪物数量，维护手段一致

```c++
#include "bits/stdc++.h"
using namespace std;
typedef long long ll; typedef double lf; typedef pair<ll, ll> II; typedef vector<ll> vec;
const inline void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0u); cout.tie(0u); }
const lf PI = acos(-1);
const lf EPS = 1e-8;
const ll INF = 1e18;
const ll MOD = 998244353;
const ll DIM = 1e5;
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll t; cin >> t;
    while (t--) {
        ll n,m,v; cin >> n >> m >> v;
        vec a(n + 1); for (ll i = 1; i <= n;i++) cin >> a[i];
        vec pfx = a; for (ll i = 1; i <= n;i++) pfx[i] += pfx[i - 1];
        pfx.push_back(pfx.back());
        vec dp1(n + 2), dp2(n + 2);
        // dp1[i] i之前（不包括）可达最大数目
        for (ll i = 1, j = 1, sm = 0; i <= n; i++) {
            while (j <= n && sm < v) {
                sm += a[j], j++;
                dp1[j] = max(dp1[j], dp1[j - 1]);
            }
            if (sm >= v) dp1[j] = dp1[i] + 1;
            sm -= a[i];
        }
        for (ll i = 1; i <= n + 1;i++) dp1[i] = max(dp1[i], dp1[i - 1]);
        if (dp1[n + 1] < m) {
            cout << -1 << endl;
            continue;
        }
        reverse(a.begin() + 1, a.end());
        // dp2[i] i之后（不包括）可达最大数目
        for (ll i = 1, j = 1, sm = 0; i <= n; i++) {
            while (j <= n && sm < v) {
                sm += a[j], j++;
                dp2[j] = max(dp2[j], dp2[j - 1]);
            }
            if (sm >= v) dp2[j] = dp2[i] + 1;
            sm -= a[i];
        }
        for (ll i = 1; i <= n + 1;i++) dp2[i] = max(dp2[i], dp2[i - 1]);
        reverse(dp2.begin(), dp2.end());
        // 找一段[i,j]段和最大且dp1[i] + dp2[j]>=m
        ll res = 0;
        for (ll i = 1, j = 0, sm = 0; i < n + 1; i++) {
            while (j <= n && dp1[i] + dp2[j + 1] >= m) j++;
            if (dp1[i] + dp2[j] >= m)
                res = max(res, pfx[j] - pfx[i - 1]);
        }
        cout << res << endl;
    }
    return 0;
}

```



## 背包变种

### 子集和 / 方案总数


- https://codeforces.com/contest/2086/problem/D；https://zhuanlan.zhihu.com/p/1891280211527590056；https://oi.wiki/dp/knapsack/#%E6%B1%82%E6%96%B9%E6%A1%88%E6%95%B0
- $c_i$为每种数量
- 观察到每种字符只能出现在奇数位置或偶数位置之一后，考虑分配$c_i$中几部分到奇数位置
- 很显然记总共有$n$个位置可取奇数$odd = \lceil n/2 \rceil$个位置，选择$c_i$中某几个构成集合$S$，使得$\sum_{j \in S} c_j <= odd$
- 计这样的子集个数，实际上为01背包问题；记$i$为背包大小（总位置数量）
  - $ dp_i = \sum_{j=odd}^{c_j} dp_{i - c_j}$
  - 从后往前递推即可
    
  - AC Code
  
  ```c++
  int main() {
      fast_io();
      /* El Psy Kongroo */
      comb::prep();
      ll t; cin >> t;
      while (t--) {
          ll sum = 0;
          vec c(26 + 1); for (ll i = 1; i <= 26; i++) cin >> c[i], sum += c[i], sum %= MOD;
          ll odd = ceil((lf)sum/2);
          ll even = sum - odd;
          vec dp(odd + 1); dp[0] = 1;
          for (ll i = 1; i <= 26; i++)
              if (c[i])  for (ll j = odd; j >= c[i];j--) dp[j] += dp[j - c[i]], dp[j] %= MOD;
          ll ways = dp[odd]; // 1～26中取出数字填充奇数位数方法个数
          // 设选取于奇数方案一定
          // 设奇数位上的数字*类型*为 j_i (1~26)
          // 奇数位方法为 odd! / c[j_1]! * c[j_2]! * c[j_3]! * ...
          // 偶数位方法为 even! / c[j_4]! * c[j_5]! * c[j_6]! * ...
          // 总方法   odd! * even! / c[j_1]! * c[j_2]! ... * c[j_n]!
          // ncomb = odd! * even! / c_1! * c_2! ... * c_26!
          // 带入选数方案即
          // ans = ways * ncomb
          ll ncomb = comb::fac[odd] % MOD * comb::fac[even] % MOD;
          ll icomb = 1; for (ll i = 1;i <= 26;i++) icomb *= comb::ifac[c[i]], icomb %= MOD;
          ll ans = ways % MOD * ncomb % MOD * icomb % MOD;
          cout << ans << endl;
      }
      return 0;
  }
  ```
