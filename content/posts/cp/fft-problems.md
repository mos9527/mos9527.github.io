---
author: mos9527
lastmod: 2025-04-17T17:32:29.981339
title: 算竞笔记 - FFT/多项式/数论专题
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Preface

参考主要来自 https://cp-algorithms.com/algebra/fft.html, https://en.wikipedia.org/wiki/Discrete_Fourier_transform, https://oi.wiki/math/poly/fft/

~~为照顾某OJ~~ 本文例程C++标准仅需`11`；**[板子传送门](#reference)**

## 定义

- 多项式$A$的$DFT$即为$A$在各单位根$w_{n, k} = w_n^k = e^{\frac{2 k \pi i}{n}}$之值

$$
\begin{align}
\text{DFT}(a_0, a_1, \dots, a_{n-1}) &= (y_0, y_1, \dots, y_{n-1}) \newline
&= (A(w_{n, 0}), A(w_{n, 1}), \dots, A(w_{n, n-1})) \newline
&= (A(w_n^0), A(w_n^1), \dots, A(w_n^{n-1}))
\end{align}
$$

- $IDFT$ ($InverseDFT$) 即从这些值$(y_0, y_1, \dots, y_{n-1})$恢复多项式$A$的系数

$$
\text{IDFT}(y_0, y_1, \dots, y_{n-1}) = (a_0, a_1, \dots, a_{n-1})
$$

- 单位根有以下性质

  - 积性
    $$
    w^n = 1 \newline
    w^k \ne 1, 0 \lt k \lt n
    $$

  - 所有单位根和为$0$
    $$
    \sum_{k=0}^{n-1} w^k = 0
    $$
    这点利用欧拉公式$e^{ix} = cos x + i\ sin x$看$n$边形对称性很显然

## 应用

考虑两个多项式$A, B$相乘
$$
(A \cdot B)(x) = A(x) \cdot B(x)
$$

- 显然运用$DFT$可得

$$
DFT(A \cdot B) = DFT(A) \cdot DFT(B)
$$

- $A \cdot B$的系数易求

$$
A \cdot B = IDFT(DFT(A \cdot B)) = IDFT(DFT(A) \cdot DFT(B))
$$

## 逆操作（IDFT）

回忆$DFT$的定义
$$
\text{DFT}(a_0, a_1, \dots, a_{n-1}) = (A(w_n^0), A(w_n^1), \dots, A(w_n^{n-1}))
$$
- 写成[矩阵形式](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT)即为

$$
F = \begin{pmatrix}
w_n^0 & w_n^0 & w_n^0 & w_n^0 & \cdots & w_n^0 \newline
w_n^0 & w_n^1 & w_n^2 & w_n^3 & \cdots & w_n^{n-1} \newline
w_n^0 & w_n^2 & w_n^4 & w_n^6 & \cdots & w_n^{2(n-1)} \newline
w_n^0 & w_n^3 & w_n^6 & w_n^9 & \cdots & w_n^{3(n-1)} \newline
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \newline
w_n^0 & w_n^{n-1} & w_n^{2(n-1)} & w_n^{3(n-1)} & \cdots & w_n^{(n-1)(n-1)}
\end{pmatrix} \newline
$$
- 那么$DFT$操作即为

$$
F\begin{pmatrix}
a_0 \newline a_1 \newline a_2 \newline a_3 \newline \vdots \newline a_{n-1}
\end{pmatrix} = \begin{pmatrix}
y_0 \newline y_1 \newline y_2 \newline y_3 \newline \vdots \newline y_{n-1}
\end{pmatrix}
$$
- 化简有

$$
y_k = \sum_{j=0}^{n-1} a_j w_n^{k j},
$$

其中范德蒙德阵$M$行列各项正交，[可做出结论](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT):

$$
F^{-1} = \frac{1}{n} F^\star, F_{i,j}^\star = \overline{F_{j,i}}
$$

既有
$$
F^{-1} = \frac{1}{n}
\begin{pmatrix}
w_n^0 & w_n^0 & w_n^0 & w_n^0 & \cdots & w_n^0 \newline
w_n^0 & w_n^{-1} & w_n^{-2} & w_n^{-3} & \cdots & w_n^{-(n-1)} \newline
w_n^0 & w_n^{-2} & w_n^{-4} & w_n^{-6} & \cdots & w_n^{-2(n-1)} \newline
w_n^0 & w_n^{-3} & w_n^{-6} & w_n^{-9} & \cdots & w_n^{-3(n-1)} \newline
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \newline
w_n^0 & w_n^{-(n-1)} & w_n^{-2(n-1)} & w_n^{-3(n-1)} & \cdots & w_n^{-(n-1)(n-1)}
\end{pmatrix}
$$
- 那么$IDFT$操作即为

$$
\begin{pmatrix}
a_0 \newline a_1 \newline a_2 \newline a_3 \newline \vdots \newline a_{n-1}
\end{pmatrix} = F^{-1} \begin{pmatrix}
y_0 \newline y_1 \newline y_2 \newline y_3 \newline \vdots \newline y_{n-1}
\end{pmatrix}
$$
- 化简有

$$
a_k = \frac{1}{n} \sum_{j=0}^{n-1} y_j w_n^{-k j}
$$
### 结论

- **注意到$w_i$使用共轭即为$n \cdot \text{IDFT}$**
- 实现中稍作调整即可同时实现$DFT,IDFT$操作；接下来会用到

## 实现（FFT）

朴素包络时间复杂度为$O(n^2)$，这里不做阐述

$FFT$的过程如下

- 令$A(x) = a_0 x^0 + a_1 x^1 + \dots + a_{n-1} x^{n-1}$, 按奇偶拆成两个子多项式

$$
\begin{align}
A_0(x) &= a_0 x^0 + a_2 x^1 + \dots + a_{n-2} x^{\frac{n}{2}-1} \newline
A_1(x) &= a_1 x^0 + a_3 x^1 + \dots + a_{n-1} x^{\frac{n}{2}-1}
\end{align}
$$
- 显然有

$$
A(x) = A_0(x^2) + x A_1(x^2).
$$
- 设$\left(y_k^0\right)_{k=0}^{n/2-1} = \text{DFT}(A_0), \left(y_k^1\right)_{k=0}^{n/2-1} = \text{DFT}(A_1)$，前$\frac{n}{2}$项即为

$$
y_k = y_k^0 + w_n^k y_k^1, \quad k = 0 \dots \frac{n}{2} - 1.
$$
- 对后半 $\frac{n}{2}$ 有

$$
\begin{align}
y_{k+n/2} &= A\left(w_n^{k+n/2}\right) \newline
&= A_0\left(w_n^{2k+n}\right) + w_n^{k + n/2} A_1\left(w_n^{2k+n}\right) \newline
&= A_0\left(w_n^{2k} w_n^n\right) + w_n^k w_n^{n/2} A_1\left(w_n^{2k} w_n^n\right) \newline
&= A_0\left(w_n^{2k}\right) - w_n^k A_1\left(w_n^{2k}\right) \newline
&= y_k^0 - w_n^k y_k^1
\end{align}
$$

- 即$y_{k+n/2} = y_k^0 - w_n^k y_k^1$，形式上非常接近$y_k$。综上：

$$
\begin{align}
y_k &= y_k^0 + w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1, \newline
y_{k+n/2} &= y_k^0 - w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1.
\end{align}
$$

该式即为所谓 **“蝶形优化”**

### 结论

- 很显然合并代价是$O(n)$；由$T_{\text{DFT}}(n) = 2 T_{\text{DFT}}\left(\frac{n}{2}\right) + O(n)$则知$FFT$可在$O(nlogn)$时间内解决问题
- 归并实现也将很简单

### Code （归并）

又称 **库利-图基演算法(Cooley-Tukey algorithm)**；分治解决

- 若使用`std::complex`实现$w_n$可以直接用[`std::exp`自带特化](https://en.cppreference.com/w/cpp/numeric/complex/exp)求得$w_n = e^{\frac{2\pi i}{n}}$
- 或者利用欧拉公式$e^{ix} = cos x + i\ sin x$可构造`Complex w_n{ .real = cos(2 * PI / n), .imag = sin(2 * PI / n) }`
- 结合之前所述的$DFT$, $IDFT$关系，使用$w_n = -e^{\frac{2\pi i}{n}}$并除$n$即求$IDFT$
- **时间复杂度$O(n\log n)$**，由于对半分后归并，**空间复杂度$O(n)$**

```c++
void FFT(cvec& A, bool invert) {
    ll n = A.size(); 
    if (n == 1) return;
	cvec A0(n / 2), A1(n / 2);
	for (ll i = 0; i < n / 2; i++) 
        A0[i] = A[i * 2], A1[i] = A[i * 2 + 1];
  FFT(A0, invert), FFT(A1, invert);
  Complex w_n = exp(Complex{ 0, 2 * PI / n });
  if (invert) 
    w_n = conj(w_n);
  Complex w_k = Complex{ 1, 0 };
	for (ll k = 0; k < n / 2; k++) {
		A[k] = A0[k] + w_k * A1[k];
		A[k + n / 2] = A0[k] - w_k * A1[k];
    // 注意：除 log2(n) 次 2 即除 2^log2(n) = n
    if (invert) 
      A[k] /= 2, A[k + n / 2] /= 2;
		w_k *= w_n;
	}   
}
void FFT(cvec& a) { FFT(a, false); }
void IFFT(cvec& y) { FFT(y, true); }
```

### Code （倍增）

归并法带来的额外空间其实可以优化掉——接下来介绍倍增法递推解决。

- 观察归并中最后回溯的顺序（以 $n=8$为例）
  -   初始序列为 $\{x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7\}$
  -   一次二分之后 $\{x_0, x_2, x_4, x_6\},\{x_1, x_3, x_5, x_7 \}$
  -   两次二分之后 $\{x_0,x_4\} \{x_2, x_6\},\{x_1, x_5\},\{x_3, x_7 \}$
  -   三次二分之后 $\{x_0\}\{x_4\}\{x_2\}\{x_6\}\{x_1\}\{x_5\}\{x_3\}\{x_7 \}$

- 注意力足够的话可以发现规律如下

```python
In [17]: [int(bin(i)[2:].rjust(3,'0')[::-1],2) for i in range(8)]
Out[17]: [0, 4, 2, 6, 1, 5, 3, 7]

In [18]: [bin(i)[2:].rjust(3,'0')[::-1] for i in range(8)]
Out[18]: ['000', '100', '010', '110', '001', '101', '011', '111']

In [19]: [bin(i)[2:].rjust(3,'0') for i in range(8)]
Out[19]: ['000', '001', '010', '011', '100', '101', '110', '111']
```

- 即二进制倒序（对称），记该倒序为 $R(x)$

```c++
auto R = [n](ll x) {
    ll msb = ceil(log2(n)), res = 0;
    for (ll i = 0;i < msb;i++)
        if (x & (1 << i))
            res |= 1 << (msb - 1 - i);
    return res;
};
```

- 从下至上，以长度为$2,4,6,\cdots,n$递推，保持该顺序即可完成归并法所完成的任务
- 又因为对称，调整顺序也可在$O(n)$内完成；**时间复杂度$O(n\log n)$，空间复杂度$O(1)$**

```c++
void FFT(cvec& A, bool invert) {
    ll n = A.size();
    auto R = [n](ll x) {
        ll msb = ceil(log2(n)), res = 0;
        for (ll i = 0;i < msb;i++)
            if (x & (1 << i))
                res |= 1 << (msb - 1 - i);
        return res;
    };
    // Resort
    for (ll i = 0;i < n;i++)
        if (i < R(i))
            swap(A[i], A[R(i)]);
    // 从下至上n_i = 2, 4, 6,...,n直接递推
    for (ll n_i = 2;n_i <= n;n_i <<= 1) {
        Complex w_n = exp(Complex{ 0, 2 * PI / n_i });
        if (invert) w_n = conj(w_n);
        for (ll i = 0;i < n;i += n_i) {
            Complex w_k = Complex{ 1, 0 };
            for (ll j = 0;j < n_i / 2;j++) {
                Complex u = A[i + j], v = A[i + j + n_i / 2] * w_k;
                A[i + j] = u + v;
                A[i + j + n_i / 2] = u - v;
                if (invert)
                    A[i+j] /= 2, A[i+j+n_i/2] /= 2;
                w_k *= w_n;
            }
        }
    }
}
void FFT(cvec& a) { FFT(a, false); }
void IFFT(cvec& y) { FFT(y, true); }
```

## 数论变换 （NTT）

虚数域内计算难免精度问题；数字越大误差越大且因为$exp$（或$sin, cos$）的使用极难修正。以下介绍数论变换（或快速数论变换）以允许在模数域下完成绝对正确的$O(nlogn)$包络。

- 在质数$p$, $F={\mathbb {Z}/p}$域下进行的DFT；注意到单位根的性质在模数下保留

- 同时显然的，有$$(w_n^m)^2 = w_n^n = 1 \pmod{p},  m = \frac{n}{2}$$；利用该性质我们可以利用快速幂求出$w_n^k$

- 当然，我们需要找到这样$g_n^n \equiv 1 \mod p$的$g$，使得$g_n$等效于$w_n$


### 原根

> 以下内容摘自：https://cp-algorithms.com/algebra/primitive-root.html#algorithm-for-finding-a-primitive-root, 

定义：**对任意$a$且存在$a$, $n$互质，且 $g^k \equiv a \mod n$，则称 $g$ 为模 $n$ 的原根。**
结论：**$n$的原根$g$,$g^k \equiv 1 \pmod n$， $k=\phi(n)$为$k$的最小解**
下面介绍一种求原根的算法：

  - 欧拉定义：若 $\gcd(a, n) = 1$，则 $a^{\phi(n)} \equiv 1 \pmod{n}$
  - 对指数$p$, 朴素解法即为$O(n^2)$时间检查$g^d, d \in [0,\phi(n)] \not\equiv 1 \pmod n$

  - 存在这样的$O(\log \phi (n) \cdot \log n)$解法：
    - 找到$\phi(n)$因数$p_i \in P$，检查$g \in [1, n]$
    - 对所有$p_i \in P$, $g ^ { \frac {\phi (n)} {p_i}} \not\equiv 1\pmod n $，此根即为一原根
  - 证明请参见原文

```c++
#include "bits/stdc++.h"
using namespace std;
typedef long long ll; typedef vector<ll> vec; 
ll binpow_mod(ll a, ll b, ll m) {
    a %= m;
    ll res = 1;
    while (b > 0) {
        if (b & 1) res = (__int128)res * a % m;
        a = (__int128)a * a % m;
        b >>= 1;
    }
    return res;
}
ll min_primitive_root(ll p) {
    vec fac; ll phi = p - 1, n = phi;
    for (ll i = 2; i * i <= n; i++)
        if (n % i == 0) {
            fac.push_back(i);
            while (n % i == 0) n /= i;
        }
    if (n != 1) fac.push_back(n);
    for (ll r = 2; r <= p; r++) {
        bool ok = true;
        for (ll i = 0; ok && i < fac.size(); i++)
            ok &= binpow_mod(r, phi / fac[i], p) != 1;
        if (ok) return r;
    }
    return -1;
}
// min_primitive_root(754974721) = 11
// min_primitive_root(998244353) = 3
// min_primitive_root(7340033) = 3
```

### 实现（倍增）

综上，有质数$p$及其原根$g$对即可做到模数域下的单位根性质；常用的有 ($p=7 \times 17 \times 2^{23}+1=998244353, g=3$,$p=7 \times 2^{20} + 1 =7340033$)

这些数的欧拉函数满足$\phi(p) = p - 1 = c \times 2^k$形式，回忆欧拉函数$g^{p-1} \equiv 1 \pmod n$，很显然这很适合接下来我们要做的事情：遍历到长度$n_i$时，$w_{n_i} = e^{\frac{2\pi}{n_i}}$即等效于$g^{\frac{p-1}{n_i}}$。由于$n_i$ 倍增，$\frac{p-1}{n_i}$即为简单移位，同时整数除法也将无误差。


### Code （倍增）

```c++
void NTT(vec& A, ll p, ll g, bool invert) {
  ll n = A.size();
  auto R = [n](ll x) {
      ll msb = ceil(log2(n)), res = 0;
      for (ll i = 0;i < msb;i++)
          if (x & (1 << i))
              res |= 1 << (msb - 1 - i);
      return res;
  };
  // Resort
  for (ll i = 0;i < n;i++)
      if (i < R(i)) swap(A[i], A[R(i)]);
  // 从下至上n_i = 2, 4, 6,...,n直接递推
  ll inv_2 = binpow_mod(2, p - 2, p);
  for (ll n_i = 2;n_i <= n;n_i <<= 1) {
      ll w_n = binpow_mod(g, (p - 1) / n_i, p);
      if (invert)
          w_n = binpow_mod(w_n, p - 2, p);
      for (ll i = 0;i < n;i += n_i) {
          ll w_k = 1;
          for (ll j = 0;j < n_i / 2;j++) {
              ll u = A[i + j], v = A[i + j + n_i / 2] * w_k;
              A[i + j] = (u + v + p) % p;
              A[i + j + n_i / 2] = (u - v + p) % p;
              if (invert) {
                  A[i + j] = A[i + j] * inv_2 % p;
                  A[i + j + n_i / 2] = A[i + j + n_i / 2] * inv_2 % p;
              }
              w_k = w_k * w_n % p;
          }
      }
  }
}
void FFT(vec& a) { NTT(a,998244353, 3, false); }
void IFFT(vec& y) { NTT(y, 998244353,3, true); }
```

## Reference

本文所提及的$\text{DFT/FFT/(F)NTT}$魔术总结如下，即插即用。为准确起见，API以`DFT(...), IDFT(...)`命名。

- https://acm.hdu.edu.cn/showproblem.php?pid=1402

```c++
#include "bits/stdc++.h"
using namespace std;
typedef long long ll; typedef double lf; typedef pair<ll, ll> II; typedef vector<ll> vec;
const inline void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0u); cout.tie(0u); }
const lf PI = acos(-1);
const lf EPS = 1e-8;
const ll INF = 1e18;
const ll MOD = 1e9 + 7;
const ll DIM = 1e5;
ll binpow_mod(ll a, ll b, ll m = MOD) {
    a %= m;
    ll res = 1;
    while (b > 0) {
        if (b & 1) res = (__int128)res * a % m;
        a = (__int128)a * a % m;
        b >>= 1;
    }
    return res;
}
namespace Poly {
    using Real = lf;
    using Complex = complex<lf>;
    using CVec = vector<Complex>;
    using RVec = vector<Real>;
    using IVec = vec;
    const ll MOD = 998244353, MOD_proot = 3;
    // 快速傅里叶变换
    inline CVec& FFT(CVec& a, bool invert) {
        ll n = a.size();
        auto R = [n](ll x) {
            ll msb = ceil(log2(n)), res = 0;
            for (ll i = 0;i < msb;i++)
                if (x & (1 << i))
                    res |= 1 << (msb - 1 - i);
            return res;
        };
        // Resort
        for (ll i = 0;i < n;i++)
            if (i < R(i))
                swap(a[i], a[R(i)]);
        // 从下至上n_i = 2, 4, 6,...,n直接递推
        for (ll n_i = 2;n_i <= n;n_i <<= 1) {
            Complex w_n = exp(Complex{ 0, 2 * PI / n_i });
            if (invert) w_n = conj(w_n);
            for (ll i = 0;i < n;i += n_i) {
                Complex w_k = Complex{ 1, 0 };
                for (ll j = 0;j < n_i / 2;j++) {
                    Complex u = a[i + j], v = a[i + j + n_i / 2] * w_k;
                    a[i + j] = u + v;
                    a[i + j + n_i / 2] = u - v;
                    if (invert)
                        a[i+j] /= 2, a[i+j+n_i/2] /= 2;
                    w_k *= w_n;
                }
            }
        }
        return a;
    }
    //（快速）数论变换
    inline IVec& NTT(IVec& a, ll p, ll g, bool invert) {
        ll n = a.size();
        auto R = [n](ll x) {
            ll msb = ceil(log2(n)), res = 0;
            for (ll i = 0;i < msb;i++)
                if (x & (1 << i))
                    res |= 1 << (msb - 1 - i);
            return res;
        };
        // Resort
        for (ll i = 0;i < n;i++)
            if (i < R(i)) swap(a[i], a[R(i)]);
        // 从下至上n_i = 2, 4, 6,...,n直接递推
        ll inv_2 = binpow_mod(2, p - 2, p);
        for (ll n_i = 2;n_i <= n;n_i <<= 1) {
            ll w_n = binpow_mod(g, (p - 1) / n_i, p);
            if (invert)
                w_n = binpow_mod(w_n, p - 2, p);
            for (ll i = 0;i < n;i += n_i) {
                ll w_k = 1;
                for (ll j = 0;j < n_i / 2;j++) {
                    ll u = a[i + j], v = a[i + j + n_i / 2] * w_k;
                    a[i + j] = (u + v + p) % p;
                    a[i + j + n_i / 2] = (u - v + p) % p;
                    if (invert) {
                        a[i + j] = (a[i + j] * inv_2 % p + p) % p;
                        a[i + j + n_i / 2] = (a[i + j + n_i / 2] * inv_2 % p + p) % p;
                    }
                    w_k = w_k * w_n % p;
                }
            }
        }
        return a;
    }
    // 虚数域
    inline CVec& DFT(CVec& a) { return FFT(a, false); }
    inline CVec& IDFT(CVec& a) { return FFT(a, true); }
    // 模数域
    inline IVec& DFT(IVec& a, ll p=MOD, ll g=MOD_proot) { return NTT(a, p, g, false); }
    inline IVec& IDFT(IVec& a, ll p=MOD, ll g=MOD_proot) { return NTT(a, p, g, true); }
    // 工具
    inline RVec as_real(CVec const& a) {
        RVec res(a.size());
        for (ll i = 0;i < a.size();i++)
            res[i] = a[i].real();
        return res;
    }
    inline CVec as_complex(RVec const& a) {
        return CVec(a.begin(), a.end());
    }
    void normalize(IVec& a, ll radiax) {
        for (ll i = 0;i < a.size() - 1;i++)
            a[i + 1] += a[i] / radiax,
            a[i] %= radiax;
    }
    void normalize(RVec& a, ll radiax) {
        for (ll i = 0;i < a.size() - 1;i++)
            a[i + 1] += (ll)a[i] / radiax,
            a[i] = (ll)a[i] % radiax;
    }
    // 多项式乘法
    template<typename T> ll mul_poly(T& a, T& b) {
        ll n = a.size() + b.size();
        n = ceil(log2(n)), n = 1ll << n;
        a.resize(n), b.resize(n);
        DFT(a), DFT(b);
        for (ll i = 0;i < n;i++)
            a[i] *= b[i];
        IDFT(a);
        return n;
    }
    ll mul_poly(RVec& a, RVec& b) {
        CVec a_c = as_complex(a), b_c = as_complex(b);
        ll n = mul_poly(a_c, b_c);
        a = as_real(a_c);
        return n;
    }
}
int main() {
    string a,b;
    while (cin >> a >> b)
    {
        {
            Poly::IVec A(a.size()), B(b.size());
            for (ll i = 0;i < a.size();i++)
                A[i] = a[a.size() - 1 - i] - '0';
            for (ll i = 0;i < b.size();i++)
                B[i] = b[b.size() - 1 - i] - '0';
            ll len = Poly::mul_poly(A, B);
            Poly::normalize(A, 10u);
            for (ll i = len - 1, flag = 0;i >= 0;i--) {
                flag |= A[i] != 0;
                if (flag || i == 0)
                    cout << (ll)A[i];
            }
            cout << endl;
        }
    }
}
```

