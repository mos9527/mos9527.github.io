---
author: mos9527
lastmod: 2025-04-17T12:14:43.116834
title: 算竞笔记 - FFT/多项式类型合集
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Preface

**注：**摘自 https://cp-algorithms.com/algebra/fft.html, https://en.wikipedia.org/wiki/Discrete_Fourier_transform, https://oi.wiki/math/poly/fft/

## 定义

- 多项式$A$的$DFT$即为$A$在各单位根$w_{n, k} = w_n^k = e^{\frac{2 k \pi i}{n}}$之值

$$
\begin{align}
\text{DFT}(a_0, a_1, \dots, a_{n-1}) &= (y_0, y_1, \dots, y_{n-1}) \\
&= (A(w_{n, 0}), A(w_{n, 1}), \dots, A(w_{n, n-1})) \\
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
    w^n = 1 \\
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
w_n^0 & w_n^0 & w_n^0 & w_n^0 & \cdots & w_n^0 \\
w_n^0 & w_n^1 & w_n^2 & w_n^3 & \cdots & w_n^{n-1} \\
w_n^0 & w_n^2 & w_n^4 & w_n^6 & \cdots & w_n^{2(n-1)} \\
w_n^0 & w_n^3 & w_n^6 & w_n^9 & \cdots & w_n^{3(n-1)} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
w_n^0 & w_n^{n-1} & w_n^{2(n-1)} & w_n^{3(n-1)} & \cdots & w_n^{(n-1)(n-1)}
\end{pmatrix} \\
$$
- 那么$DFT$操作即为

$$
F\begin{pmatrix}
a_0 \\ a_1 \\ a_2 \\ a_3 \\ \vdots \\ a_{n-1}
\end{pmatrix} = \begin{pmatrix}
y_0 \\ y_1 \\ y_2 \\ y_3 \\ \vdots \\ y_{n-1}
\end{pmatrix}
$$
- 化简有

$$
y_k = \sum_{j=0}^{n-1} a_j w_n^{k j},
$$
其中范德蒙德阵$M$行列各项正交，[可做出结论](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT) $F^{-1} = \frac{1}{n}F^*$, $F_{i,j}^* = \overline{F_{j,i}}$，既有
$$
F^{-1} = \frac{1}{n}
\begin{pmatrix}
w_n^0 & w_n^0 & w_n^0 & w_n^0 & \cdots & w_n^0 \\
w_n^0 & w_n^{-1} & w_n^{-2} & w_n^{-3} & \cdots & w_n^{-(n-1)} \\
w_n^0 & w_n^{-2} & w_n^{-4} & w_n^{-6} & \cdots & w_n^{-2(n-1)} \\
w_n^0 & w_n^{-3} & w_n^{-6} & w_n^{-9} & \cdots & w_n^{-3(n-1)} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
w_n^0 & w_n^{-(n-1)} & w_n^{-2(n-1)} & w_n^{-3(n-1)} & \cdots & w_n^{-(n-1)(n-1)}
\end{pmatrix}
$$
- 那么$IDFT$操作即为

$$
\begin{pmatrix}
a_0 \\ a_1 \\ a_2 \\ a_3 \\ \vdots \\ a_{n-1}
\end{pmatrix} = F^{-1} \begin{pmatrix}
y_0 \\ y_1 \\ y_2 \\ y_3 \\ \vdots \\ y_{n-1}
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
A_0(x) &= a_0 x^0 + a_2 x^1 + \dots + a_{n-2} x^{\frac{n}{2}-1} \\
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
- 对后半$\frac{n}{2}$有$$y_{k+n/2} = A\left(w_n^{k+n/2}\right) = A_0\left(w_n^{2k+n}\right) + w_n^{k + n/2} A_1\left(w_n^{2k+n}\right) = A_0\left(w_n^{2k} w_n^n\right) + w_n^k w_n^{n/2} A_1\left(w_n^{2k} w_n^n\right) = A_0\left(w_n^{2k}\right) - w_n^k A_1\left(w_n^{2k}\right) =  y_k^0 - w_n^k y_k^1$$

- **即$$y_{k+n/2} = y_k^0 - w_n^k y_k^1$$，形式上非常接近$y_k$。**综上：

$$
\begin{align}
y_k &= y_k^0 + w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1, \\
y_{k+n/2} &= y_k^0 - w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1.
\end{align}
$$

该式即为所谓**“蝶形优化”**

### 结论

- 很显然合并代价是$O(n)$；由$T_{\text{DFT}}(n) = 2 T_{\text{DFT}}\left(\frac{n}{2}\right) + O(n)$则知$FFT$可在$O(nlogn)$时间内解决问题
- 归并实现也将很简单

### Code （归并）

- 若使用`std::complex`实现$w_n$可以直接用[`std::exp`自带特化](https://en.cppreference.com/w/cpp/numeric/complex/exp)求得$w_n = e^{\frac{2\pi i}{n}}$
- 或者利用欧拉公式$e^{ix} = cos x + i\ sin x$可构造`Complex w_n{ .real = cos(2 * PI / n), .imag = sin(2 * PI / n) }`
- 结合之前所述的$DFT$, $IDFT$关系，使用$w_n = -e^{\frac{2\pi i}{n}}$并除$n$即求$IDFT$

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

- 完整实现如下

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

- 在质数$p$, $F={\mathbb {Z}/p}$域下进行的DFT；注意到单位根的性质在模数下保留

- 同时显然的，有$$(w_n^2)^m = w_n^n = 1 \pmod{p},  m = \frac{n}{2}$$；利用该性质我们可以利用快速幂求出$w_n^k$

- 当然，我们需要找到这样$g_n^n \equiv 1 \mod p$的$g$，即$g_n$等效于$w_n$

  - 对于$p = c2^k + 1$类的数字，其原根$g$满足该条件且$g^{cn} \equiv 1 \mod p$

    - 见 https://cp-algorithms.com/algebra/fft.html#number-theoretic-transform, https://oi.wiki/math/poly/ntt
  - 求$g_n = g^c \pmod p$即可作为$w_n$等价

### $g_n$计算

以下方法可找出一个符合该性质的$g_n$

> 以下内容摘自：https://cp-algorithms.com/algebra/primitive-root.html#algorithm-for-finding-a-primitive-root

  **对任意$a$, $a$, $n$互质，且 $g^k \equiv a \mod n$，则称 $g$ 为模 $n$ 的原根。**
  结论：**$n$的原根$g$, $k=\phi(n)$为$g^k \equiv 1 \pmod n$，$k$的最小解**

  - 欧拉定义：若 $\gcd(a, n) = 1$，则 $a^{\phi(n)} \equiv 1 \pmod{n}$
  - 对指数$p$, 朴素解法即为$O(n^2)$时间检查$g^d, d \in [0,\phi(n)] \not\equiv 1 \pmod n$

  - 下面介绍$O(\log \phi (n) \cdot \log n)$解法
    - 找到$\phi(n)$因数$p_i \in P$，检查$g \in [1, n]$
    - 对所有$p_i \in P$, $g ^ { \frac {\phi (n)} {p_i}} \not\equiv 1\pmod n $，此根即为一原根
  - 证明请参见原文

```c++
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
ll ntt_find_root(ll c, ll k) {
    ll p = c * (1ll << k) + 1;
    cout << "c=" << c << endl;
    cout << "k=" << k << endl;
    cout << "p=" << p << endl;
    ll g = min_primitive_root(p);
    cout << "g=" << g << endl;
    ll g_n = binpow_mod(g, c, p);
    ll k_2 = binpow_mod(2, k, p);
    ll g_n_n = binpow_mod(g_n, k_2, p);
    cout << "g_n^{2^k} mod p=" << g_n_n << endl;
    cout << "g_n=" << g_n << endl;
    return g_n;
}
int main() {
    ntt_find_root(7, 20);
    cout << "---" << endl;
    ntt_find_root(9 * 5, 24);
    return 0;
}
/*
c=7
k=20
p=7340033
g=3
g_n^{2^k} mod p=1
g_n=2187
---
c=45
k=24
p=754974721
g=11
g_n^{2^k} mod p=1
g_n=739831874
*/
```

### Code （倍增）

```c++
void NTT(vec& A, ll c, ll k, ll g_n, bool invert) {
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
  ll p = c * (1ll << k) + 1;
  ll root = g_n, root_inv = binpow_mod(g_n, p - 2, p);
  ll inv_2 = binpow_mod(2, p - 2, p);
  for (ll n_i = 2;n_i <= n;n_i <<= 1) {
      ll w_n = invert ? root_inv : root;
      for (ll i = n_i; i < 1ll << k; i <<= 1)
          w_n = 1LL * w_n * w_n % p;
      for (ll i = 0;i < n;i += n_i) {
          ll w_k = 1;
          for (ll j = 0;j < n_i / 2;j++) {
              ll u = A[i + j], v = A[i + j + n_i / 2] * w_k;
              A[i + j] = (u + v + p) % p;
              A[i + j + n_i / 2] = (u - v + p) % p;
              if (invert)
                  A[i+j] *= inv_2, A[i+j] %= p, A[i+j+n_i/2] /= 2, A[i+j+n_i/2] %= p;
              w_k *= w_n, w_k %= p;
          }
      }
  }
}
```

## 完整多项式魔术板子

```c++
#include "bits/stdc++.h"
using namespace std;
#define PRED(T,X) [&](T const& lhs, T const& rhs) {return X;}
typedef long long ll; typedef double lf; typedef complex<lf> Complex;
const lf PI = acos(-1);
#ifdef __SIZEOF_INT128__
typedef __int128_t i128;
#endif
typedef pair<ll, ll> II; typedef vector<ll> vec; typedef vector<Complex> cvec;
template<size_t size> using arr = array<ll, size>;
const static void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0); }
const static ll lowbit(const ll x) { return x & -x; }
const ll DIM = 1e5;
const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const lf EPS = 1e-8;
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
void NTT(vec& A, ll c, ll k, ll g_n, bool invert) {
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
    ll p = c * (1ll << k) + 1;
    ll root = invert ? binpow_mod(g_n, p - 2, p) : g_n;
    ll inv_2 = binpow_mod(2, p - 2, p);
    for (ll n_i = 2;n_i <= n;n_i <<= 1) {
        ll w_n = root;
        for (ll i = n_i;i < (1ll << k);i <<= 1)
            w_n *= w_n, w_n %= p;
        for (ll i = 0;i < n;i += n_i) {
            ll w_k = 1;
            for (ll j = 0;j < n_i / 2;j++) {
                ll u = A[i + j], v = A[i + j + n_i / 2] * w_k % p;
                A[i + j] = (u + v + p) % p;
                A[i + j + n_i / 2] = (u - v + p) % p;
                if (invert)
                    A[i+j] *= inv_2, A[i+j] %= p, A[i+j+n_i/2] /= 2, A[i+j+n_i/2] %= p;
                w_k *= w_n, w_k %= p;
            }
        }
    }
}
void FFT(vec& a) { NTT(a, 7, 20, 5, false); }
void IFFT(vec& y) { NTT(y, 7, 20, 5, true); }
vec multiply(vec const& a, vec const& b) {
    vec fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < a.size() + b.size())
        n <<= 1;
    fa.resize(n);
    fb.resize(n);

    FFT(fa);
    FFT(fb);
    for (int i = 0; i < n; i++)
        fa[i] *= fb[i];
    IFFT(fa);

    vec result(n);
    for (int i = 0; i < n; i++)
        result[i] = fa[i];
    // normalize
    for (int i = 0; i < n - 1; i++) {
        result[i + 1] += result[i] / 10;
        result[i] %= 10;
    }
    return result;
}
int main() {
    vec a{2,2}, b{3,3};
    vec c = multiply(a, b);
    for (auto i : c) cout << i << " ";
}
```

