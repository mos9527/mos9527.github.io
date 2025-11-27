---
author: mos9527
lastmod: 2025-08-27T09:51:14.093499
title: 算竞笔记 - 题集/板子整理（C++）
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---
# Preface

参考主要来自[《算法竞赛入门经典：训练指南》](https://cread.jd.com/read/startRead.action?bookId=30133704&readType=1)、[OIWiki](https://oi-wiki.org/)、[CP Algorithms](https://cp-algorithms.com/)等资源和多方博客、课程，在自己的码风下所著

**注：** 部分实现可能用到较新语言特性，烦请修改后在较老OJ上使用；**原则上提供的代码兼容符合Cpp14及以上标准的编译器**

以下板子/题目为自己接触ACM一年期间积累。现整理为单个、便携式 Markdown 文档方便打印。体量较大，还请谅解。

[TOC]



# Header
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
int main() {
    fast_io();
    /* El Psy Kongroo */

    return 0;
}
```
# 数学

## 快速幂

```c++
// 注：爆int64考虑__int128或相关intrinsic
// MSVC: https://codeforces.com/blog/entry/106396
// Clang on Visual Studio: https://learn.microsoft.com/en-us/cpp/build/clang-support-cmake?view=msvc-170
template<typename T> T binpow(T a, T res, ll b) {
	while (b > 0) {
		if (b & 1) res = res * a;
		a = a * a;
		b >>= 1;
	}
	return res;
}
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
```

## 拓展欧几里得
求解 $gcd(a,b)$ 和 $ax+by=gcd(a,b)$ 的一组解
```c++
ll exgcd(ll a, ll b, ll& x, ll& y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, x, y);
    ll t = x;
    x = y, y = t - (a / b) * y;
    return d;
}
```

- https://codeforces.com/gym/100963/attachments J - Once Upon A Time

  ```c++
  int main() {
      fast_io();
      /* El Psy Kongroo */
      while (1) {
          ll n, m, a, k; cin >> n >> m >> a >> k;
          if (n == 0  ) break;
          // Y1 = n + mt_1
          // Y2 = k + at_2
          // solve for Y1 = Y2
          // n + mt_1 = k + at_2
          // n + mx = k + ay
          // mx - ay = k - n
          // let A = m, B = -a, M = k - n
          // Ax + By = M
          // Solve for any x,y
          // Solve for
          // Ax + By = 1. With Bezout A B must be coprime
          ll g, x, y, C = n - k;
          ll A = a, B = -m;
          g = exgcd(A, B, x, y);
          if (C % g) cout << "Impossible" << endl;
          else {
              x *= C / g, y *= C / g;
              // Ax + By = M
              // We need positive soultions. What do?
  			// x = x - B, y = y + A
  			// x = x + B, y = y - A
              ll kx = abs(B / g);
              while (x <= 0) x += 10000 * kx; // ？？？
              x %= kx;
              while (x <= 0) x += kx;
  			y = (C - A * x) / B;
  			cout << k + a * x << endl;
          }
      }
      return 0;
  }
  ```

  

## 线性代数

### 矩阵
```c++
template<typename T, size_t Size> struct matrix {
	T m[Size][Size]{};
	struct identity {};
	matrix() {}; // zero matrix
	matrix(identity const&) { for (size_t i = 0; i < Size; i++) m[i][i] = 1; } // usage: matrix(matrix::identity{})
	matrix(initializer_list<initializer_list<T>> l) { // usage: matrix({{1,2},{3,4}})
		size_t i = 0;
		for (auto& row : l) { size_t j = 0; for (auto& x : row) m[i][j++] = x; i++; }
	}
	matrix operator*(matrix const& other) const {
		matrix res;
		for (size_t i = 0; i < Size; i++)
			for (size_t j = 0; j < Size; j++)
				for (size_t k = 0; k < Size; k++)
					res.m[i][j] = (res.m[i][j] + m[i][k] * other.m[k][j]) % MOD;
		return res;
	}
};
typedef matrix<ll, 2> mat2;
typedef matrix<ll, 3> mat3;
typedef matrix<ll, 4> mat4;
```

- https://codeforces.com/gym/105170/submission/261977724

  ```c++
  mat2 Ti[DIM];
  bool s[DIM];
  ll s_len = 0;
  int main() {
  	ios::sync_with_stdio(false); cin.tie(0); cout.tie(0); cerr.tie(0);
  	mat2 F0{ {0,1} }; // [F0,F1]
  	mat2 T0 = F0;
  	mat2 T1{ {0, 1}, {1, 1} };
  	mat2 T{ {1, 1}, {1, 2} }; // T0 + T1
  	char b;
  	while ((b = getchar()) != EOF) if (b == '1' || b == '0') s[s_len++] = b == '1';
  	Ti[0] = mat2{ mat2::identity{} }; Ti[1] = T; for (ll i = 2; i < s_len; i++) Ti[i] = Ti[i - 1] * T;
  	ll ans = 0, n1 = 0;
  	for (ll mask = 0; mask < s_len; mask++) {
  		bool b = s[mask];
  		ll rest = s_len - mask - 1;
  		if (b) {
  			ll d = (T0 * Ti[rest]).m[0][0];
  			T0 = T0 * T1;
  			ans = (ans + d) % MOD;
  			// cerr << "d: " << d << " ans: " << ans << "\n";	
  			n1++;
  		}
  	}
  	ans = (ans + T0.m[0][0]) % MOD;
  	cout << ans;
  	return 0;
  }
  ```

- https://codeforces.com/gym/105336/submission/280576093 (D 编码器-解码器)

  ```c++
  typedef matrix<ll, 101> m100;
  map<char, m100> mp;
  int main() {
      fast_io();
      /* El Psy Kongroo */
      string s, t; cin >> s >> t;
      for (char c : "abcdefghijklmnopqrstuvwxyz") mp[c] = m100{ m100::identity{} };
      for (ll i = 0; i < t.length(); i++) mp[t[i]].m[i][i+1] = 1;
      m100 ans = mp[s[0]];
      for (ll i = 1; i < s.length(); i++) ans = ans * mp[s[i]] * ans;
      cout << ans.m[0][t.length()] << '\n';
      return 0;
  }
  ```

- https://codeforces.com/gym/105170/submission/309042797 (Fib递推，二项式展开)

  ```c++
  typedef matrix<ll, 2> mat2;
  typedef matrix<ll, 3> mat3;
  typedef matrix<ll, 4> mat4;
  mat2 M[DIM];
  mat2 T = mat2{{0,1},{1,1}};
  mat2 I = mat2{mat2::identity{}};
  char S[DIM];
  int main() {
      fast_io();
      M[0] = I;
      M[1] = T + I;
      /*
       * 0 000
       * 1 001
       * 2 010
       * 3 011
       * 4 100
       * 5 101
       * 6 110
       * 7 111
       * -----
       * I -> 0, T -> 1
       * I^3 + I^2 * T + I.. + T^3
       * This is binom expansion
       * \sum = (I + T) ^ m where 2^m - 1 = 7
      */
      scanf("%s", S);
      ll n = strlen(S);
      for (ll i = 2; i < n + 1; i++) M[i] = M[i - 1] * M[1];
      ll ans = 0;
      mat2 pre = T;
      for (ll i = 0; i < n; i++) {
          if (S[i] == '1') {
              // 2^(n - i) - 1
              ans += (pre * M[n - i - 1]).m[0][0] ;
              ans %= MOD;
              pre = pre * T;
          }
      }
      ans += pre.m[0][0];
      ans %= MOD;
      cout << ans << endl;
      return 0;
  }
  ```

  

### 线性基

```c++
struct linear_base : array<ll, 64> {
    void insert(ll x) {
        for (ll i = 63; i >= 0; i--) if ((x >> i) & 1) {
            if (!(*this)[i]) {
                (*this)[i] = x;
                break;
            }
            x ^= (*this)[i];
        }
    }
};
```

- https://oi.men.ci/linear-basis-notes/

- https://www.luogu.com.cn/article/zo12e4s5

- https://codeforces.com/gym/105336/submission/280570848（J 找最小）

  ```c++
  int main() {
      fast_io();
      /* El Psy Kongroo */
      ll t; cin >> t;
      while (t--) {
          ll n; cin >> n;
          ll xor_a = 0, xor_b = 0;
          vec a(n); for (ll& x : a) cin >> x, xor_a ^= x;
          vec b(n); for (ll& x : b) cin >> x, xor_b ^= x;
          // swapping a[i],b[i] equals to xor_a ^= a[i] ^ b[i], xor_b ^= a[i] ^ b[i]
          linear_base lb{};
          for (ll i = 0; i < n; i++) lb.insert(a[i] ^ b[i]);
          for (ll i = 63; i >= 0; i--) {
              ll base = lb[i];
              if (max(xor_a, xor_b) > (max(xor_a ^ base, xor_b ^ base))) {
                  xor_a ^= base;
                  xor_b ^= base;
              }
          }
          cout << max(xor_a, xor_b) << endl;
      }
      return 0;
  }
  ```

  

## 数论杂项

### 皮萨诺周期

*摘自 https://oi-wiki.org/math/combinatorics/fibonacci/#%E7%9A%AE%E8%90%A8%E8%AF%BA%E5%91%A8%E6%9C%9F*

模 $m$ 意义下斐波那契数列的最小正周期被称为 [皮萨诺周期](https://en.wikipedia.org/wiki/Pisano_period)
皮萨诺周期总是不超过 $6m$，且只有在满足 $m=2\times 5^k$ 的形式时才取到等号。

当需要计算第 $n$ 项斐波那契数模 $m$ 的值的时候，如果 $n$ 非常大，就需要计算斐波那契数模 $m$ 的周期。当然，只需要计算周期，不一定是最小正周期。
容易验证，斐波那契数模 $2$ 的最小正周期是 $3$，模 $5$ 的最小正周期是 $20$。
显然，如果 $a$ 与 $b$ 互素，$ab$ 的皮萨诺周期就是 $a$ 的皮萨诺周期与 $b$ 的皮萨诺周期的最小公倍数。

结论 2：于奇素数 $p\equiv 2,3 \pmod 5$，$2p+2$ 是斐波那契数模 $p$ 的周期。即，奇素数 $p$ 的皮萨诺周期整除 $2p+2$。

结论 3：对于素数 $p$，$M$ 是斐波那契数模 $p^{k-1}$ 的周期，等价于 $Mp$ 是斐波那契数模 $p^k$ 的周期。特别地，$M$ 是模 $p^{k-1}$ 的皮萨诺周期，等价于 $Mp$ 是模 $p^k$ 的皮萨诺周期。

---
**因此也等价于 $Mp$ 是斐波那契数模 $p^k$ 的周期。**
**因为周期等价，所以最小正周期也等价。**

- https://codeforces.com/contest/2033/submission/287844746

  ```c++
  ll A[DIM], G[DIM];
  bool vis[DIM];
  int main() {
      fast_io();
      /* El Psy Kongroo */
      ll t; cin >> t;
      while (t--) {
          ll n, k; cin >> n >> k;
          ll a = 1, b = 1, pos = 1;
          if (k == 1) cout << n % MOD << endl;
          else {
              for (ll i = 3; i <= 6 * k + 1; i++) {
                  ll c = (a + b) % k;
                  if (c % k == 0) {
                      pos = i;
                      break;
                  }
                  a = b % k, b = c % k;
              }
              cout << (n % MOD) * pos % MOD << endl;
          }
      }
      return 0;
  }
  ```

  

## 计算几何

### 二维几何

```c++
template <typename T> struct vec2 {
  T x, y;
  ///
  inline T length_sq() const { return x * x + y * y; }
  inline T length() const { return sqrt(length_sq()); }
  inline vec2 &operator+=(vec2 const &other) {
    x += other.x, y += other.y;
    return *this;
  }
  inline vec2 &operator-=(vec2 const &other) {
    x -= other.x, y -= other.y;
    return *this;
  }
  inline vec2 &operator*=(T const &other) {
    x *= other, y *= other;
    return *this;
  }
  inline vec2 &operator/=(T const &other) {
    x /= other, y /= other;
    return *this;
  }
  inline vec2 operator+(vec2 const &other) const {
    vec2 v = *this;
    v += other;
    return v;
  }
  inline vec2 operator-(vec2 const &other) const {
    vec2 v = *this;
    v -= other;
    return v;
  }
  inline vec2 operator*(T const &other) const {
    vec2 v = *this;
    v *= other;
    return v;
  }
  inline vec2 operator/(T const &other) const {
    vec2 v = *this;
    v /= other;
    return v;
  }
  ///
  inline static T dist_sq(vec2 const &a, vec2 const &b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
  }
  inline static T dist(vec2 const &a, vec2 const &b) {
    return sqrt(vec2::dist_sq(a, b));
  }
  inline static T cross(vec2 const &a, vec2 const &b) {
    return a.x * b.y - a.y * b.x;
  }
  inline static T dot(vec2 const &a, vec2 const &b) {
    return a.x * b.x + a.y * b.y;
  }
  ///
  inline friend bool operator<(vec2 const &a, vec2 const &b) {
    if (a.x != b.x)
      return a.x < b.x;
    return a.y < b.y;
  }
  inline friend bool operator==(vec2 const &a, vec2 const &b) {
    return a.x == b.x && a.y == b.y;
  }
};
typedef vec2<lf> point;
```

#### 二维凸包

- https://www.cnblogs.com/WIDA/p/17633758.html#%E9%9D%99%E6%80%81%E5%87%B8%E5%8C%85with-point%E6%96%B0%E7%89%88

```c++
vector<point> convex_hull(vector<point> &p) { // 逆时针
  vector<point> hi, lo;
  sort(p.begin(), p.end());
  p.erase(unique(p.begin(), p.end()), p.end());
  ll n = p.size();
  if (n <= 1)
    return p;
  for (auto a : p) {
    while (hi.size() > 1 &&
           point::cross(a - hi.back(), a - hi[hi.size() - 2]) <= 0)
      hi.pop_back();
    while (lo.size() > 1 &&
           point::cross(a - lo.back(), a - lo[lo.size() - 2]) >= 0)
      lo.pop_back();
    lo.push_back(a);
    hi.push_back(a);
  }

  lo.pop_back();
  reverse(hi.begin(), hi.end());
  hi.pop_back();
  lo.insert(lo.end(), hi.begin(), hi.end());
  return lo;
}
```

- https://codeforces.com/gym/104639/submission/281132024

```c++
typedef vec2<lf> point;
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll n, q; cin >> n >> q;
    vector<point> convex_hull(n); // 逆时针    
    for (auto& p : convex_hull) cin >> p;
    convex_hull.push_back(convex_hull.front());
    while (q--) {
        ll x11, y11, x22, y22; cin >> x11 >> y11 >> x22 >> y22;
        point c1{ (lf)x11,(lf)y11 }, c2{ (lf)x22,(lf)y22 };
        /*
        圆外一点到圆内所有一点平均距离 -> 记点到圆心距为 $$d$$ 对称性易知
        $$d^2_{sum} = \int_0^R \int_0^{2\pi} \sqrt{r^2 + d^2 - 2rd\cos \theta} \, r \, d\theta \, dr = \frac{\pi r^4}{2}+\pi d^2r^2$$
        $$d^2_{avg} = d^2_{sum} / S_{C} = \frac{r^2}{2}+d^2$$
        故求$$d_min$$；点在凸包内显然直接取$$0$$,凸包外则取点到边垂距min（若可取）或点到点min
        */
        point C = (c1 + c2) / 2.0; lf R = (c1 - c2).length() / 2;
        lf ans = R * R / 2;
        if (!is_inside(C)) {
            lf dis = min_dis(C);
            ans += dis * dis;
        }            
        cout << fixed << setprecision(10) << ans << endl;
    }
    return 0;
}
```

#### 旋转卡壳

```c++
lf caliper(vector<point> const &p) { // p为处理好的凸包
  ll j = 2, n = p.size();
  lf mx_dis = 0;
  // 每条边,查j+1 和边 (i,i+1) 的距离是不是比 j 更大，如果是就将 j
  // 加一，否则说明 j 是此边的最优点
  // 检查可以通过叉积比较面积比较距离(垂点)
  for (ll i = 0; i < n; i++) {
    auto e1 = p[i % n], e2 = p[(i + 1) % n];
    while (j < n + i) {
      auto p1 = p[j % n], p2 = p[(j + 1) % n];
      lf a1 = abs(point::cross(e2 - e1, p1 - e1));
      lf a2 = abs(point::cross(e2 - e1, p2 - e1));
      if (a1 <= a2)
        j++;
      else
        break;
    }
    // 此时j点一定为该边上距离最远
    mx_dis = max(
        {mx_dis, point::dist_sq(e1, p[j % n]), point::dist_sq(e2, p[j % n])});
  }
  return mx_dis;
}
```



## 组合数

Lucas：$$\binom{n}{m}\bmod p = \binom{\left\lfloor n/p \right\rfloor}{\left\lfloor m/p\right\rfloor}\cdot\binom{n\bmod p}{m\bmod p}\bmod p$$​
```c++
namespace comb {
	ll fac[DIM], ifac[DIM]; // x!, 1/x!
	void prep(ll N = DIM - 1) {
		fac[0] = fac[1] = ifac[0] = ifac[1] = 1;
		for (ll i = 2; i <= N; i++) fac[i] = fac[i - 1] * i % MOD;
		ifac[N] = binpow_mod(fac[N], MOD - 2, MOD);
		for (ll i = N - 1; i >= 1; i--) ifac[i] = ifac[i + 1] * (i + 1) % MOD;
	}
	ll comb(ll n, ll m) {
		return fac[n] * ifac[m] % MOD * ifac[n - m] % MOD;
	}
	ll lucas(ll n, ll m) {
		if (m == 0) return 1;
		return comb(n % MOD, m % MOD) * lucas(n / MOD, m / MOD) % MOD;
	}
}
```


## 数论
### 乘法逆元
- https://acm.hdu.edu.cn/showproblem.php?pid=7437

给定质数$m$,求$a$的逆元$a^{-1}$​

- 欧拉定理知 $a^{\phi (m)} \equiv 1 \mod m$
- 对质数 $m$, $\phi (m) = m - 1$
- 此情景即为费马小定理，i.e. $a^{m - 1} \equiv 1 \mod m$
- 左右同时乘$a^{-1}$,可得 $a ^ {m - 2} \equiv a ^ {-1} \mod m$
- 即 `a_inv = binpow_mod(a, m - 2, m)`

相关trick

- $p = \frac{x}{x+y}\ mod\ p$ 即为 `p = x * binpow_mod(x + y, m - 2, m)`
  - $1 - p$ 可以为 $1 - \frac{x}{x+y} = \frac{y}{x+y}$ 即为 `inv_p = y * binpow_mod(x+u, m - 2, m)`
  - 也可以为$1 - p\ mod\ m = m + 1 - p\ mod\ m$ 即为 `inv_p = binpow_mod(MOD + 1 - p, m - 2, m)`

### CRT / 中国剩余定理

#### 定义

- https://oi.wiki/math/number-theory/crt/

中国剩余定理 (Chinese Remainder Theorem, CRT) 可求解如下形式的一元线性同余方程组（其中 $n_1, n_2, \cdots, n_k$ 两两互质）：

$$
\begin{cases}
x &\equiv a_1 \pmod {n_1} \newline
x &\equiv a_2 \pmod {n_2} \newline
  &\vdots \newline
x &\equiv a_k \pmod {n_k} \newline
\end{cases}
$$

#### 过程

1.  计算所有模数的积 $n$；
2.  对于第 $i$ 个方程：
    1.  计算 $m_i=\frac{n}{n_i}$；
    2.  计算 $m_i$ 在模 $n_i$ 意义下的 [逆元](./inverse.md)  $m_i^{-1}$；
    3.  计算 $c_i=m_im_i^{-1}$（**不要对 $n_i$ 取模**）。
3.  方程组在模 $n$ 意义下的唯一解为：$x=\sum_{i=1}^k a_ic_i \pmod n$。

### Eratosthenes 筛

- https://oi-wiki.org/math/number-theory/sieve
- https://www.luogu.com.cn/problem/P2158 (欧拉函数)

```c++
namespace eratosthenes_sieve { // Eratosthenes筛法 + 区间筛
    vec primes;
    bool not_prime[DIM];

    void init(ll N=DIM - 1) {
        for (ll i = 2; i <= N; ++i) {
            if (!not_prime[i]) primes.push_back(i);
            for (auto j : primes) {
                if (i * j > N) break;
                not_prime[i * j] = true;
                if (i % j == 0) break;
            }
        }
    }
    void update_range(ll l, ll r) {
        for (auto p : primes) {
            for (ll j = max((ll)ceil(1.0 * l / p), p) * p; j <= r; j += p) not_prime[j] = true;
    	}
    }
}

namespace eratosthenes_sieve_d { // https://oi-wiki.org/math/number-theory/sieve/#筛法求约数个数
    vec primes;
    bool not_prime[DIM];
    ll D[DIM], num[DIM];

    void init(ll N = DIM - 1) {
        D[1] = 1;
        for (ll i = 2; i <= N; ++i) {
            if (!not_prime[i]) primes.push_back(i), D[i] = 2, num[i] = 1;
            for (auto j : primes) {
                if (i * j > N) break;
                not_prime[i * j] = true;
                if (i % j == 0) {
                    num[i * j] = num[i] + 1;
                    D[i * j] = D[i] / num[i * j] * (num[i * j] + 1);
                    break;
                }
                num[i * j] = 1;
                D[i * j] = D[i] * 2;
            }
        }
    }
}
   
namespace eratosthenes_sieve_phi {  // https://oi.wiki/math/number-theory/sieve/#筛法求欧拉函数
    vec primes;    
    bool not_prime[DIM];
    ll phi[DIM];

    void init(ll N = DIM - 1) {
        phi[1] = 1;
        for (ll i = 2; i <= N; ++i) {
            if (!not_prime[i]) primes.push_back(i), phi[i] = i - 1;
            for (auto j : primes) {
                if (i * j > N) break;
                not_prime[i * j] = true;
                if (i % j == 0) {
                    phi[j * i] = phi[i] * j;
                    break;
                }
				phi[j * i] = phi[i] * (j - 1); // phi(j)

            }
        }
    }
}

```

### Miller-Rabin

```c++
bool Miller_Rabin(ll p) {  // 判断素数
    if (p < 2) return 0;
    if (p == 2) return 1;
    if (p == 3) return 1;
    ll d = p - 1, r = 0;
    while (!(d & 1)) ++r, d >>= 1;  // 将d处理为奇数
    for (ll a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 823}) {
        if (p == a) return 1;
        ll x = binpow_mod(a, d, p);
        if (x == 1 || x == p - 1) continue;
        for (int i = 0; i < r - 1; ++i) {
            x = (__int128)x * x % p;
            if (x == p - 1) break;
        }
        if (x != p - 1) return 0;
    }
    return 1;
}
```

### Pollard-Rho

```c++
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
ll Pollard_Rho(ll x) {  // 找出x的一个非平凡因数
    if (x % 2 == 0) return 2;
    ll s = 0, t = 0;
    ll c = ll(rng()) % (x - 1) + 1;
    ll val = 1;
    for (ll goal = 1; ; goal *= 2, s = t, val = 1) {
        for (ll step = 1; step <= goal; step++) {
            t = ((__int128)t * t + c) % x;
            val = (__int128)val * abs((long long)(t - s)) % x;
            if (step % 127 == 0) {
                ll g = gcd(val, x);
                if (g > 1) return g;
            }
        }
        ll d = gcd(val, x);
        if (d > 1) return d;
    }
}
```

### 分解质因数

```c++
// MR+PR
void Prime_Factor(ll x, v& res) {   
    auto f = [&](auto f,ll x){
        if (x == 1) return;
        if (Miller_Rabin(x)) return res.push_back(x);
        ll y = Pollard_Rho(x);
        f(f,y),f(f,x / y);
    };
    f(f,x),sort(res.begin(),res.end());    
}
// Euler
namespace euler_sieve {
	vector<vec> primes;
	void Prime_Factor_Offline(ll MAX) {
		primes.resize(MAX);
		for (ll i = 2; i < MAX; i++) {
			if (!primes[i].empty()) continue;
			for (ll j = i; j < MAX; j += i) {
				ll mj = j;
				while (mj % i == 0) {
					primes[j].push_back(i);
					mj /= i;
				}
			}
		}
	}

	void Prime_Factor(ll x, vec& res) {
		for (ll i = 2; i * i <= x; i++) while (x % i == 0) res.push_back(i), x /= i;
		if (x != 1) res.push_back(x);
	}
}
```

- https://ac.nowcoder.com/acm/contest/81603/E

### 离线分解因数

```c++
vec coeff[DIM];
for (ll i = 2; i < DIM;i++)
    for (ll j = i; j < DIM; j += i)
        coeff[j].push_back(i);
```

# 图论

## 拓扑排序
### Khan BFS

```c++
struct graph {
	vector<vector<ll>> G; // 邻接表
	vector<ll> in; // 入度
    
	ll n;
	graph(ll dimension) : n(dimension), G(dimension + 1),in(dimension + 1) {};
	void add_edge(ll from, ll to) {
		G[from].push_back(to);
		in[to]++;
	}
	bool topsort() {
		L.clear();
		queue<ll> S;
		ll ans = 0;
		for (ll i = 1; i <= n; i++) {
			if (in[i] == 0) S.push(i), dp[i] = 1;
		}
		while (!S.empty()) {
			ll v = S.front(); S.pop();
			L.push_back(v);	
			for (auto& out : G[v])
				if (--in[out] == 0)
					S.push(out);
		}
		return ((L.size() == n) ? true : false); // 当且仅当图为DAG时成立
	}
};
```

### DFS

```c++
// vector<vec> adj
vec vis(n), dep(n), topo; topo.reserve(n);
auto dfs = [&](ll u, auto&& dfs) -> bool {
    vis[u] = 1;
    for (auto& v : adj[u]) {
        dep[v] = max(dep[u] + 1, dep[v]);
        if (vis[v] == 1) /*visiting*/ return false;
        if (vis[v] == 0 && !dfs(v, dfs)) /*to visit*/ return false;
    }
    vis[u] = 2; /*visited*/ 
    topo.push_back(u);
    return true;
};
bool ok = true;
for (ll i = 0; ok && i < n; i++) if (vis[i] == 0) ok &= dfs(i, dfs);
```
## 最短路

### Floyd

```c++
ll F[DIM][DIM];
int main() {
	ll n, m; cin >> n >> m;
	memset(F, 63, sizeof(F));
	for (ll v = 1; v <= n; v++) F[v][v] = 0;
	while (m--) {
		ll u, v, w; cin >> u >> v >> w;
		F[u][v] = min(F[u][v], w);
		F[v][u] = min(F[v][u], w);
	}
	for (ll k = 1; k <= n; k++) {
		for (ll i = 1; i <= n; i++) {
			for (ll j = 1; j <= n; j++) {
				F[i][j] = min(F[i][j], F[i][k] + F[k][j]);
			}
		}
	}
	for (ll i = 1; i <= n; i++) {
		for (ll j = 1; j <= n; j++) {
			cout << F[i][j] << ' ';
		}
		cout << '\n';
	}
}
```

### Dijkstra

```c++
auto dijkstra = [&](ll start) {
    vec dis(n + 1, INF), vis(n + 1, false);
    priority_queue<II, vector<II>, greater<>> T; // 小顶堆
    T.emplace( 0, start );
    dis[start] = 0;
    while (!T.empty())
    {
        auto [_, u] = T.top(); T.pop();
        if (!vis[u]) {
            vis[u] = true;
            for (ll v : G[u]) { // 松弛出边
                if (dis[v] > dis[u] + 1) {
                    dis[v] = dis[u] + 1;
                    T.emplace( dis[v], v );
                }
            }
        }
    }
    return dis;
};
```

## 最小生成树

### Kruskal 

```c++
struct dsu {
	vector<ll> pa;
	dsu(const ll size) : pa(size) { iota(pa.begin(), pa.end(), 0); }; // 初始时，每个集合都是自己的父亲
	inline bool is_root(const ll leaf) { return pa[leaf] == leaf; }
	inline ll find(const ll leaf) { return is_root(leaf) ? leaf : pa[leaf] = find(pa[leaf]); } // 路径压缩
	inline void unite(const ll x, const ll y) { pa[find(x)] = find(y); }
};
struct edge { ll from, to, weight; };
int main() {
	ll n, m; cin >> n >> m;
	vector<edge> edges(m);
	for (auto& edge : edges)
		cin >> edge.from >> edge.to >> edge.weight;
	sort(edges.begin(), edges.end(), PRED(lhs.weight < rhs.weight));
	dsu U(n + 1);
	ll ans = 0;
	ll cnt = 0;
	for (auto& edge : edges) {
		if (U.find(edge.from) != U.find(edge.to)) {
			U.unite(edge.from, edge.to);
			ans += edge.weight;
			cnt++;
		}
	}
	if (cnt == n - 1) cout << ans;
	else cout << "orz";
}
```

## 欧拉回路

在图论中，**欧拉路径（Eulerian path）**是经过图中每条边恰好一次的路径，**欧拉回路（Eulerian circuit）**是经过图中每条边恰好一次的回路。 如果一个图中存在欧拉回路，则这个图被称为**欧拉图（Eulerian graph）**

### Hierholzer

```c++
struct eulerian_path {
    map<ll, set<ll>> G;
    map<ll,ll> in;
    void add_edge(ll u, ll v) {
        G[u].insert(v);
        G[v].insert(u);
        in[u]++; in[v]++;
    }
    vec hierholzer(ll s) {
        ll odds = 0;
        for (auto [v,i] : in) odds += i & 1;
        if (odds != 0 /* 欧拉回路 */ && odds != 2 /* 欧拉路 */) return {};
        vec res;
        auto dfs = [&](ll u, auto&& dfs) -> void {
            while (!G[u].empty()) {
                ll v = *G[u].begin();
                G[u].erase(v);
                G[v].erase(u);
                dfs(v, dfs);
            }
            res.push_back(u);
        };
        dfs(s,dfs);
        reverse(res.begin(), res.end());
        return move(res);
    }
};
```

- https://www.luogu.com.cn/problem/P1341 - 无序字母对

```c++
int main() {
    fast_io();
    /* El Psy Kongroo */
    const string ascii = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    ll n, mn = INF; cin >> n;
    eulerian_path p;
    while (n--) {
        string s; cin >> s;
        ll u = s[0], v = s[1];
        p.add_edge(u, v); // 构造二分图
        mn = min({mn,u,v});
    }
    for (char c : ascii)
        if (p.in[c] & 1) { mn = c; break; } // 欧拉路情况；找最小点切入
    auto res = p.hierholzer(mn);
    if (!res.size()) cout << "No Solution" << endl;
    else {
        for (auto& x : res)
            cout << (char)x;
        cout << endl;
    }
    return 0;
}
```

- https://codeforces.com/contest/2110/problem/E - Melody

```c++
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll t; cin >> t;
    while (t--) {
        ll n; cin >> n;
        eulerian_path euler;
        map<II, ll> inv;
        ll root = 0;
        for (ll i = 1; i <= n; i++) {
            ll v, p; cin >> v >> p;
            euler.add_edge(v, -p); // 构造二分图
            inv[{v,p}] = i;
            root = v;
        }
        for (auto [v, i] : euler.in)
            if (i & 1) { root = v; } // 欧拉路情况起点必然是单入度
        auto res = euler.hierholzer(root);
        if (res.size() != n + 1) {
            cout << "NO" << endl;
        }else {
            cout << "YES" << endl;
            if (n == 1) cout << 1;
            else {
                for (ll i = 0; i < n; i++) {
                    ll u = res[i], v = res[i + 1];
                    if (u < 0) swap(u,v);
                    cout << inv[{u,-v}] << ' ';
                }
            }
            cout << endl;
        }
    }
    return 0;
}
```



## LCA

- RMQ (ST表)

```c++
template<typename Container> struct sparse_table {
	ll len;
	vector<Container> table; // table[i,j] -> [i, i + 2^j - 1] 最大值
	void init(const Container& data) {
		len = data.size();
		ll l1 = ceil(log2(len)) + 1;
		table.assign(len, Container(l1));
		for (ll i = 0; i < len; i++) table[i][0] = data[i];
		for (ll j = 1; j < l1; j++) {
			ll jpow2 = 1LL << (j - 1);
			for (ll i = 0; i + jpow2 < len; i++) {
				// f(i,j) = max(f(i,j-1), f(i + 2^(j-1), j-1))
				table[i][j] = min(table[i][j - 1], table[i + jpow2][j - 1]);
			}
		}
	}
	auto query(ll l, ll r) {
		ll s = floor(log2(r - l + 1));
		// op([l,l + 2^s - 1], [r - 2^s + 1, r])
		// -> op(f(l,s), f(r - 2^s + 1, s))
		return min(table[l][s], table[r - (1LL << s) + 1][s]);
	}
};

struct graph {
	vector<v> G;
	ll n;

	v pos;
	vector<II> depth_dfn;
	sparse_table<vector<II>> st;

	graph(ll n) : n(n), G(n + 1), pos(n + 1) { depth_dfn.reserve(2 * n + 5); };

	void add_edge(ll from, ll to) {
		G[from].push_back(to);
	}

	void lca_prep(ll root) {
		ll cur = 0;		
		// 样例欧拉序 -> 4 2 4 1 3 1 5 1 4
		// 样例之深度 -> 0 1 0 1 2 1 2 1 0
		// 求 2 - 3  ->   ^ - - ^
		// 之间找到深度最小的点即可
		// 1. 欧拉序		
		depth_dfn.clear();
		auto dfs = [&](ll u, ll pa, ll dep, auto& dfs) -> void {
			depth_dfn.push_back({ dep, u }), pos[u] = depth_dfn.size() - 1;
			for (auto& v : G[u]) 
				if (v != pa) {
					dfs(v, u, dep+1, dfs);
					depth_dfn.push_back({ dep, u });
				}
		};
		dfs(root, root, 0,  dfs);
		// 2. 建关于深度st表；深度顺序即欧拉序
		st.init(depth_dfn);
	}
	ll lca(ll x, ll y) {
		ll px = pos[x], py = pos[y]; // 找到x,y的欧拉序
		if (px > py) swap(px, py);
		return st.query(px, py).second;	// 直接query最小深度点；对应即为lca
	}
};

int main() {
	ll n, m, s; scanf("%lld%lld%lld", &n, &m, &s);
	graph G(n + 1);
	for (ll i = 1; i < n; i++) {
		ll x, y; scanf("%lld%lld", &x, &y);
		G.add_edge(x,y);
		G.add_edge(y,x);
	}

	G.lca_prep(s);
	while (m--) {
		ll x, y; scanf("%lld%lld", &x, &y);
		ll ans = G.lca(x, y);
		printf("%lld\n",ans);
	}
}
```

- 倍增思路
  - https://codeforces.com/contest/2033/submission/288921361
  - https://blog.csdn.net/weixin_45799835/article/details/117289362
  - https://www.luogu.com.cn/problem/P5903 (会T...)


```c++
struct graph {
    ll n;

    vector<vector<ll>> G;

    vector<vec> fa;
    vec depth, dis;

    graph(ll n) : n(n), fa(ceil(log2(n)) + 1, vec(n)), depth(n), G(n), dis(n) {}

    void add_edge(ll from, ll to) {
        G[from].push_back(to);
        G[to].push_back(from);
    }

    void prep(ll root) {
        auto dfs = [&](ll u, ll pa, ll dep, auto& dfs) -> void {
            fa[0][u] = pa, depth[u] = dep;
            for (ll i = 1; i < fa.size(); i++) {
                // u 的第 2^i 的祖先是 u 第 (2^(i-1)) 个祖先的第 (2^(i-1)) 个祖先
                fa[i][u] = fa[i - 1][fa[i - 1][u]];
            }
            for (auto& e : G[u]) {
                if (e == pa) continue;
                dis[e] = dis[u] + 1;
                dfs(e, u, dep + 1, dfs);
            }
        };
        dfs(root, root, 1, dfs);
    }

    ll lca(ll x, ll y) {
        if (depth[x] > depth[y]) swap(x, y); // y 更深
        ll diff = depth[y] - depth[x];
        for (ll i = 0; diff; i++, diff >>= 1) // 让 y 上升到 x 的深度
            if (diff & 1) y = fa[i][y];
        if (x == y) return x;
        for (ll i = fa.size() - 1; i >= 0; i--) {
            if (fa[i][x] != fa[i][y]) {
                x = fa[i][x];
                y = fa[i][y];
            }
        }
        return { fa[0][x] };
    }

    ll kth_parent(ll u, ll k){
        for (ll i = 63;i >= 0;i--) if (k & (1ll << i)) u = fa[i][u];
        return u;
    }
};
```

## 树的直径

- https://oi-wiki.org/graph/tree-diameter/

```c++
struct edge { ll to, cost; };
struct graph {
	ll n;

	vector<vector<edge>> G;
	v dis, fa;
	vector<bool> tag;
	graph(ll n) : n(n), G(n), dis(n), fa(n), tag(n) {};

	void add_edge(ll from, ll to, ll cost = 1) {
		G[from].push_back({ to, cost });
	}

	// 实现 1：两次DFS -> 起止点
	// 不能处理负权边（？）
	ll path_dfs() {
		ll end = 0; dis[end] = 0;
		auto dfs = [&](ll u, ll pa, auto& dfs) -> void {
			fa[u] = pa; // 反向建图
			for (auto& e : G[u]) {
				if (e.to == pa) continue;
				dis[e.to] = dis[u] + e.cost;
				if (dis[e.to] > dis[end]) end = e.to;
				dfs(e.to, u, dfs);				
			}
		};
		// 在一棵树上，从任意节点 y 开始进行一次 DFS，到达的距离其最远的节点 z 必为直径的一端。
		dfs(1, 1, dfs); // 1 -> 端点 A
		ll begin = end;
		dis[end] = 0;   
		dfs(end,end, dfs); // 端点 A -> B
		// fa回溯既有 B -> A 路径；省去额外dfs
		fa[begin] = 0; for (ll u = end; u ; u = fa[u]) tag[u] = true;
		return dis[end];
	}

	// 实现 2：树形DP -> 长度
	ll path_dp() {
		v dp(n); // 定义 dp[u]：以 u 为根的子树中，从 u 出发的最长路径
		// dp[u] = max(dp[u], dp[v] + cost(u,v)), v \in G[u]
		ll ans = 0;
		auto dfs = [&](ll u, ll pa, auto& dfs) -> void {
			for (auto& e : G[u]) {
				if (e.to == pa) continue;
				dfs(e.to, u, dfs);
				ll cost = e.cost;
				// 题解：第一条直径边权设为-1
				// - 若这些边被选择（与第二条边重叠），贡献则能够被抵消，否则这些边将走两遍
				// - 若没被选择，则不对第二次答案造成影响
				if (tag[u] && tag[e.to]) cost = -1;
				ans = max(ans, dp[u] + dp[e.to] + cost);
				dp[u] = max(dp[u], dp[e.to] + cost);
			}
		};
		dfs(1, 1, dfs);
		return ans;
	}
};

// 便携板子
auto diameter = [&]()
{
    vec dis(n + 1);
    ll end = 0; dis[end] = 0;
    auto dfs = [&](ll u, ll pa, auto&& dfs) -> void {
        for (auto& e : G[u]) {
            if (e == pa) continue;
            dis[e] = dis[u] + 1;
            if (dis[e] > dis[end]) end = e;
            dfs(e, u, dfs);
        }
    };
    // 在一棵树上，从任意节点 y 开始进行一次 DFS，到达的距离其最远的节点 z 必为直径的一端。
    dfs(1, 1, dfs); // 1 -> 端点 A
    ll begin = end;
    dis[end] = 0;
    dfs(end,end, dfs); // 端点 A -> B
    return dis[end];
};
```

- https://codeforces.com/contest/2107/problem/D

  限时$5s$，就题目数据量可以偷懒...

  需要$(d,u,v)$序列字典序最大很显然需要$d$最大；图中最长简单路径即树的直径

  额外限制即为$u，v$选择唯一（字典序！）故DFS时选终点，同长度选点编号更大者（才知道`std::pair`能做`max`）

  选一个直径之后‘删掉’这些点，重复至没有点为止。

  ```c++
  int main() {
      fast_io();
      /* El Psy Kongroo */
      ll t; cin >> t;
      while (t--) {
  		ll n; cin >> n;
  		vector<vec> G(n + 1);
          for (ll i = 1; i < n; i++) {
              ll u, v; cin >> u >> v;
              G[u].push_back(v);
              G[v].push_back(u);
          }
          unordered_set<ll> res;
          vector<tuple<ll, ll, ll>> ans;
          while (res.size() != n) {
              vec fa(n + 1, -1), vis(n + 1), dis(n + 1);
              auto dfs = [&](ll u, ll pa, auto&& dfs) -> II {
                  fa[u] = pa;
                  vis[u] = 1;				
                  II end{ 1, u };
                  for (ll v : G[u]) {
                      if (v == pa || res.contains(v)) continue;
                      auto nxt = dfs(v, u, dfs);
                      nxt.first += 1;
                      end = max(end, nxt);
                  }
                  return end;
              };
              for (ll i = 1; i <= n; i++) {                
                  if (!vis[i] && !res.contains(i)) {
                      // 树的直径
                      auto end = dfs(i, -1, dfs);
  					ll u = end.second;
                      end = dfs(u, -1, dfs);
                      ll v = end.second;
                      ll d = end.first;                    
                      // (d,u,v)
                      ans.push_back({ d,max(u,v),min(u,v)});
                      while (v != -1) {
                          res.insert(v);
                          v = fa[v];
                      }
                  }
              }
          }
          sort(ans.begin(), ans.end());
          reverse(ans.begin(), ans.end());
          for (auto [d, u, v] : ans)
              cout << d << " " << u << " " << v << " ";
          cout << endl;
      }
      return 0;
  }
  ```

## Dinic 最大(最小费用)流

- https://www.cnblogs.com/SYCstudio/p/7260613.html

```c++
struct dinic_flow {
    ll n, cnt = 0;
    vec nxt, head;
    struct edge { ll v, capacity, cost; };
    vector<edge> e;
    dinic_flow(ll verts, ll edges = DIM) : e(edges), nxt(edges, -1), head(edges, -1), n(verts), dis(n + 1), cur(n + 1), vis(n + 1) {}
private:
    void add_edge(ll u, ll v, ll capacity, ll cost) {
        nxt[cnt] = head[u];
        e[cnt] = {v, capacity, cost};
        head[u] = cnt;
        cnt++;
    }
    vec dis, cur; // 最短路, 当前弧
    vector<bool> vis;
    // 最大流
    // 分层图
    bool dinic_bfs(ll s, ll t) {
        // 分层
        queue<ll> Q;
        dis.assign(n + 1, 0);
        dis[s] = 1; Q.push(s);
        // O(mn)
        while (!Q.empty()){
            ll u = Q.front(); Q.pop();
            for (ll i = head[u]; i != -1; i = nxt[i]) {
                auto [v, capacity, cost] = e[i];
                // 还有容量即可传递
                if (capacity > 0 && dis[v] == 0) {
                    dis[v] = dis[u] + 1;
                    Q.push(v);
                }
            }
        }
        return dis[t] > 0;
    }
    // 最小费用最大流 Min-cost-max-flow
    // 每次找费用最少的分层图
    bool dinic_bfs_mcmf(ll s, ll t) {
        vis.assign(n + 1, 0);
        dis.assign(n + 1, INF);
        queue<ll> Q;
        dis[s] = 0, vis[s] = 1, Q.push(s);
        while (!Q.empty()){
            const ll u = Q.front(); Q.pop(), vis[u] = false;
            for (ll i = head[u]; i != -1; i = nxt[i]) {
                auto [v, capacity, cost] = e[i];
                // SPFA 找到最低cost路径传递,松弛出边
                if (capacity > 0 && dis[v] > dis[u] + cost) {
                    dis[v] = dis[u] + cost;
                    if (!vis[v]) Q.push(v), vis[v] = true;
                }
            }
        }
        return dis[t] != INF;
    }
    // 最大流
    // 增广路
    ll dinic_dfs(ll u, ll t, ll flow = INF) {
        if (u == t) return flow;
        for (ll& i = cur[u] /* 维护掉已经走过的弧 */; i != -1; i = nxt[i]) {
            auto& [v, capacity, cost] = e[i];
            auto& [v_inv, capacity_inv, _] = e[i^1];
            if (capacity > 0 && dis[v] == dis[u] + 1) {
                ll d = dinic_dfs(v, t, min(flow, capacity));
                // 往下传递到未走边
                if (d > 0)
                {
                    capacity -= d, capacity_inv += d; // 传递反向边
                    return d; // 向上传递
                }
            }
        }
        return 0; // 没有增广路
    }
    // 最小费用最大流 Min-cost-max-flow
    // 增广路
    ll cost_mcmf = 0;
    ll dinic_dfs_mcmf(ll u, ll t, ll flow = INF) {
        // 找增广路
        if (u == t) return flow;
        vis[u] = true;
        ll cur_flow = 0;
        for (ll& i = cur[u] /* 维护掉已经走过的弧 */; i != -1 && cur_flow < flow; i = nxt[i]) {
            auto& [v, capacity, cost] = e[i];
            auto& [v_inv, capacity_inv, _] = e[i^1];
            if (!vis[v] && capacity > 0 && dis[v] == dis[u] + cost) {
                ll d = dinic_dfs_mcmf(v, t, min(flow - cur_flow, capacity));
                // 往下传递到未走边
                if (d > 0)
                {
                    capacity -= d, capacity_inv += d; // 传递反向边
                    cur_flow += d, cost_mcmf += d * cost;
                    // return d
                    // 继续遍历,最大流f此时一定
                    // 需要找到最小费用即需要该复杂度,最终O(mnf)
                }
            }
        }
        vis[u] = false;
        return cur_flow; // 没有增广路时仍为0
    }
public:
    void dinic_add_edge(ll u, ll v, ll capacity, ll cost = 0) {
        add_edge(u, v, capacity, cost); // W[i]
        add_edge(v, u, 0, -cost); // W[i^1]
    }
    // 最大流 O(mn)
    // 可复用
    ll dinic(ll s, ll t) {
        ll ans = 0;
        auto e_pre = e;
        while (dinic_bfs(s, t)) {
            cur = head;
            while (ll d = dinic_dfs(s, t)) ans += d;
        }
        e = e_pre;
        return ans;
    }
    // 最小费用最大流 O(mnf), f为最大流 -> [最大流, 最小费用]
    // 可复用
    II dinic_mcmf(ll s, ll t) {
        ll ans = 0;
        cost_mcmf = 0;
        auto e_pre = e;
        while (dinic_bfs_mcmf(s, t)) {
            cur = head;
            while (ll d = dinic_dfs_mcmf(s, t)) ans += d;
        }
        e = e_pre;
        return {ans, cost_mcmf};
    }
};
```

- https://codeforces.com/gym/105336/submission/280592598 (G. 疯狂星期六)

  ```c++
  int main() {
      fast_io();
      /* El Psy Kongroo */
      ll n, m; cin >> n >> m;
      ll t = n + m + 1;
      vec A(n + 1), V(n + 1);
      for (ll i = 1; i <= n; i++) cin >> A[i] >> V[i];
      dinic_flow G(n + m + 1);
      ll cost_yyq = V[1], cost_all = 0;
      for (ll i = 1; i <= m; i++) {
          ll x, y, W; cin >> x >> y >> W;
          if (x == 1 || y == 1) cost_yyq += W;
          cost_all += W;
          // 源点-菜，菜-两个人
          // 规定人点m开始
          G.dinic_add_edge(0, i, W);
          G.dinic_add_edge(i, x + m, W);
          G.dinic_add_edge(i, y + m, W);
      }
      ll mxcost_yyq = min(cost_yyq, A[1]);
      for (ll i = 1; i <= n; i++) {
          if (i > 1 && V[i] >= mxcost_yyq) {
              cout << "NO"; return 0;
          } // 车费特判
          if (i == 1) G.dinic_add_edge(i + m, t, mxcost_yyq - V[i]); // 最大费用->结点
          else G.dinic_add_edge(i + m, t, min(mxcost_yyq - 1, A[i]) - V[i]); // **严格** 大于他人花费
      }
      ll ans = G.dinic(0, t);
      if (ans == cost_all) cout << "YES";
      else cout << "NO";
      return 0;
  }
  ```

- https://www.luogu.com.cn/problem/P4015 运输问题

  ```c++
  /*
  2 3
  220 280
  170 120 210
  77 39 105
  150 186 122
   */
  int main() {
      fast_io();
      /* El Psy Kongroo */
      ll m,n; cin >> m >> n;
      vec a(m + 1), b(n + 1);
      vector<vec> c(m + 1, vec(n + 1));
      for (ll i = 1; i <= m; i++) cin >> a[i];
      for (ll i = 1; i <= n; i++) cin >> b[i];
      for (ll i = 1; i <= m; i++)
          for (ll j = 1; j <= n; j++)
              cin >> c[i][j];
      ll sz = m + n + 20;
      dinic_flow G(sz + 20);
      ll s = sz + 1, t = sz + 2; // 虚拟源点和汇点
      for (ll i = 1; i <= m; i++) {
          // 虚拟源点到每一个供货点
          G.dinic_add_edge(s, i, a[i],0);
      }
      for (ll i = 1; i <= n; i++) {
          // 需求点到虚拟汇点
          G.dinic_add_edge(m + i, t, b[i], 0);
      }
      for (ll i = 1; i <= m; i++) {
          for (ll j = 1; j <= n; j++) {
              // 每个供货点到需求点
              G.dinic_add_edge(i, m + j, INF, c[i][j]);
          }
      }
      auto [max_flow, min_cost] = G.dinic_mcmf(s, t);
      // cost取反再求mincost即可求maxcost
      for (auto& e : G.e)
          e.cost *= -1;
      auto [_, max_cost] = G.dinic_mcmf(s, t);
      max_cost = -max_cost;
      cout << min_cost << endl;
      cout << max_cost << endl;
      return 0;
  }
  ```



## 树链剖分 / HLD

- https://www.cnblogs.com/WIDA/p/17633758.html#%E6%A0%91%E9%93%BE%E5%89%96%E5%88%86hld
- https://oi-wiki.org/graph/hld/
- https://cp-algorithms.com/graph/hld.html
- https://www.luogu.com.cn/problem/P5903

```c++
struct HLD {
    ll n, dfn_cnt = 0;
    vec sizes, depth, top /*所在重链顶部*/, parent, dfn /*DFS序*/, dfn_out /* 链尾DFS序 */, inv_dfn, heavy /*重儿子*/;
    vector<vec> G;
    HLD(ll n) : n(n), G(n), sizes(n), depth(n), top(n), parent(n), dfn(n), dfn_out(n), inv_dfn(n), heavy(n) {};
    void add_edge(ll u, ll v) {
        G[u].push_back(v);
        G[v].push_back(u);
    }
    // 注：唯一的重儿子即为最大子树根
    void dfs1(ll u) {
        heavy[u] = -1;
        sizes[u] = 1;
        for (ll& v : G[u]) {
            if (depth[v]) continue;
            depth[v] = depth[u] + 1;
            parent[v] = u;
            dfs1(v);
            sizes[u] += sizes[v];
            // 选最大子树为重儿子
            if (heavy[u] == -1 || sizes[v] > sizes[heavy[u]]) heavy[u] = v;
        }
    }
    // 注：dfn为重边优先时顺序
    void dfs2(ll u, ll v_top) {
        top[u] = v_top;
        dfn[u] = ++dfn_cnt;
        inv_dfn[dfn[u]] = u;
        if (heavy[u] != -1) {
            // 优先走重儿子
            dfs2(heavy[u], v_top);
            for (ll& v : G[u])
                if (v != heavy[u] && v != parent[u]) dfs2(v, v);
        }
        dfn_out[u] = dfn_cnt;
    }
    // 预处理(!!)
    void prep(ll root) {
        depth[root] = 1;
        dfs1(root);
        dfs2(root, root);
    }
    // 多点lca
    ll lca(ll a, ll b, ll c) {
        return lca(a, b) ^ lca(b, c) ^ lca(c, a);
    }
    // 树上两点距离
    ll dist(ll u, ll v) {
        return depth[u] + depth[v] - 2 * depth[lca(u, v)] + 1;
    }
    // logn求LCA
    ll lca(ll u, ll v) {
        while (top[u] != top[v]) // 到同一重链
        {
            // 跳到更深的链
            if (depth[top[u]] < depth[top[v]]) swap(u, v);
            u = parent[top[u]];
        }
        return depth[u] < depth[v] ? u : v;
    }
    // 路径上区间query dfn序
    void path_sum(ll u, ll v, auto&& query) {
        while (top[u] != top[v]) // 到同一重链
        {
            // 跳到更深的链
            if (depth[top[u]] < depth[top[v]]) swap(u, v);
            // [dfn[top[u]],[u]]上求和 (在此插入RMQ)
            query(dfn[top[u]], dfn[u]);
            u = parent[top[u]];
        }
        if (dfn[v] > dfn[u]) swap(u, v);
        query(dfn[v], dfn[u]);
    }
    // 第k的父亲
    ll kth_parent(ll u, ll k) {
      ll dep = depth[u] - k;
      while (depth[top[u]] > dep) u = parent[top[u]];
      return inv_dfn[dfn[u] - (depth[u] - dep)];
    }
    // v属于u的子树
    bool is_child_of(ll u, ll v) {
        return dfn[u] <= dfn[v] && dfn[v] <= dfn_out[u];
    }
};
```

### 树上并查集 / DSU On Tree

- https://zhuanlan.zhihu.com/p/658598885?theme=dark
- https://www.bilibili.com/video/BV1ujo6YCEjt

```c++
	ll n, dfn_cnt = 0;
	vec sizes, depth, top /*所在重链顶部*/, parent, dfn /*DFS序*/, dfn_out /* 链尾DFS序 */, inv_dfn, heavy /*重儿子*/;
	vector<vec> G;
	DSU(ll n) : n(n), G(n), sizes(n), depth(n), top(n), parent(n), dfn(n), dfn_out(n), inv_dfn(n + 1), heavy(n) {};
	void add_edge(ll u, ll v) {
		G[u].push_back(v);
		G[v].push_back(u);
	}
	// 注：唯一的重儿子即为最大子树根
	void dfs1(ll u) {
		sizes[u] = 1;		
		dfn[u] = ++dfn_cnt;
		inv_dfn[dfn_cnt] = u;
		for (ll& v : G[u]) {
			if (depth[v]) continue;
			depth[v] = depth[u] + 1;
			parent[v] = u;
			dfs1(v);
			sizes[u] += sizes[v];
			// 选最大子树为重儿子
			if (!heavy[u] || sizes[v] > sizes[heavy[u]]) heavy[u] = v;
		}
		dfn_out[u] = dfn_cnt;
	}
	// 树上集合合并；考虑小集合入大集合
	// 考虑分成轻重链以后作为启发合并
	// 先处理完轻儿子后合并到重儿子上
	// 轻儿子大小<=重儿子；合并后大小>=2N，等效于倍增；
	void dfs2(ll u, ll pa, bool keep) {
		// 轻贡献
		for (ll& v : G[u]) {
			if (v == pa || heavy[u] == v) continue;
			dfs2(v, u, false); // 不保留 - 方便其他轻子树传递
		}
		// 重贡献
		if (heavy[u])
			dfs2(heavy[u], u, true); // 保留 - 合并到此子树上
		// 合并所有*轻*子树贡献
		insert(u);
		for (ll& v : G[u]) {
			if (v == pa || heavy[u] == v) continue;
			for (ll w = dfn[v]; w <= dfn_out[v]; w++)
				insert(inv_dfn[w]);
		}
		save(u);
		// 不保留情况
		if (!keep) {
			for (ll w = dfn[u]; w <= dfn_out[u]; w++)
				remove(inv_dfn[w]);
		}
	}
    // 预处理(!!)
    void prep(ll root = 1) {
        depth[root] = 1;
        dfs1(root);
        dfs2(root, 0, false);
    }
	// u点状态维护完毕    
	void save(ll u) {

	}
	// 在该子树构成集合+点    
	void insert(ll u) {

	}
	// 撤销该子树对当前集合贡献
	void remove(ll u) {

	}
};
```

- https://www.luogu.com.cn/problem/U41492 (U41492 树上数颜色)

```c++
 vec c, ans, cols; ll unique_cols = 0;
	...
	// u点状态维护完毕    
	void save(ll u) {
		ans[u] = unique_cols;
	}
	// 在该子树构成集合+点    
	void insert(ll u) {
		ll col = c[u];
		if (cols[col] == 0) unique_cols++;
		cols[col]++;
	}
	// 撤销该子树对当前集合贡献
	void remove(ll u) {
		ll col = c[u];
		cols[col]--;
		if (cols[col] == 0) unique_cols--;
		unique_cols = max(unique_cols, 0LL);
	}
};
int main() {
	fast_io();
	/* El Psy Kongroo */
	ll n; cin >> n;
	DSU dsu(n + 1);
	for (ll i = 0; i < n - 1; i++) {
		ll x, y; cin >> x >> y;
		dsu.add_edge(x, y);		
	}
	c.resize(n + 1), ans.resize(n + 1), cols.resize(n + 1),	unique_cols = 0;
	for (ll i = 1; i <= n; i++) cin >> c[i];
	dsu.prep(1);
	ll m; cin >> m;
	while (m--) {
		ll u; cin >> u;
		cerr << "##";
		cout << ans[u] << endl;
	}
	return 0;
}
```

- https://codeforces.com/contest/600/problem/E (E. Lomsat gelral)

```c++
vec c, ans, cols; ll cur_mx = 0, cur_sum = 0;
...
		// 不保留情况
		if (!keep) {			
			cur_mx = cur_sum = 0; // 额外有脏状态需要清掉
			for (ll w = dfn[u]; w <= dfn_out[u]; w++)
				remove(inv_dfn[w]);
		}
	}
	// 预处理(!!)
	void prep(ll root = 1) {
		dfs1(root);
		dfs2(root, 0, false);
	}	
	// u点状态维护完毕    
	void save(ll u) {
		ans[u] = cur_sum;
	}
	// 在该子树构成集合+点    
	void insert(ll u) {
		ll col = c[u];		
		cols[col]++;
		// new dominating color
		// count from here onwards
		if (cols[col] > cur_mx)
			cur_mx = cols[col], cur_sum = 0;
		if (cols[col] == cur_mx)
			cur_sum += col;
	}
	// 撤销该子树对当前集合贡献
	void remove(ll u) {
		ll col = c[u];
		cols[col]--;		
	}
};
int main() {
	fast_io();
	/* El Psy Kongroo */
	ll n; cin >> n;
	DSU dsu(n + 1);
	c.resize(n + 1), ans.resize(n + 1), cols.resize(n + 1);
	for (ll i = 1; i <= n; i++) cin >> c[i];
	for (ll i = 0; i < n - 1; i++) {
		ll x, y; cin >> x >> y;
		dsu.add_edge(x, y);		
	}
	dsu.prep(1);
	for (ll i = 1; i <= n; i++) {
		cout << ans[i] << " ";
	}
	return 0;
}
```

- https://codeforces.com/contest/570/problem/D (D. Tree Requests)

```c++
array<array<ll, DIM>, 26> cnt; // char,depth -> count
array<vector<II>, DIM> queries; // subtree->depth,index
vec ans; vector<char> c;
...
	// u点状态维护完毕    
	void save(ll u) {
		for (auto [dep, o] : queries[u]) {
			ll sum = 0, odd = 0;
			for (ll i = 0; i < 26; i++) {
				ll cur = cnt[i][dep];
				if (cur & 1) odd++;
				sum += cur;
			}
			if ((odd == 1 && sum & 1) || (odd == 0 && sum % 2 == 0)) ans[o] = true;
			else ans[o] = false;
		}
	}
	// 在该子树构成集合+点    
	void insert(ll u) {	
		cnt[c[u]][depth[u]]++;
	}
	// 撤销该子树对当前集合贡献
	void remove(ll u) {
		cnt[c[u]][depth[u]]--;
	}
};
int main() {
	fast_io();
	/* El Psy Kongroo */
	ll n, m; cin >> n >> m;
	DSU dsu(n + 1);
	c.resize(n + 1), ans.resize(m + 1), c.resize(n + 1);
	for (ll i = 2; i <= n; i++) {
		ll pa; cin >> pa;
		dsu.add_edge(i, pa);
	}
	for (ll i = 1; i <= n; i++) cin >> c[i], c[i] -= 'a';
	for (ll i = 1; i <= m; i++) {
		ll v, h; cin >> v >> h;
		queries[v].push_back({ h,i });
	}
	dsu.prep(1);
	for (ll i = 1; i <= m; i++) {
		cout << (ans[i] ? "Yes" : "No") << endl;
	}
	return 0;
		queries[v].push_back({ h,i });
	}
	dsu.prep(1);
	for (ll i = 1; i <= m; i++) {
		cout << (ans[i] ? "Yes" : "No") << endl;
	}
	return 0;
}
```



 ## 强连通分量 / SCC

### Tarjan
```c++
struct SCC {
  ll n, dfn_cnt;
  vec dfn, low, sta, dis;
  vec inv_scc;
  stack<ll> stk;
  vector<vector<II>> G;
  SCC(ll n)
      : n(n), sta(n), dfn(n) /* DFS序 */, low(n) /* 所在SCC首个（dfn最低）点 */,
        G(n), dis(n) /* 到根距离 */, dfn_cnt(0),
        inv_scc(n) /* 点所在SCC的根点 */ {};

  void add_edge(ll u, ll v, ll w = 1) { G[u].push_back({v, w}); }

  map<ll, vec> scc;
  void tarjan(ll u = 1) {
    if (dfn[u])
      return;
    dfn[u] = low[u] = ++dfn_cnt, stk.push(u), sta[u] = 1;
    for (auto const &[v, w] : G[u]) {
      if (!dfn[v])
        dis[v] = dis[u] + w, tarjan(v), low[u] = min(low[u], low[v]);
      else if (sta[v])
        low[u] = min(low[u], dfn[v]);
    }
    if (dfn[u] == low[u]) {
      ll v;
      do {
        v = stk.top(), stk.pop();
        sta[v] = 0;
        scc[u].push_back(v);
        inv_scc[v] = u;
      } while (u != v);
    }
  }
};
```

- https://byvoid.com/zhs/blog/scc-tarjan/

- https://codeforces.com/gym/105578/problem/M 2024 沈阳 M

  - https://codeforces.com/gym/105578/submission/312325530
  
  ```c++
  int main() {
      fast_io();
      /* El Psy Kongroo */
      ll n, m, q; cin >> n >> m >> q;
      SCC scc(n + 2);
      while (m--) {
          ll a, b; cin >> a >> b;
          // a % n -> nth floor [-n/2,n/2]
          scc.add_edge((a % n + n - 1) % n, ((a + b) % n + n - 1) % n, b);
      }
  	for (ll i = 0; i < n; i++) scc.tarjan(i);
      while (q--) {
          ll x; cin >> x;
  		x = (x % n + n - 1) % n;
          if (scc.in_loop[x]) cout << "Yes\n";
          else cout << "No\n";
      }
      return 0;
  }
  ```

- https://www.luogu.com.cn/problem/P3387 缩点版题

```c++
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll n, m; cin >> n >> m;
    SCC scc(n + 1);
    vec w(n + 1);
    for (ll i = 1; i <= n; i++) cin >> w[i];
	for (ll i = 0; i < m; i++) {
		ll u, v; cin >> u >> v;
		scc.add_edge(u, v);
	}
    for (ll i = 1; i <= n; i++) scc.tarjan(i);
    // 强连通分量可当成一点
    // 缩点后构成的新图*一定*是树/DAG
    vector<vec> G(n + 1);
    vec a(n + 1), in(n + 1);
    for (ll u = 1; u <= n; u++)
        a[scc.inv_scc[u]] += w[u];
    for (ll u = 1; u <= n; u++) {
		for (auto [v, e] : scc.G[u]) {
            ll scc_u = scc.inv_scc[u], scc_v = scc.inv_scc[v];			            
            if (scc_u != scc_v) 
                G[scc_u].push_back(scc_v), in[scc_v]++;
        }
    }
    // 现在即树上找最长路径 
    // 考虑拓扑序；dp[v]->v结尾最大值，从上到下传递有(u->v), dp[v] = max(dp[v], dp[u] + a[v])    
    queue<ll> S;
    vec dp = a;
    for (ll i = 1; i <= n; i++) if (in[i] == 0 && G[i].size()) S.push(i);
    while (!S.empty()) {
        ll u = S.front(); S.pop();        
        for (ll& v : G[u]) {
            dp[v] = max(dp[v], dp[u] + a[v]);
            if (--in[v] == 0)
                S.push(v);
        }
    }    
	cout << *max_element(dp.begin(), dp.end()) << endl;
    return 0;
}
```

- https://codeforces.com/gym/101170 - Birtish Food (最长路)

```c++
int main() {
  fast_io();
  /* El Psy Kongroo */
  ll n, m;
  cin >> n >> m;
  SCC scc(n + 1);
  for (ll i = 0; i < m; i++) {
    ll u, v;
    cin >> u >> v;
    scc.add_edge(u, v);
  }
  for (ll i = 1; i <= n; i++)
    scc.tarjan(i);
  vec dp(n + 1, 1);
  vec in(n + 1, 0);
  for (ll u = 1; u <= n; u++) {
    for (auto const &[v, w] : scc.G[u])
      if (scc.inv_scc[u] != scc.inv_scc[v])
        in[scc.inv_scc[v]]++;
  }
  queue<ll> Q;
  for (ll i = 1; i <= n; i++)
    if (in[i] == 0)
      Q.push(i);
  vec dp2(n + 1, 0);
  vec vis(n + 1, 0);
  auto dfs = [&](ll u, ll d, auto &&dfs) -> void { // O(n!)
    vis[u] = 1;
    dp2[u] = max(dp2[u], d);
    for (auto [v, w] : scc.G[u])
      if (scc.inv_scc[u] == scc.inv_scc[v] && !vis[v])
        dfs(v, d + 1, dfs);
    vis[u] = 0;
  };
  // 现在即树上找最长路径 
  // 考虑拓扑序；dp[v]->v结尾最大值，从上到下传递有(u->v), dp[v] = max(dp[v], dp[u] + a[v])    
  while (!Q.empty()) {
    ll u = Q.front();
    Q.pop();
    // 跑完scc上所有可能的最长路组合
    // 每个点都会跑一边跑整张(子)图,O()
    for (ll v : scc.scc[u])
      dfs(v, dp[v], dfs);
    for (ll v : scc.scc[u]) {
      for (auto [e, w] : scc.G[v]) {
        if (scc.inv_scc[e] != scc.inv_scc[v]) {
          in[scc.inv_scc[e]]--;
          if (!in[scc.inv_scc[e]])
            Q.push(scc.inv_scc[e]);
          dp[e] = max(dp[e], dp2[v] + 1);
        }
      }
    }
  }
  ll ans = *max_element(dp2.begin(), dp2.end());
  cout << ans << endl;
  return 0;
}
```

# 数据结构 / DS

## RMQ 系列
### 滑动窗口（单调队列）

- https://oi-wiki.org/ds/monotonous-queue/

```c++
deque<ll> dq; // k大小窗口
for (ll i = 1; i <= n; i++) {
    // 维护k窗口min
    while (dq.size() && dq.front() <= i - k) dq.pop_front();
    while (dq.size() && a[dq.back()] >= a[i]) dq.pop_back();
    dq.push_back(i);
    if (i >= k) cout << a[dq.front()] << ' ';
}
for (ll i = 1; i <= n; i++) {
    // 维护k窗口max
    while (dq.size() && dq.front() <= i - k) dq.pop_front();
    while (dq.size() && a[dq.back()] <= a[i]) dq.pop_back();
    dq.push_back(i);
    if (i >= k) cout << a[dq.front()] << ' ';
}
```

### ST 表

```c++
template<typename Container> struct sparse_table {
	ll len;
	vector<Container> table; // table[i,j] -> [i, i + 2^j - 1] 最大值
	void init(const Container& data) {
		len = data.size();
		ll l1 = ceil(log2(len)) + 1;
		table.assign(len, Container(l1));
		for (ll i = 0; i < len; i++) table[i][0] = data[i];
		for (ll j = 1; j < l1; j++) {
			ll jpow2 = 1LL << (j - 1);
			for (ll i = 0; i + jpow2 < len; i++) {
				// f(i,j) = max(f(i,j-1), f(i + 2^(j-1), j-1))
				table[i][j] = min(table[i][j - 1], table[i + jpow2][j - 1]);
			}
		}
	}
	auto query(ll l, ll r) {
		ll s = floor(log2(r - l + 1));
		// op([l,l + 2^s - 1], [r - 2^s + 1, r])
		// -> op(f(l,s), f(r - 2^s + 1, s))
		return min(table[l][s], table[r - (1LL << s) + 1][s]);
	}
};
```



### 树状数组
```c++
struct fenwick : public vec {
    using vec::vec;
    void init(vec const& a) {
        for (ll i = 0; i < a.size(); i++) {
            (*this)[i] += a[i]; // 求出该子节点
            ll j = i + lowbit(i);
            if (j < size()) (*this)[j] += (*this)[i]; // ...后更新父节点
        }
    }
    // \sum_{i=1}^{n} a_i
    ll sum(ll n) {
        ll s = 0;
        for (; n; n -= lowbit(n)) s += (*this)[n];
        return s;
    };
    ll query(ll l, ll r) {
        return sum(r) - sum(l - 1);
    }
    void add(ll n, ll k) {
        for (; n < size(); n += lowbit(n)) (*this)[n] += k;
    };
};
```
#### 求逆序对

>在一个排列中，如果某一个较大的数排在某一个较小的数前面，就说这两个数构成一个 **逆序**（inversion）或反序。这里的比较是在自然顺序下进行的。
>  在一个排列里出现的逆序的总个数，叫做这个置换的 **逆序数**。排列的逆序数是它恢复成正序序列所需要做相邻对换的最少次数。因而，排列的逆序数的奇偶性和相应的置换的奇偶性一致。这可以作为置换的奇偶性的等价定义。

```c++
// https://oi-wiki.org/math/permutation/#%E9%80%86%E5%BA%8F%E6%95%B0
// 带离散化
ll inversion_discreet(vec& a) {
    map<ll, ll> inv;
    vec b = a;
    sort(b.begin(), b.end());
    b.resize(unique(b.begin(), b.end()) - b.begin());

    fenwick F(b.size());
    for (ll i = 0; i < b.size(); i++)
        inv[b[i]] = b.size() - i;

    ll ans = 0;
    for (ll x : a) {
        ll i = inv[x];
        ans += F.sum(i - 1);
        F.add(i, 1);
    }
    return ans;
}
// 不带离散化；注意上下界
ll inversion(vec& a) {
    fenwick F(*max_element(a.begin(),a.end()) + 1);
    ll ans = 0;
    for (ll i = a.size() - 1; i >= 0; i--) {
        ans += F.sum(a[i] - 1);
        F.add(a[i], 1);
    }
    return ans;
}
```
- https://codeforces.com/gym/105578/problem/D 2024 沈阳 D
  - https://codeforces.com/gym/105578/submission/312290991
  
    ```c++
    int main() {
        fast_io();
        /* El Psy Kongroo */
        ll t; cin >> t;
        while (t--) {
            ll n; cin >> n;
            vec a(n); for (ll& x: a ) cin >> x;
            vec b(n); for (ll& x: b ) cin >> x;
            ll invs = inversion(a) + inversion(b);
            // one can only *swap* when
            // a_i * b_i + a_j * b_j < a_i * b_j + a_j * b_i
            // (a_i - a_j) * (b_i - b_j) < 0
            // operating this on either a or b will *decrease* the number of inversions on either side
            // since the order would be more 'sorted' after the swap.
            // no. of swaps is exactly the number of inversions.
            // A starts first, odd ops -> A, even ops -> B
            cout << "AB"[invs % 2 == 0];
            for (ll i = 0; i < n - 1;i++) {
                char t; ll l,r,d; cin >> t >> l >> r >>d;
                // t -> a or b to 'shuffle' on. doesn't matter
                // since we care only about the *total* inversions here
                // shuffling an array in range [l,r] by d is simply
                // swapping (r - l) * d times.
                // we can simply add this to the total no. swaps
                invs += (r - l) * d;
                cout << "AB"[invs % 2 == 0];
            }
            cout << endl;
        }
        return 0;
    }
    ```
  
    


#### 支持不可差分查询模板

- 解释：https://oi-wiki.org/ds/fenwick/#树状数组维护不可差分信息
- 题目：https://acm.hdu.edu.cn/showproblem.php?pid=7463

```C++
struct fenwick {
    ll n;
    v a, C, Cm;
    fenwick(ll n) : n(n), a(n + 1), C(n + 1, -1e18), Cm(n + 1, 1e18) {}
    ll getmin(ll l, ll r) {
        ll ans = 1e18;
        while (r >= l) {
            ans = min(ans, a[r]); --r;
            for (; r - LOWBIT(r) >= l; r -= LOWBIT(r)) ans = min(ans, Cm[r]);
        }
        return ans;
    }
    ll getmax(ll l, ll r) {
        ll ans = -1e18;
        while (r >= l) {
            ans = max(ans, a[r]); --r;
            for (; r - LOWBIT(r) >= l; r -= LOWBIT(r)) ans = max(ans, C[r]);
        }
        return ans;
    }
    void update(ll x, ll v) {
        a[x] = v;
        for (ll i = x; i <= n; i += LOWBIT(i)) {
            C[i] = a[i]; Cm[i] = a[i];
            for (ll j = 1; j < LOWBIT(i); j *= 2) {
                C[i] = max(C[i], C[i - j]);
                Cm[i] = min(Cm[i], Cm[i - j]);
            }
        }
    }
};
```

#### 区间模板

- 解释：https://oi-wiki.org/ds/fenwick/#区间加区间和
- 题目：https://hydro.ac/d/ahuacm/p/Algo0304
```c++
int main() {
    std::ios::sync_with_stdio(false); std::cin.tie(0); std::cout.tie(0);
    /* El Psy Kongroo */
	ll n, m; cin >> n >> m;
	fenwick L(n + 1), R(n + 1);
	auto add = [&](ll l, ll r, ll v) {
		L.add(l, v); R.add(l, l * v);
		L.add(r + 1, -v); R.add(r + 1, -(r + 1) * v);
	};
	auto sum = [&](ll l, ll r) {
		return (r + 1) * L.sum(r) - l * L.sum(l - 1) - R.sum(r) + R.sum(l - 1);
	};
	for (ll i = 1; i <= n; i++) {
		ll x; cin >> x;
		add(i, i, x);
	}
	while (m--) {
		ll op; cin >> op;
		if (op == 1) {
			ll x, y, k; cin >> x >> y >> k;
			add(x, y, k);
		}
		else {
			ll x; cin >> x;
			cout << sum(x, x) << endl;
		}
	}
    return 0;
} 
```

## 优先队列（二叉堆）

> ```c++
> auto pp = PRED(ll, lhs > rhs);
> priority_queue<ll,vector<ll>,decltype(pp)> Q {pp};
> ```

## DSU

- 不考虑边权

```C++
struct dsu {
    vector<ll> pa;
    dsu(const ll size) : pa(size) { iota(pa.begin(), pa.end(), 0); }; // 初始时，每个集合都是自己的父亲
    inline bool is_root(const ll leaf) { return pa[leaf] == leaf; }
    inline ll find(const ll leaf) { return is_root(leaf) ? leaf : pa[leaf] = find(pa[leaf]); } // 路径压缩
    inline void unite(const ll x, const ll y) { pa[find(x)] = find(y); }
};
```

- 需要计算到根距离
  - https://codeforces.com/contest/2008/submission/280865425


```c++
struct dsu {
    vector<ll> pa, root_dis, set_size; // 父节点，到父亲距离，自己为父亲的集合大小
    dsu(const ll size) : pa(size), root_dis(size, 0), set_size(size, 1) { iota(pa.begin(), pa.end(), 0);  }; // 同上
    inline bool is_root(const ll leaf) { return pa[leaf] == leaf; }
    inline ll find(const ll leaf) { 
        if (is_root(leaf)) return leaf;
        const ll f = find(pa[leaf]);
        root_dis[leaf] += root_dis[pa[leaf]]; // 被压缩进去的集合到根距离变长
        pa[leaf] = f;
        return pa[leaf];
    }
    inline void unite(const ll x, const ll y) {
        if (x == y) return;
        const ll fx = find(x);
        const ll fy = find(y);
        pa[fx] = fy;
        root_dis[fx] += set_size[fy]; // 同 find
        set_size[fy] += set_size[fx]; // 根集合大小扩大
    }
    inline ll distance(const ll x, const ll y) {
        const ll fx = find(x);
        const ll fy = find(y);
        if (fx != fy) return -1; // 同最终父亲才可能共享路径
        return abs(root_dis[x] - root_dis[y]) - 1;
    }
};
```

# 字符串
## AC自动机
- https://www.luogu.com.cn/problem/P3796

```c++
struct AC {
    int tr[DIM][26], tot;
    int idx[DIM], fail[DIM], val[DIM], cnt[DIM];

    void init() {
        tot = 0;
        memset(tr, 0, sizeof(tr));
        memset(idx, 0, sizeof(idx));
        memset(fail, 0, sizeof(fail));
        memset(val, 0, sizeof(val));
        memset(cnt, 0, sizeof(cnt));
    }

    void insert(string const& s, int id) {
        int u = 0;
        for (char c : s) {
            if (!tr[u][c - 'A']) tr[u][c - 'A'] = ++tot;  // 如果没有则插入新节点
            u = tr[u][c - 'A'];                              // 搜索下一个节点
        }
        idx[u] = id;  // 以 u 为结尾的字符串编号为 idx[u]
    }


    void build() {
        queue<int> q;
        for (int i = 0; i < 26; i++)
            if (tr[0][i]) q.push(tr[0][i]);
        while (q.size()) {
            int u = q.front();
            q.pop();
            for (int i = 0; i < 26; i++) {
                if (tr[u][i]) {
                    fail[tr[u][i]] = tr[fail[u]][i];  // fail数组：同一字符可以匹配的其他位置
                    q.push(tr[u][i]);
                }
                else
                    tr[u][i] = tr[fail[u]][i];
            }
        }
    }

    void query(string const& s) {
        int u = 0;
        for (char c : s) {
            u = tr[u][c - 'A'];  // 转移
            for (int j = u; j; j = fail[j]) val[j]++;
        }
        for (int i = 0; i <= tot; i++)
            if (idx[i]) cnt[idx[i]] = val[i];
    }
}
```



## 字符串哈希
- https://acm.hdu.edu.cn/showproblem.php?pid=7433
- https://acm.hdu.edu.cn/contest/problem?cid=1125&pid=1011
```c++
// https://oi-wiki.org/string/hash/
namespace substring_hash
{
    const ull BASE = 3;
    static ull pow[DIM];
    void init() {
        pow[0] = 1;
        for (ll i = 1; i < DIM; i++) pow[i] = (pow[i - 1] * substring_hash::BASE);
    }
    struct hash : public uv {
        void init(string const& s) { init(s.c_str(), s.size()); }
        void init(const char* s) { init(s, strlen(s));}
        void init(const char* s, ll n) {
            resize(n + 1);
			(*this)[0] = 0;
			for (ll i = 0; i < n; i++) {
				(*this)[i + 1] = ((*this)[i] * BASE) + s[i];
			}
		}    
        // string[0, size()) -> query[l, r)
        ull query(ll l, ll r) const {
            return (*this)[r] - (*this)[l] * pow[r - l];
        }
    };
};
```

# 杂项

## 二分

```c++
// 找min
ll l = 0, r = INF;
while (l < r) {
    ll m = (l + r) >> 1;
    if (check(m)) r = m;
    else l = m + 1;
}
cout << l << endl;
// 找max
ll l = 0, r = INF;
while (l < r) {
    ll m = (l + r) >> 1;
    if (check(m)) l = m + 1;
    else r = m;
}
cout << l - 1 << endl;
```

## 常见贪心

- 线段最大交集

```c++
sort(p.begin(),p.end());
priority_queue<ll, vec, greater<>> pq;
for (auto [l,r] : p) {
    while (!pq.empty() && pq.top() < l) pq.pop();
    pq.push(r);
    res = max(res, (ll)pq.size());
}
```

## 置换环

- https://www.cnblogs.com/TTS-TTS/p/17047104.html

典中典之将长度$n$排列$p$中元素$i,j$交换$k$次使其变为排列$p'$，求最小$k$?

- 两个排列顺序连边；显然排列一致时图中有$n$个一元环
- 在一个环中交换一次可多分出一个环；记环的大小为$s$
- 显然，分成$n$个一元环即分环$s-1$次；记有$m$个环
- 可得 $k = \sum_{1}^{m}{s - 1} = n - m$

附题：https://codeforces.com/contest/2033/submission/287844212

- 不同于一般排序题，这里排列不需要完全一致；$p_i = i, p_i = p_{{i}_{i}}$皆可
- 意味着，最后要的环大小也可以是$2$，此时显然大小更优；更改$k$的计算为$k = \sum_{1}^{m}{\frac{s - 1}{2}}$即可

## 离散化

适用于大$a_i$但小$n$情形

- 在线`map`写法

```c++
map<ll, ll> pfx;        
for (auto [ai, bi] : a) {
    pfx[ai + 1] += 1;
    pfx[bi + 1] -= 1;           
}
for (auto it = next(pfx.begin()); it != pfx.end(); it++) 
    it->second += prev(it)->second;
auto query = [&](ll x) -> ll {
    if (pfx.contains(x)) return pfx[x];
    auto it = pfx.lower_bound(x);
    if (it != pfx.begin()) it = prev(it);
    else if (it->first != x) return 0; // 上界之前
    return it->second;
};        
```

- 离线`map`写法

```c++
map<ll, ll> R;
for (auto& ai : a) R[ai] = 1;
vec Ri; // kth-big
ll cnt = 0; for (auto& [x, i] : R) i = cnt++, Ri.push_back(x);
for (auto& [ai, bi] : a) ai = R[ai], bi = R[bi];
```

- 离线`set`写法
  - 注意该`set`若为STL set，复杂度($R(x)$) 实为$O(n)$
    - 详见 https://codeforces.com/blog/entry/123961
    - [TL;DR `std::distance`**对且仅对*随机***迭代器为$O(1)$操作](https://en.cppreference.com/w/cpp/iterator/distance)，其余迭代器(如果适用)皆为$O(n)$
    - 在 https://codeforces.com/contest/2051/submission/298511255 可见产生TLE
      - `map`解法(AC)：https://codeforces.com/contest/2051/submission/298511985

```c++
set<ll> Rs;
vector<II> a(n);
for (auto& ai : a) Rs.insert(ai);
vec Ri(R.begin(), R.end()); // kth-big
auto R = [&](ll x) -> ll { return distance(Rs.begin(), Rs.lower_bound(x)); };
```

## 前缀和

摘自：https://oi-wiki.org/basic/prefix-sum/

- 1D
  $$
  p(x) = \sum_{i=0}^{n}{a(i)}
  $$

  ```c++
  p[0] = a[0];
  for (ll i = 1;i < n;i++) p[i] = p[i - 1] + a[i];
  ```

- 2D
  $$
  S_{i,j} = \sum_{i'\le i}\sum_{j'\le j}A_{i',j'}.
  $$
  

  - 容斥$O(1)$解法
  $$
  p(x,y) = S_{i,j} = A_{i,j} + S_{i-1,j} + S_{i,j-1} - S_{i-1,j-1}
  $$

  ```c++
  for (ll i = 1; i <= n; i++)
  	for (ll j = 1; j <= m; j++)
  		p[i][j] = p[i][j - 1] + p[i - 1][j] - p[i - 1][j - 1] + a[i][j];
  ```

  

- N-D
> 显然的算法是，每次只考虑一个维度，固定所有其它维度，然后求若干个一维前缀和，这样对所有 $k$ 个维度分别求和之后，得到的就是 $k$ 维前缀和。

三维样例如下：
```c++
// Prefix-sum for 3rd dimension.
for (int i = 1; i <= N1; ++i)
  for (int j = 1; j <= N2; ++j)
    for (int k = 1; k <= N3; ++k) p[i][j][k] += p[i][j][k - 1];

// Prefix-sum for 2nd dimension.
for (int i = 1; i <= N1; ++i)
  for (int j = 1; j <= N2; ++j)
    for (int k = 1; k <= N3; ++k) p[i][j][k] += p[i][j - 1][k];

// Prefix-sum for 1st dimension.
for (int i = 1; i <= N1; ++i)
  for (int j = 1; j <= N2; ++j)
    for (int k = 1; k <= N3; ++k) p[i][j][k] += p[i - 1][j][k];

```

- https://codeforces.com/gym/105486/problem/B 2024 成都 B
  - https://codeforces.com/gym/105486/submission/312575722




## 二进制奇技淫巧

- https://codeforces.com/blog/entry/94470

```c++
a | b == (a ^ b) + (a & b);

(a ^ (a & b)) == ((a | b) ^ b);
(b ^ (a & b)) == ((a | b) ^ a);
((a & b) ^ (a | b)) == (a ^ b);

(a + b) == (a | b) + (a & b);
(a + b) == (a ^ b) + 2 * (a & b);

(a - b) == ((a ^ (a & b)) - ((a | b) ^ a));
(a - b) == (((a | b) ^ b) - ((a | b) ^ a));
(a - b) == ((a ^ (a & b)) - (b ^ (a & b)));
(a - b) == (((a | b) ^ b) - (b ^ (a & b)));
```


## bits/stdc++.h

```c++
#ifndef _GLIBCXX_NO_ASSERT
#include <cassert>
#endif
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <ciso646>
#include <climits>
#include <clocale>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if __cplusplus >= 201103L
#include <ccomplex>
#include <cfenv>
#include <cinttypes>
#include <cstdbool>
#include <cstdint>
#include <ctgmath>
#include <cwchar>
#include <cwctype>
#endif

// C++
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>

#if __cplusplus >= 201103L
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <forward_list>
#include <future>
#include <initializer_list>
#include <mutex>
#include <random>
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <system_error>
#include <thread>
#include <tuple>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#endif
```

# Treap

**ATTENTION:** 出门左转 https://caterpillow.github.io/byot 谢谢喵

又名**笛卡尔树**，**随机BST**；支持$log n$插入，删除，查找及闭包操作(`push_up`)

- https://cp-algorithms.com/data_structures/treap.html
- https://oi.baoshuo.ren/fhq-treap

## A. `std::set` 类容器

```c++
template<typename T> struct treap {
    struct node {
        T key; ll priority;
        ll l, r;
        // push_up maintains
        ll size;
    };
    vector<node> tree;
    vec free_list;
private:
    void push_up(ll o) {
        tree[o].size = tree[tree[o].l].size + tree[tree[o].r].size + 1;
    }
    II split_by_value(ll o, T const& key) { // 偏序 [<, >=]
        if (!o) return {0,0};
        if (tree[o].key < key) { // 左大右小
            auto [ll,rr] = split_by_value(tree[o].r, key);
            tree[o].r = ll;
            push_up(o);
            return {o,rr};
        } else {
            auto [ll,rr] = split_by_value(tree[o].l, key);
            tree[o].l = rr;
            push_up(o);
            return {ll,o};
        }
    }  
    ll merge(ll l, ll r) {
        if (!l || !r) return l + r;        
        if (tree[l].priority < tree[r].priority) // 保持堆序; 优先级小的在上
        {
            tree[l].r = merge(tree[l].r, r);
            push_up(l);
            return l;
        } else {
            tree[r].l = merge(l, tree[r].l);
            push_up(r);
            return r;
        }
    }
public:
    ll root = 1;
    treap(ll n): tree(n + 2), free_list(n - 1) {
        iota(free_list.rbegin(),free_list.rend(),2); // 1 is root
    }
    ll find(ll o, T const& key) {
        if (!o) return 0;
        if (tree[o].key == key) return o;
        if (tree[o].key > key) return find(tree[o].l, key);
        return find(tree[o].r, key);
    }
    ll find(T const& key) {
        return find(root, key);
    }
    void insert(ll& o, T const& key, ll const& priority) {
        auto [l,r] = split(o, key);
        ll next = free_list.back(); free_list.pop_back();
        tree[next].key = key, tree[next].priority = priority;
        l = merge(l, next);
        o = merge(l, r);
    }
    void insert(ll& o, T const& key) {
        insert(o, key, rand());
    }
    void insert(T const& key) {
        insert(root, key);
    }
    void erase(ll& o, T const& key) {
        auto [l,r] = split_by_value(o, key);
        ll next = find(r ,key);
        if (next) {
            free_list.push_back(next);
            r = merge(tree[next].l, tree[next].r);
        }
        o = merge(l, r);
    }
    void erase(T const& key) {
        erase(root,key);
    }
};
```

## B. 懒标记区间Treap

- 1. 提供**删除**操作，区间修改；不支持查找；支持RMQ

```c++
#include "bits/stdc++.h"
using namespace std;
#define PRED(T,X) [&](T const& lhs, T const& rhs) {return X;}
typedef long long ll; typedef unsigned long long ull; typedef double lf; typedef long double llf;
typedef __int128 i128; typedef unsigned __int128 ui128;
typedef pair<ll, ll> II; typedef vector<ll> vec;
template<size_t size> using arr = array<ll, size>;
const static void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0); }
const static ll lowbit(const ll x) { return x & -x; }
mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());
const ll DIM = 1e6;
const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const lf EPS = 1e-8;
template<typename T> struct treap {
    struct node {
        T key; // 应该为BST序 (但 lazy_add -> 无效故不实现find())
        ll priority; // heap
        // children
        ll l, r;
        // push_up maintains
        ll size;
        T sum;
        // push_down maintains (lazy)
        T lazy_add;
    };
    vector<node> tree;
    vec free_list;
private:
    void push_up(ll o) {
        tree[o].size = tree[tree[o].l].size + tree[tree[o].r].size + 1;
        tree[o].sum = tree[tree[o].l].sum + tree[tree[o].r].sum + tree[o].key;
    }
    void push_down(ll o) {
        if (tree[o].lazy_add) {
            if (tree[o].l) tree[tree[o].l].lazy_add += tree[o].lazy_add;
            if (tree[o].r) tree[tree[o].r].lazy_add += tree[o].lazy_add;
            tree[o].key += tree[o].lazy_add;
            tree[o].sum += tree[o].lazy_add * tree[o].size;
            tree[o].lazy_add = 0;
        }
    }
    II split_by_size(ll o, ll size) { // -> size:[k, n-k]
        if (!o) return {0,0};
        push_down(o);
        if (tree[tree[o].l].size >= size) {
            auto [ll,rr] = split_by_size(tree[o].l, size);
            tree[o].l = rr;
            push_up(o);
            return {ll,o};
        } else {
            auto [ll,rr] = split_by_size(tree[o].r, size - tree[tree[o].l].size - 1);
            tree[o].r = ll;
            push_up(o);
            return {o,rr};
        }
    }
    ll merge(ll l, ll r) {
        if (!l || !r) return l + r;
        push_down(l), push_down(r);
        if (tree[l].priority < tree[r].priority) // 保持堆序; 优先级小的在上
        {
            tree[l].r = merge(tree[l].r, r);
            push_up(l);
            return l;
        } else {
            tree[r].l = merge(l, tree[r].l);
            push_up(r);
            return r;
        }
    }
public:
    ll root = 0, top_p = 0;
    treap(ll n): tree(n + 2), free_list(n - 1) {
        iota(free_list.rbegin(),free_list.rend(),root + 1);
    }
    ll insert(ll pos, ll key) {
        auto [l,r] = split_by_size(root, pos);
        ll next = free_list.back(); free_list.pop_back();
        tree[next].key = tree[next].sum = key, tree[next].priority = rand();
        l = merge(l, next);
        return root = merge(l, r);
    }
    ll erase(ll pos) {
        auto [l,mid] = split_by_size(root, pos - 1);
        auto [erased, r] = split_by_size(mid, 1);
        free_list.push_back(erased);
        return root = merge(l,r);
    }
    ll range_add(ll v, ll L, ll R) {
        auto [p1, r] = split_by_size(root, R);
        auto [l, p2] = split_by_size(p1, L - 1);
        tree[p2].lazy_add += v;
        l = merge(l, p2);
        return root = merge(l ,r);
    }
    ll range_sum(ll L, ll R) {
        auto [p1, r] = split_by_size(root, R);
        auto [l, p2] = split_by_size(p1, L - 1);
        push_down(p2);
        ll sum = tree[p2].sum;
        l = merge(l, p2);
        root = merge(l ,r);
        return sum;
    }
};
int main() {
    fast_io();
    /* El Psy Kongroo */
    treap<ll> T(10);
    for (ll i = 1;i<=5;i++)T.insert(i - 1, i);
    T.range_add(1,1,2);
    ll ans = T.range_sum(1,3); // 8
    cout << ans << endl;
    T.erase(1);
    ans = T.range_sum(1,3); // 10
    cout << ans << endl;
    return 0;
}
```

- 2. 单点修改（无懒标记）

  参见 https://mos9527.com/posts/cp/gcd-problems/#p11373-czoi-r2%E5%A4%A9%E5%B9%B3

# FFT
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
    w_n^n = 1 \newline
    w_n^{\frac{n}{2}} = -1 \newline
    w_n^k \ne 1, 0 \lt k \lt n
    $$
    
  - 所有单位根和为$0$
    $$
    \sum_{k=0}^{n-1} w_n^k = 0
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

- 令 $A(x) = a_0 x^0 + a_1 x^1 + \dots + a_{n-1} x^{n-1}$, 按奇偶拆成两个子多项式

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

- 设 
$$
\left(y_k^0 \right)_{k=0}^{n/2-1} = \text{DFT}(A_0)
$$

$$
\left(y_k^1 \right)_{k=0}^{n/2-1} = \text{DFT}(A_1)
$$

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

## 余弦变换（DCT）

见下文实现；采用了以下$\text{DCT-II, DCT-III}$形式：

- DCT-2 及其正则化系数

$$
y_k = 2f \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right) \newline
\begin{split}f = \begin{cases}
\sqrt{\frac{1}{4N}} & \text{if }k=0, \\
\sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}\end{split}
$$

- DCT-3
  $$
  y_k = \frac{x_0}{\sqrt{N}} + \sqrt{\frac{2}{N}} \sum_{n=1}^{N-1} x_n
  \cos\left(\frac{\pi(2k+1)n}{2N}\right)
  $$
  

## Reference (lib/poly.hpp)

本文所提及的$\text{DFT/FFT/(F)NTT}$魔术总结如下，开箱即用。

```c++
#include <span>
#include <cmath>
#include <vector>
#include <complex>
#include <numbers>
constexpr double PI = std::numbers::pi;
inline long long binpow_mod(long long a, long long b, long long m, long long res = 1) {
    for (a %= m; b; b >>= 1) res = (b & 1) ? (res * a % m) : res, a = a * a % m;
    return res;
};
inline bool is_pow2(const size_t x) { return (x & (x - 1)) == 0; }

enum class transform_result {
    SUCCESS = 0,
    INVALID_SIZE = 1,
    INVALID_INPUT = 2,
    INVALID_COEFFICIENT = 3,
};
template <typename T = size_t> struct bit_reversal {
    std::vector<T> bit;
    explicit bit_reversal(const size_t n) : bit(n) {
        for (size_t i = 0; i < n; i++) {
            bit[i] = bit[i >> 1] >> 1;
            if (i & 1) bit[i] |= (n >> 1);
        }
    }
    size_t operator[](size_t i) const { return bit[i]; }
};
template <typename Complex = std::complex<double>, bool Invert = false> struct FFT {
    using value_t = Complex;
    using span_t = std::span<value_t>;
    class twiddle_factor {
        std::vector<Complex> omega;
    public:
        explicit twiddle_factor(const size_t n) : omega(n) {
            // \sum_{i=1}^{\log_2 n} 2^{i - 1} = 2^{\log_2 n} - 1 = n - 1
            for (size_t n_i = 2, i = 1; n_i <= n; n_i <<= 1) {
                Complex w_n = std::exp(Complex{ 0, -2 * PI / n_i });
                if constexpr (Invert) w_n = std::conj(w_n);
                Complex w_k = Complex{ 1, 0 };
                for (size_t k_i = 0; k_i < n_i / 2; k_i++) {
                    omega[i++] = w_k;
                    w_k *= w_n;
                }
            }
        }
        Complex get(const size_t n_i, const size_t k_i) const { return omega[n_i / 2 + k_i]; }
        const size_t size() const { return omega.size(); }
    };
private:
    const twiddle_factor omega;
    const bit_reversal<> rev;
public:
    const size_t size;
    FFT(const size_t n) : omega(n), rev(n), size(n) {};
    transform_result operator()(span_t a) const {
        const size_t n = a.size();
        if (!is_pow2(n) || size != n) return transform_result::INVALID_SIZE;
        for (size_t i = 0, r; i < n; i++)
            if (i < (r = rev[i])) std::swap(a[i], a[r]);
        for (size_t n_i = 2; n_i <= n; n_i <<= 1) {
            for (size_t i = 0; i < n; i += n_i) {
                for (size_t j = 0; j < n_i / 2; j++) {
                    Complex u = a[i + j], v = a[i + j + n_i / 2] * omega.get(n_i, j);
                    a[i + j] = u + v;
                    a[i + j + n_i / 2] = u - v;
                    if constexpr (Invert) a[i + j] /= 2, a[i + j + n_i / 2] /= 2;
                }
            }
        }
        return transform_result::SUCCESS;
    }
};
template <typename Complex = std::complex<double>> using IFFT = FFT<Complex, true>;
template <typename Integer = long long, bool Invert = false> struct NTT {
    using value_t = Integer;
    using span_t = std::span<value_t>;
    class twiddle_factor {
        std::vector<Integer> omega;
    public:
        explicit twiddle_factor(const size_t n, const Integer p, const Integer g) : omega(n) {
            // \sum_{i=1}^{\log_2 n} 2^{i - 1} = 2^{\log_2 n} - 1 = n - 1
            for (size_t n_i = 2, i = 1; n_i <= n; n_i <<= 1) {
                Integer w_n = binpow_mod(g, (p - 1) / n_i, p);
                if constexpr (Invert) w_n = binpow_mod(w_n, p - 2, p);
                Integer w_k = 1;
                for (size_t k_i = 0; k_i < n_i / 2; k_i++) {
                    omega[i++] = w_k;
                    w_k = w_n * w_k % p;
                }
            }
        }
        Integer get(const size_t n_i, const size_t k_i) const { return omega[n_i / 2 + k_i]; }
        const size_t size() const { return omega.size(); }
    };
private:
    const twiddle_factor omega;
    const bit_reversal<> rev;
public:
    const Integer p, g, inv2;
    const size_t size;
    NTT(const size_t n, const Integer p, const Integer g)
        : omega(n, p, g), rev(n), size(n), p(p), g(g), inv2(binpow_mod(2, p - 2, p)) {};
    transform_result operator()(span_t a) const {
        const size_t n = a.size();
        if (!is_pow2(n) || size != n) return transform_result::INVALID_SIZE;
        for (size_t i = 0, r; i < n; i++)
            if (i < (r = rev[i])) std::swap(a[i], a[r]);
        for (size_t n_i = 2; n_i <= n; n_i <<= 1) {
            for (size_t i = 0; i < n; i += n_i) {
                Integer w_k = 1;
                for (size_t j = 0; j < n_i / 2; j++) {
                    Integer u = a[i + j], v = a[i + j + n_i / 2] * omega.get(n_i, j);
                    a[i + j] = (u + v + p) % p;
                    a[i + j + n_i / 2] = (u - v + p) % p;
                    if constexpr (Invert) {
                        a[i + j] = (a[i + j] * inv2 % p + p) % p;
                        a[i + j + n_i / 2] = (a[i + j + n_i / 2] * inv2 % p + p) % p;
                    }
                }
            }
        }
        return transform_result::SUCCESS;
    }
};
template <typename Integer = long long> using INTT = NTT<Integer, true>;
template <typename Real = double, typename DFT = FFT<>> struct DCT2 {
    using value_t = Real;
    using span_t = std::span<value_t>;
    using complex = typename DFT::value_t;
    using work_area_span_t = std::span<complex>;
private:
    DFT dft;
    std::vector<complex> work_area;
    std::vector<complex> omega;
public:
    const size_t size;
    DCT2(const size_t n, bool create_work_area = true) : dft(n * 2), omega(n), size(n) {
        for (size_t m = 0, N = 2 * n; m < n; m++) {
            Real w_ang = -PI * m / N;
            omega[m] = std::exp(complex{ 0, w_ang });
        }
        if (create_work_area) work_area.resize(n * 2);
    };
    transform_result operator()(span_t a, work_area_span_t work_area) const {
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
        // https://zh.wikipedia.org/wiki/离散余弦变换#方法一[8]
        const size_t n = a.size(), N = 2 * n;
        if (!is_pow2(n) || size != n || work_area.size() != N) return transform_result::INVALID_SIZE;
        for (size_t i = 0; i < n; i++) work_area[i] = work_area[N - i - 1] = a[i];
        dft(work_area);
        const Real k2N = std::sqrt(N), k4N = std::sqrt(2.0 * N);
        for (size_t m = 0; m < n; m++) {
            complex w_n = omega[m];
            a[m] = (work_area[m] * w_n).real(); // imag = 0
            a[m] /= (m == 0 ? k4N : k2N);
        }
        return transform_result::SUCCESS;
    }
    transform_result operator()(span_t a) { return operator()(a, work_area); }
};
template <typename Real = double, typename IDFT = IFFT<>> struct DCT3 {
    using value_t = Real;
    using span_t = std::span<value_t>;
    using complex = typename IDFT::value_t;
    using work_area_span_t = std::span<complex>;
private:
    IDFT idft;
    std::vector<complex> work_area;
    std::vector<complex> omega;
public:
    const size_t size;
    DCT3(const size_t n, bool create_work_area = true) : idft(n), size(n), omega(n) {
        for (size_t m = 0, N = 2 * n; m < n; m++) {
            Real w_ang = PI * m / N;
            omega[m] = std::exp(complex{ 0, w_ang });
        }
        if (create_work_area) work_area.resize(n);
    };
    transform_result operator()(span_t a, work_area_span_t work_area) const {
        // https://dsp.stackexchange.com/questions/51311/computation-of-the-inverse-dct-idct-using-dct-or-ifft
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
        const size_t n = a.size(), N = 2 * n;
        if (!is_pow2(n) || size != n || work_area.size() != n) return transform_result::INVALID_SIZE;
        for (size_t i = 0; i < n; i++) work_area[i] = a[i];
        a[0] /= std::sqrt(2.0);
        const Real k2N = std::sqrt(N);
        for (size_t m = 0; m < n; m++) {
            complex w_n = omega[m];
            work_area[m] = a[m] * k2N * w_n;
        }
        idft(work_area);
        for (size_t m = 0; m < n / 2; m++) {
            a[m * 2] = work_area[m].real();
            a[m * 2 + 1] = work_area[n - m - 1].real();
        }
        return transform_result::SUCCESS;
    }    
    transform_result operator()(span_t a) { return operator()(a, work_area); }
};
```

## Problems

### A * B

- https://acm.hdu.edu.cn/showproblem.php?pid=1402

- 大整数乘法

- $10$ 进制数，各位数字从低到高为$d_i$可看作是多项式$A(x) = x^n \times d_n + ... + x^1 \times d_1 + x^0 \times d_0$于$x=10$时的解

- 两个十进制数即可看成是$A(x), B(x)$，求$A(x) * B(x)$即求$AB(x)$，由上文所述$\text{DFT,IDFT}$关系已知我们可以借此通过$\text{FFT}$在$O(n\log n)$时间计算这样的数

- 由于是$10$进制，最后多项式的系数即对应$x=10$解；注意进位。

```c++
void carry(Poly::IVec& a, ll radiax) {
    for (ll i = 0; i < a.size() - 1; i++)
        a[i + 1] += a[i] / radiax,
        a[i] %= radiax;
}
int main() {
    fast_io();
    /* El Psy Kongroo */
    string a, b;
    while (cin >> a >> b)
    {
        {
            Poly::IVec A(a.size()), B(b.size());
            for (ll i = 0; i < a.size(); i++)
                A[i] = a[a.size() - 1 - i] - '0';
            for (ll i = 0; i < b.size(); i++)
                B[i] = b[b.size() - 1 - i] - '0';
            ll len = Poly::conv::convolve(A, B);
            carry(A, 10u);
            for (ll i = len - 1, flag = 0; i >= 0; i--) {
                flag |= A[i] != 0;
                if (flag || i == 0)
                    cout << (ll)A[i];
            }
            cout << endl;
        }
    }
}
```

### A + B 频率

- https://open.kattis.com/problems/aplusb

- 给定整数序列$A$,$B$，求$a \in A, b \in B, a + b$的结果可能及数量

- 考虑这样转化成多项式问题：令 $ P_a(x) = \sum x^{A_i}, P_b(x) = \sum x^{B_i} $

- 给定例子$a = [1,~ 2,~ 3], b = [2,~ 4]$，这样构造的$P_aP_b$有
  $$
  (1 x^1 + 1 x^2 + 1 x^3) (1 x^2 + 1 x^4) = 1 x^3 + 1 x^4 + 2 x^5 + 1 x^6 + 1 x^7
  $$

- 如此发现指数对应系数即各种可能数量

### 循环数乘

- 给定长$n$整数序列$A$,$B$，令$C_{p,i} = B_{(i + p) \mod n}$,求任意$A \cdot C_p$的值

- 回顾多项式相乘的系数即这样的包络
  $$
  c[k] = \sum_{i+j=k} a[i] b[j]
  $$
  
- 令$A$逆序，然后补$n$个$0$；令$B$补$B$本身

- 即$A_i = 0 (i \gt n - 1)$, 可见此时我们有

$$
c[k] = \sum_{i+j=k} a[i] b[j] = \sum_{i=0}^{n-1} a[i] b[k-i]
$$

- 对$i + k > n$, $b[(i+k) \% n] = b[i + k - n + 1]$；上式即为$p = k - n + 1$时结果

- 即$c[p + n - 1]$对应$p$时原$A \cdot C_p$值

### 字串匹配

- 给定字串$S$和模式串$P$，每个字符$C_i\in[0,26]$,统计$P$在$S$中出现总次数
  - 构造多项式$A(x) = \sum a_i x^i$，其中$a_i = e^{\frac{2 \pi S_i}{26}}$
  - 令$S$为其倒序，构造多项式$B(x)=\sum b_i x^i$,其中$b_i = e^{-\frac{2 \pi P_i}{26}}$
- 注意包络后

$$
c_{m-1+i} = \sum_{j = 0}^{m-1} a_{i+j} \cdot b_{m-1-j} = \sum_{j=0}^{m-1}e^{\frac{2 \pi S_{i+j} - 2\pi P_j}{26}}
$$

显然若匹配则$e^{\frac{2 \pi S_{i+j} - 2\pi P_j}{26}} = e^0 = 1$，那么全部匹配当且仅当$c_{m-1+i} = m$，模式串$P$在$S_i$处有出现

#### 附：部分匹配

- 设$P$中部分字符任意，则倒序后可令这些位置多项式系数$b_i=0$；设有$x$个这种位置
- 回顾上式易知当且仅当匹配到这些系数时有$c_i = \sum_{j=0}^{m-1-x} e^{\cdots} + \sum_0^x 0$
- 显然，当$c_{m-1+i} = m - x$，带任意匹配模式的模式串$P$在$S_i$处有出现

## 图像处理？？？

> 正常人应该用[FFTW](https://www.fftw.org/) - 但可惜你是ACM选手。

### lib/image.hpp

>  STB is All You Need.

```c++
#pragma once
#ifndef _POLY_HPP
#include "poly.hpp"
#endif
#define _IMAGE_HPP
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
namespace Image {
using Texel = unsigned char;
using Image = std::vector<Poly::RVec2>;
using Poly::ll, Poly::lf;
// Channels, Height, Width
inline std::tuple<ll, ll, ll> image_size(const Image& img) {
    if (!img.size()) return { 0, 0, 0 };    
    auto [h, w] = Poly::utils::size_of(img[0]);    
    return { img.size(), h, w };
}
// Assuming 8bit sRGB space
template <typename Texel> Image from_texels(const Texel* img_data, int w, int h, int nchn) {
    Image chns(nchn, Poly::RVec2(h, Poly::RVec(w)));
    for (ll y = 0; y < h; ++y)
        for (ll x = 0; x < w; ++x)
            for (ll c = 0; c < nchn; ++c) chns[c][y][x] = img_data[(y * w + x) * nchn + c];
    return chns;
}
vector<Texel> to_texels(const Image& res, int& w, int& h, int& nchn) {
    std::tie(nchn, h, w) = image_size(res);
    vector<Texel> texels(w * h * nchn);
    for (ll y = 0; y < h; ++y)
        for (ll x = 0; x < w; ++x)
            for (ll c = 0; c < nchn; ++c) {
                ll t = std::round(res[c][y][x]);
                texels[(y * w + x) * nchn + c] = max(min(255ll, t), 0ll);
            }
    return texels;
}
inline Image from_file(const char* filename, bool hdr = false) {
    int w, h, nchn;
    Texel* img_data = stbi_load(filename, &w, &h, &nchn, 0);
    assert(img_data && "cannot load image");
    auto chns = from_texels(img_data, w, h, nchn);
    stbi_image_free(img_data);
    return chns;
}
inline void to_file(const Image& res, const char* filename, bool hdr = false) {
    int w, h, nchn;
    auto texels = to_texels(res, w, h, nchn);
    int success = stbi_write_png(filename, w, h, nchn, texels.data(), w * nchn);
    assert(success && "image data failed to save!");
}
inline Image create(int nchn, int h, int w, lf fill) {
    Image image(nchn);
    for (auto& ch : image) Poly::utils::resize(ch, { h, w }, fill);
    return image;
}
inline Poly::RVec2& to_grayscale(Image& image) {
    auto [nchn, h, w] = image_size(image);
    auto& ch0 = image[0];
    // L = R * 299/1000 + G * 587/1000 + B * 114/1000
    for (ll c = 0; c < nchn; c++) {
        for (ll i = 0; i < h; i++) {
            for (ll j = 0; j < w; j++) {
                if (c == 0 && nchn != 1) ch0[i][j] *= 0.299;
                if (c == 1) ch0[i][j] += image[1][i][j] * 0.587;
                if (c == 2) ch0[i][j] += image[2][i][j] * 0.144;
            }
        }
    }
    return ch0;
}
} // namespace Image
```

### 二维包络

> 想玩转超大kernel还想不等半年？？

- 设原图像$A[N,M]$,包络核$B[K,L]$空间上进行包络有时间复杂度$O(N * M * K * L)$
- 利用$\text{FFT}$则为$O(N * M * log(N * M))$

#### 高斯模糊

```c++
#include "bits/stdc++.h"
using namespace std;
typedef long long ll; typedef double lf; typedef pair<ll, ll> II; typedef vector<ll> vec;
const inline void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0u); cout.tie(0u); }
const lf PI = acos(-1);

#include "lib/poly.hpp"
#include "lib/image.hpp"
Poly::RVec2 gaussian(ll size, lf sigma) {
    Poly::RVec2 kern(size, Poly::RVec(size));
    lf sum = 0.0;
    ll x0y0 = size / 2;
    lf sigma_sq = sigma * sigma;
    lf term1 = 1.0 / (2.0 * PI * sigma_sq);
    for (ll i = 0; i < size; ++i) {
        for (ll j = 0; j < size; ++j) {
            ll x = i - x0y0, y = j - x0y0;
            lf term2 = exp(-(lf)(x * x + y * y) / (2.0 * sigma_sq));
            kern[i][j] = term1 * term2;
            sum += kern[i][j];
        }
    }
    for (ll i = 0; i < size; ++i)
        for (ll j = 0; j < size; ++j)
            kern[i][j] /= sum;
    return kern;
}
const auto __Exec = std::execution::par_unseq;
int main() {
    const char* input = "data/input.png";
    const char* output = "data/output.png";
    const int kern_size = 25;
    const lf kern_sigma = 7.0;

    Poly::RVec2 kern = gaussian(kern_size, kern_sigma);
    auto image = Image::from_file(input);
    {
        auto [nchn,h,w] = Image::image_size(image);
        cout << "preparing image w=" << w << " h=" << h << " nchn=" << nchn << endl;
        for_each(__Exec, image.begin(), image.end(), [&](auto& ch) {
            cout << "channel 0x" << hex << &ch << dec << endl;
            auto c_ch = Poly::utils::as_complex(ch), k_ch = Poly::utils::as_complex(kern);
            Poly::conv::convolve2D(c_ch, k_ch, __Exec);
            ch = Poly::utils::as_real(c_ch);
        });
    }
    {
        Image::to_file(image, output);
        auto [nchn,h,w] = Image::image_size(image);
        cout << "output image w=" << w << " h=" << h << " nchn=" << nchn << endl;
    }
    return 0;
}
```

- 测试样例

  | 输入                                                         | 输出                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![input](/image-github/434868266-52c8860a-c118-406c-9ef1-2211b9e5ecc9.png) | ![output](/image-github/434868287-7f7bfe51-db49-4295-ab3a-76751c395c1b.png) |

#### Wiener 去卷积（逆包络）

> 2025，Codeforces 4.1 H题见

- https://en.wikipedia.org/wiki/Wiener_deconvolution
- Wiener 去卷积可表示为

$$
\ F(f) = \frac{H^\star(f)}{ |H(f)|^2 + N(f) }G(f)= \frac{H^\star(f)}{ H(f)\times H^\star(f) + N(f) }G(f)
$$

- 都在频域下，其中$F$为原图像，$G$为包络后图像，$H$为卷积核，$N$为噪声函数

```c++
#include "bits/stdc++.h"
using namespace std;
typedef long long ll; typedef double lf; typedef pair<ll, ll> II; typedef vector<ll> vec;
const inline void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0u); cout.tie(0u); }
const lf PI = acos(-1);

#include "lib/poly.hpp"
#include "lib/image.hpp"
Poly::RVec2 gaussian(ll size, lf sigma) {
	Poly::RVec2 kern(size, Poly::RVec(size));
	lf sum = 0.0;
	ll x0y0 = size / 2;
	lf sigma_sq = sigma * sigma;
	lf term1 = 1.0 / (2.0 * PI * sigma_sq);
	for (ll i = 0; i < size; ++i) {
		for (ll j = 0; j < size; ++j) {
			ll x = i - x0y0, y = j - x0y0;
			lf term2 = exp(-(lf)(x * x + y * y) / (2.0 * sigma_sq));
			kern[i][j] = term1 * term2;
			sum += kern[i][j];
		}
	}
	for (ll i = 0; i < size; ++i)
		for (ll j = 0; j < size; ++j)
			kern[i][j] /= sum;
	return kern;
}
const auto exec = std::execution::par_unseq;
int main() {
    const char* input = "data/blurred.png";
    const char* output = "data/deblur.png";
    const int kern_size = 25;
    const lf kern_sigma = 7.0;

    Poly::RVec2 kern = gaussian(kern_size, kern_sigma);
    auto wiener = [&](Poly::RVec2& ch, Poly::RVec2 kern, lf noise = 5e-6) {
        II og_size = { ch.size(), ch[0].size() };
        II size = Poly::utils::to_pow2({ ch.size(), ch[0].size() }, { kern.size(), kern[0].size() });
        auto [N, M] = size;
        Poly::utils::resize(ch, size, 255.0);
        // 需要窗口
        Poly::CVec2 img_fft = Poly::utils::as_complex(ch);
        ch = Poly::utils::as_real(img_fft);
        Poly::transform::DFT2(img_fft, exec);
        Poly::CVec2 kern_fft = Poly::utils::as_complex(kern);
        Poly::utils::resize(kern_fft, size);
        Poly::transform::DFT2(kern_fft, exec);
        for (ll i = 0; i < N; i++)
            for (ll j = 0; j < M; j++) {
                auto kern_fft_conj = conj(kern_fft[i][j]);
                auto denom = kern_fft[i][j] * kern_fft_conj + noise;
                img_fft[i][j] = (img_fft[i][j] * kern_fft_conj) / denom;
            }
        Poly::transform::IDFT2(img_fft, exec);
        ch = Poly::utils::as_real(img_fft);
        Poly::utils::resize(ch, og_size);
    };
    auto image = Image::from_file(input);
    {
        auto [nchn,h,w] = Image::image_size(image);
        cout << "preparing image w=" << w << " h=" << h << " nchn=" << nchn << endl;
        for_each(exec, image.begin(), image.end(), [&](auto& ch) {
            cout << "channel 0x" << hex << &ch << dec << endl;
            wiener(ch, kern);
        });
    }
    {
        Image::to_file(image, output);
        auto [nchn,h,w] = Image::image_size(image);
        cout << "output image w=" << w << " h=" << h << " nchn=" << nchn << endl;
    }
    return 0;
}
```
- 测试样例

  | 输入                                                         | 输出                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![output](/image-github/435036278-13695e56-aa4e-4352-a90d-07ca14620479.png) | ![deblur](/image-github/435036293-38ad63d7-a12a-4032-8d08-3fd7e872d752.png) |

## 图像压缩 （DCT）

JPEG格式采用的即为$8\times8$ DCT块变换，丢掉高频信息（频域$u,v$大位置）后量化存储

这里演示一种naive的压缩方式，和[MATLAB](https://ww2.mathworks.cn/help/images/discrete-cosine-transform.html)所述图像压缩样例一致，以下面矩阵掩盖系数：
$$
\text{mask} =
\begin{bmatrix}
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \newline
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \newline
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \newline
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \newline
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \newline
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \newline
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \newline
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \newline
\end{bmatrix}
$$

```c++
#include "bits/stdc++.h"
using namespace std;
typedef long long ll;typedef double lf;typedef pair<ll, ll> II; typedef vector<ll> vec;
const lf PI = acos(-1);
#include "lib/image.hpp"
#include "lib/poly.hpp"
auto block8x8 = [](auto&& op, auto& src) { return Poly::block::block2D(op, src, 8, 8, std::execution::par_unseq); };
int main() {
    /* image to dct */
    auto image = Image::from_file("data/cameraman.png");
    auto& source = Image::to_grayscale(image);
    auto [nchn, h, w] = Image::image_size(image);
    Poly::utils::resize(source, { Poly::utils::to_pow2(h), Poly::utils::to_pow2(w) });
    cout << "Processing..." << w << "x" << h << endl;
    block8x8([](Poly::RVec2& rect) { Poly::transform::DCT2(rect, execution::seq); }, source);
    cout << "Saving." << endl;
    Image::to_file(Image::Image{ source }, "data/dct.png");
    cout << "Dropping coefficents." << endl;
    block8x8(
        [](Poly::RVec2& rect) {
            auto [n, m] = Poly::utils::size_of(rect);
            for (ll i = 0; i < n; i++) for (ll j = 0; j < m; j++)
                if (i >= 4 || j >= (n / 2 - i)) rect[i][j] = 0;
        },
        source);
    Image::to_file(Image::Image{ source }, "data/dct_dropped.png");
    cout << "Restoring." << endl;
    block8x8([](Poly::RVec2& rect) { Poly::transform::IDCT2(rect, execution::seq); }, source);
    cout << "Saving." << endl;
    Image::to_file(Image::Image{ source }, "data/idct.png");
    return 0;
}

```



| 输入                                                         | DCT                                                          | 丢掉三角阵的DCT                                              | IDCT                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![cameraman](/image-github/435438684-40514c3d-3866-4a68-b47e-d8ac54b0f2ad.png) | ![dct](/image-github/435438689-920e7453-3831-401a-aeb9-3c380cea524f.png) | ![dct_dropped](/image-github/435438694-4c715305-258e-4a9b-9066-307963f54375.png) | ![idct](/image-github/435438681-5d2619c4-b919-46d5-8822-a775d9b54779.png) |

# GCD

## 691C. Row GCD
> You are given two positive integer sequences $a_1, \ldots, a_n$ and $b_1, \ldots, b_m$. For each $j = 1, \ldots, m$ find the greatest common divisor of $a_1 + b_j, \ldots, a_n + b_j$.

- **引理：** $gcd(x,y) = gcd(x,y-x)$
- **引理：** 可以拓展到 **数组 $gcd$数值上等于数组差分 $gcd$ **；证明显然，略
  - 注意该命题在**数组及其差分上取子数组时**上并不成立，如 991F.
  	- *Typora怎么快速加页面内链接...*

- 记$g_{pfx} = gcd(a_2-a_1,a_3-a_2,...a_n-a_{n-1})$
- 于本题利用$gcd(a_1+b_1,a_2+b_1,...a_n+b_1) = gcd(a_1+b1,a_2-a_1,a_3-a_2,...a_n-a_{n-1}) = gcd(a_1 + b_1, g_{pfx})$即可

```c++
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll n,m; cin >> n >> m;
    vector<ll> a(n);
    for (ll& x: a) cin >> x;
    // gcd(a_1+b_1,a_2+b_1,...a_n+b_1) -> gcd(a_1+b1,a_2-a_1,a_3-a_2,...a_n-a_{n-1})
    for (ll i = n - 1;i >= 1;i--) a[i] -= a[i-1], a[i] = abs(a[i]);
    ll ans = n > 1 ? a[1] : 0;
    for (ll i = 1;i < n;i++) ans = gcd(ans,a[i]);
    for (ll i = 0,x;i < m;i++) cin >> x, cout << (ans ? gcd(ans,a[0]+x) : a[0]+x) << ' ';
    return 0;
}
```

## 991F. Maximum modulo equality

>You are given an array $a$ of length $n$ and $q$ queries $l$, $r$.
>For each query, find the maximum possible $m$, such that all elements $a_l$, $a_{l+1}$, ..., $a_r$ are equal modulo $m$. In other words, $a_l \bmod m = a_{l+1} \bmod m = \dots = a_r \bmod m$, where $a \bmod b$ — is the remainder of division $a$ by $b$. In particular, when $m$ can be infinite, print $0$.

- **引理:** 模$m$意义下相等 ($x \mod m = y \mod m$) $\iff$ $|x-y| \mod m = 0$
- 故本题$a_l \bmod m = a_{l+1} \bmod m = \dots = a_r \bmod m \iff |a_{l+1} - a_{l}| \mod m = |a_{l+2} - a_{l}| \mod m = ... = |a_{r} - a_{r-1}| \mod m = 0$
- 很显然这里最大的$m$即为差分数组的$gcd$
- 处理query实现$gcd$ RMQ即可；注意由（2）边界应该为$[l+1,r]$；$l=r$情形即为$m$可取$\inf$

```c++
template<typename T> struct segment_tree {
    struct node {
        ll l, r; // 区间[l,r]
        T gcd_v;
        ll length() const { return r - l + 1; }
        ll mid() const { return (l + r) / 2; }
    };
    vector<node> tree;
private:
    ll begin = 1, end = 1;
    void push_up(ll o) {
        // 向上传递
        ll lc = o * 2, rc = o * 2 + 1;
        tree[o].gcd_v = gcd(tree[lc].gcd_v,tree[rc].gcd_v);
    }
    node query(ll o, ll l, ll r) {
        ll lc = o * 2, rc = o * 2 + 1;
        if (tree[o].l == l && tree[o].r == r) return tree[o];
        ll mid = tree[o].mid();
        if (r <= mid) return query(lc, l, r);
        else if (mid < l) return query(rc, l, r);
        else {
            node p = query(lc, l, mid);
            node q = query(rc, mid + 1, r);
            return {
                l, r,
                gcd(p.gcd_v, q.gcd_v)
            };
        }
    }
    void build(ll o, ll l, ll r, const T* src = nullptr) {
        ll lc = o * 2, rc = o * 2 + 1;
        tree[o] = {};
        tree[o].l = l, tree[o].r = r;
        if (l == r) {
            if (src) tree[o].gcd_v = tree[o].gcd_v = src[l];
            return;
        }
        ll mid = tree[o].mid();
        build(lc, l, mid, src);
        build(rc, mid + 1, r, src);
        push_up(o);
    }
    void build(const T* src = nullptr) { build(begin, begin, end, src); }
public:
    node range_query(ll l, ll r) { return query(begin, l, r); }
    T range_gcd(ll l, ll r) { return range_query(l, r).gcd_v; }
    void reserve(const ll n) { tree.reserve(n); }
    void reset(const ll n) { end = n; tree.resize(end << 2); build(); }
    void reset(const vector<T>& src) {
        end = src.size(); tree.resize(end << 2);
        build(src.data() - 1);
    }
    explicit segment_tree() {};
    explicit segment_tree(const ll n) : begin(1), end(n) { reset(n); }
};

int main() {
    fast_io();
    /* El Psy Kongroo */
    ll t; cin >> t;
    while (t--) {
        ll n, q; cin >> n >> q;
        vector<ll> a(n); for (ll& x : a) cin >> x;
        for (ll i = n - 1;i >= 1;i--) a[i] -= a[i-1], a[i] = abs(a[i]);
        segment_tree<ll> seg(n); seg.reset(a);
        while (q--) {
            ll l,r; cin >> l >> r; l++;
            ll ans = l <= r ? seg.range_gcd(l,r) : 0;
            cout << ans << ' ';
        }
        cout << endl;
    }
    return 0;
}

```

## P11373 「CZOI-R2」天平

> 你有 $n$ 个**砝码组**，编号为 $1$ 至 $n$。对于第 $i$ 个**砝码组**中的砝码有共同的正整数质量 $a_i$，每个**砝码组**中的**砝码**数量无限。
> 其中，有 $q$ 次操作：
> - `I x v`：在第 $x$ 个**砝码组**后新增一组单个**砝码**质量为 $v$ 的**砝码组**，当 $x=0$ 时表示在最前面新增；
> - `D x`：删除第 $x$ 个**砝码组**；
> - `A l r v`：把从 $l$ 到 $r$ 的所有**砝码组**中的砝码质量加 $v$；
> - `Q l r v`：判断能否用从 $l$ 到 $r$ 的**砝码组**中的砝码，称出质量 $v$。每个砝码组中的砝码可以使用任意个，也可以不用。
> 对于操作 `I` 和 `D`，操作后编号以及 $n$ 的值自动变化。
> 称一些**砝码**可以称出质量 $v$，当且仅当存在将这些砝码分别放在天平两边的摆放方法，使得将 $1$ 个质量为 $v$ 的物体摆放在某边可以让天平平衡。

- **引理：** **裴蜀等式**（英语：Bézout's identity），或**丢番图方程一次特殊情况**：设$a_1, \cdots a_n$为$n$个整数，$d$是它们的最大公约数，那么存在整数$x_1, \cdots x_n$ 使得 $x_1\cdot a_1 + \cdots x_n\cdot a_n = d$

- 对操作`Q`，即询问$l,r$中的整数$x_i$能否找到系数$a_i$构成 $x_1\cdot a_1 + \cdots x_n\cdot a_n = kd = v \to v \mod gcd(a_1,...,a_n) = 0 $

- 对操作`I,D,A`维护个平衡树/Treap吧
  - 思路上和[线段树 Subtask 3](https://mos9527.com/posts/cp/segment-tree-problems/#p11373-czoi-r2%E5%A4%A9%E5%B9%B3)基本一致
  - 额外考虑对**增，删**的维护；简单操作相邻差分值即可
  - 由于是单点修改，同样不需要`push_down`传递懒标记
    - 洛谷b评测为什么不显示编译警告= =; `insert`没`return`直接RTE了无数发...


```c++
#include "bits/stdc++.h"
using namespace std;
#define PRED(T,X) [&](T const& lhs, T const& rhs) {return X;}
typedef long long ll; typedef unsigned long long ull; typedef double lf; typedef long double llf;
typedef __int128 i128; typedef unsigned __int128 ui128;
typedef pair<ll, ll> II; typedef vector<ll> vec;
template<size_t size> using arr = array<ll, size>;
const static void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0); }
const static ll lowbit(const ll x) { return x & -x; }
mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());
const ll DIM = 3e5;
const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const lf EPS = 1e-8;
template<typename T> struct treap {
    struct node {
        ll priority; // heap序
        // children
        ll l, r;
        // push_up maintains
        ll size;
        T key; // (带修改；无法保证BST性质)
        T sum; // 差分和
        T gcd; // 差分gcd
    };
    vector<node> tree;
    vec free_list;
private:
    void push_up(ll o) {
        tree[o].size = tree[tree[o].l].size + tree[tree[o].r].size + 1;
        tree[o].sum = tree[o].key + tree[tree[o].l].sum + tree[tree[o].r].sum;
        tree[o].gcd = gcd(abs(tree[o].key), gcd(abs(tree[tree[o].l].gcd), abs(tree[tree[o].r].gcd)));
    }
    II split_by_size(ll o, ll size) { // -> size:[k, n-k]
        if (!o) return { 0,0 };
        if (tree[tree[o].l].size >= size) {
            auto [ll, rr] = split_by_size(tree[o].l, size);
            tree[o].l = rr;
            push_up(o);
            return { ll,o };
        }
        else {
            auto [ll, rr] = split_by_size(tree[o].r, size - tree[tree[o].l].size - 1);
            tree[o].r = ll;
            push_up(o);
            return { o,rr };
        }
    }
    ll merge(ll l, ll r) {
        if (!l || !r) return l + r;
        if (tree[l].priority < tree[r].priority) // 保持堆序; 优先级小的在上
        {
            tree[l].r = merge(tree[l].r, r);
            push_up(l);
            return l;
        }
        else {
            tree[r].l = merge(l, tree[r].l);
            push_up(r);
            return r;
        }
    }
public:
    ll root = 0, top_p = 0;
    treap(ll n) : tree(n + 2), free_list(n - 1) {
        iota(free_list.rbegin(), free_list.rend(), root + 1);
    }
    ll insert(ll pos, ll key) {
        auto [l, r] = split_by_size(root, pos);
        ll index = free_list.back(); free_list.pop_back();
        tree[index].key = tree[index].sum = tree[index].gcd = key, tree[index].priority = rand(), tree[index].size = 1;
        l = merge(l, index);
        return root = merge(l, r);
    }
    ll erase(ll pos) {
        auto [l, mid] = split_by_size(root, pos - 1);
        auto [erased, r] = split_by_size(mid, 1);
        free_list.push_back(erased);
        tree[erased] = node{};
        return root = merge(l, r);
    }
    ll add(ll v, ll pos) {
        if (range_query(pos, pos).size == 0) insert(pos, 0);
        auto [l, mid] = split_by_size(root, pos - 1);
        auto [p, r] = split_by_size(mid, 1);
        // 单点改
        tree[p].key += v; tree[p].sum = tree[p].gcd = tree[p].key;
        l = merge(l, p);
        return root = merge(l, r);
    }
    node range_query(ll L, ll R) {
        auto [p1, r] = split_by_size(root, R);
        auto [l, p2] = split_by_size(p1, L - 1);
        node res = tree[p2];
        l = merge(l, p2);
        root = merge(l, r);
        return res;
    }
};
treap<ll> T(DIM);
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll n, q; cin >> n >> q;
    vec src(n); for (ll& x : src) cin >> x;
    for (ll i = n - 1;i >= 1;i--) src[i] -= src[i-1];
    for (ll i = 0; i < n; i++) T.insert(i, src[i]);
    while (q--) {
        char op; cin >> op;
        switch (op)
        {
            case 'I':
            {
                ll x, v; cin >> x >> v;
                ll
                    prev = x ? T.range_query(1, x).sum : 0,
                    next = T.range_query(1, x + 1).sum;
                T.add(-(next - prev) + (next - v), x + 1);
                T.insert(x, v - prev);
                break;
            }
            case 'D':
            {
                ll x; cin >> x;
                ll
                    prev = x > 1 ? T.range_query(1, x - 1).sum : 0,
                    curr = T.range_query(1, x).sum;
                T.erase(x);
                T.add(curr - prev, x);
                break;
            }
            case 'A':
            {
                ll l, r, v; cin >> l >> r >> v;
                T.add(v, l);
                T.add(-v ,r + 1);
                break;
            }
            case 'Q':
            default:
            {
                ll l, r, v; cin >> l >> r >> v;
                ll a = T.range_query(1,l).sum;
                ll b_gcd = l != r ? T.range_query(l + 1, r).gcd : 0LL;
                ll range_gcd = gcd(a,b_gcd);
                if (v % range_gcd == 0) cout << "YES\n";
                else cout << "NO\n";
                break;
            }
        }
    }
    return 0;
}
```

### 附：一些 Trick

- $a$是否存在子序列$s$使得$gcd(s_1, s_2, ...) = k$

  - 例：https://codeforces.com/contest/2084/problem/B

  ```c++
  for (ll i = 1; i < n; i++) {
      if (a[i] % k == 0) a[i] /= k;
      else a[i] = 0;
  }
  ll g = 0;
  for (ll i = 1; i < n; i++) g = gcd(g, a[i]);
  // g == 1 即可， 否则不存在
  ```

  
  # 字符串哈希
  
  ## [1. Candy Rush](https://codeforces.com/gym/104736/problem/C)
  
  TL;DR - 串哈希加速比对 + big hash trick
  
  [2023-2024 ACM-ICPC Latin American Regional Programming Contest](https://codeforces.com/gym/104736)
  
  官方题解：https://codeforces.com/gym/104736/attachments/download/22730/editorial.pdf
  
  ---
  
  1. 取$C_i$出现频率为向量做前缀和后可发现有效区间频率差为$k \cdot 1_n$；直接比对平摊复杂度为$O(nk\log{n})$,$O(k)$来自于比对本身
  
  2. 而比对可以转换为$O(1)$形式，trick如下：
  
  - 若 $a,b$ 对应区间有效，一定有：$freq_b = k \cdot 1_n + freq_a$
  - 而在做前缀和是，可以及时消掉$k \cdot 1_n$项：即为全非$0$时全$-1$
  - 此时的比对即变成$ hash(freq_b) = hash(freq_a) $
  - 造一个能$O(1)$更新的$hash$即可让最终复杂度可以变为优秀的$O(nlogn)$
  
  题解$hash$为$(base^i \cdot c_i)\%MOD$ - 避免碰撞可以采用更大int类型或模数，或同code开`array`于不同base计算后比较
  
  ---
  
  
  ```c++
  #pragma GCC optimize("O3","unroll-loops","inline")
  #include "bits/stdc++.h"
  using namespace std;
  #define PRED(T,X) [](T const& lhs, T const& rhs) {return X;}
  typedef long long ll; typedef unsigned long long ull; typedef double lf; typedef long double llf;
  typedef __int128 i128; typedef unsigned __int128 ui128;
  typedef pair<ll, ll> II; typedef vector<ll> vec;
  template<size_t size> using arr = array<ll, size>;
  const static void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0); }
  const static ll lowbit(const ll x) { return x & -x; }
  mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());
  const ll DIM = 5e5;
  const ll MOD = 1e9 + 7;
  const ll INF = 1e10;
  const lf EPS = 1e-8;
  ll freq[DIM];
  arr<3> operator*(const arr<3>& a, const arr<3>& b) { return { a[0] * b[0], a[1] * b[1], a[2] * b[2] };}
  arr<3> operator+(const arr<3>& a, const arr<3>& b) { return { a[0] + b[0], a[1] + b[1], a[2] + b[2] }; }
  arr<3> operator-(const arr<3>& a, const arr<3>& b) { return { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }
  arr<3> operator%(const arr<3>& a, const ll b) { return { a[0] % b, a[1] % b, a[2] % b }; }
  arr<3> P[DIM];
  int main() {
  	fast_io();
  	/* El Psy Kongroo */	
  	const arr<3> BASE = { 3,5,7 };
  	P[0] = { 1,1,1 };
  	for (ll i = 1; i < DIM; i++) P[i] = (P[i - 1] * BASE) % MOD;
  	map<arr<3>, II> mp;
  	ll ans = 0;
  	ll n, k; cin >> n >> k;
  	arr<3> hash_v = { 0,0,0 };
  	auto upd_hash_add = [&](ll i) {
  		hash_v = (hash_v + P[i]) % MOD;
  	};
  	auto upd_hash_sub = [&](ll i) {
  		hash_v = (hash_v - P[i] + arr<3>{MOD, MOD, MOD}) % MOD;
  	};
  	mp[hash_v] = -1;
  	for (ll c, nth = 0, n_nonzero = 0; nth < n; nth++) {
  		cin >> c;
  		if (!freq[c]) n_nonzero++;
  		freq[c]++;
  		upd_hash_add(c);
  		if (n_nonzero == k) {
  			for (ll i = 1; i <= k; i++) {
  				if (freq[i] == 1) n_nonzero--;
  				freq[i]--;
  				upd_hash_sub(i);
  			}
  		}
  		// update ranges
  		if (!mp.contains(hash_v)) mp[hash_v] = { nth,nth };
  		mp[hash_v].first = min(mp[hash_v].first, nth);
  		mp[hash_v].second = max(mp[hash_v].second, nth);
  		ans = max(ans, mp[hash_v].second - mp[hash_v].first);
  	}
  	cout << ans << endl;
  	return 0;
  }
  ```
  
  ## [2. G - Cubic?](https://atcoder.jp/contests/abc238/tasks/abc238_g)
  
  TL;DR - 随机化哈希
  
  https://atcoder.jp/contests/abc238
  
  官方题解：https://atcoder.jp/contests/abc238/editorial/3372
  
  ---
  
  (据信)很典的哈希题（但对自己来说算初见了...）；以下简要记录
  
  1. 记$freq_i$即为这里将$A_i$分解质因数后的质因数个数
     - 有效区间$a,b$对应的比对即为$f =\sum_{a}^{b}{freq_i}$有$\forall x \in f,x \equiv0\mod 3$
     - $freq$做前缀和后可有$O(kn\log n)$复杂度，$k$为质因数种数；空间时间复杂度显然不过关
  2. 不同于上一题：由$f$形式知需要维护可差分的前缀区间；这样的hash设计如下：
    - 为质因数$p$取随机整数$H_p \in [0,2]$
    - 记每个数质因数集为$S_i$,计算其hash即为$hash_i = \sum_{}{H_j}, \forall j \in S_i $
    - 显然，**这样的hash有充分性**：$ A_1 \cdot A_2 \cdot ... A_n 的每个质因数个数为3的倍数 \implies hash_n \equiv 0 \mod 3$
    - **但必要性并不存在**，但是已有的充分性（不会有假阴性）可以被利用 ；随机化基础上反复验证（+运气）可以K题
  3. hash做前缀和后由于模数性质仍可以就$\mod 3$直接判断区间OK；详见code
  
  ---
  
  ```c++
  #pragma GCC optimize("O3","unroll-loops","inline")
  #include "bits/stdc++.h"
  using namespace std;
  #define PRED(T,X) [](T const& lhs, T const& rhs) {return X;}
  typedef long long ll; typedef unsigned long long ull; typedef double lf; typedef long double llf;
  typedef __int128 i128; typedef unsigned __int128 ui128;
  typedef pair<ll, ll> II; typedef vector<ll> vec;
  template<size_t size> using arr = array<ll, size>;
  const static void fast_io() { ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0); }
  const static ll lowbit(const ll x) { return x & -x; }
  mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());
  const ll DIM = 1e6 + 10;
  const ll MOD = 1e9 + 7;
  const ll INF = 2e18;
  const lf EPS = 1e-8;
  namespace euler_sieve {
  	vector<vec> primes;
  	void Prime_Factor_Offline(ll MAX) {
  		primes.resize(MAX);
  		for (ll i = 2; i < MAX; i++) {
  			if (!primes[i].empty()) continue;
  			for (ll j = i; j < MAX; j += i) {
  				ll mj = j;
  				while (mj % i == 0) {
  					primes[j].push_back(i);
  					mj /= i;
  				}
  			}
  		}
  	}
  
  	void Prime_Factor(ll x, vec& res) {
  		for (ll i = 2; i * i <= x; i++) while (x % i == 0) res.push_back(i), x /= i;
  		if (x != 1) res.push_back(x);
  	}
  }
  ll A[DIM], Hp[DIM], H[DIM], L[DIM], R[DIM], ans[DIM];
  int main() {
  	fast_io();
  	/* El Psy Kongroo..! */
  	uniform_int_distribution dis(0, 2);
  	euler_sieve::Prime_Factor_Offline(DIM);
  	ll n, q; cin >> n >> q;	
  	fill(ans + 1, ans + q + 1, true);
  	for (ll i = 1; i <= n; i++) cin >> A[i];
  	for (ll i = 1; i <= q; i++) cin >> L[i] >> R[i];
  	for (ll B = 0; B < 100; B++) {
  		fill(H + 1, H + n + 1, 0);
  		for (ll i = 1; i <= DIM; i++) Hp[i] = dis(RNG);
  		for (ll i = 1; i <= n; i++)
  			for (ll p : euler_sieve::primes[A[i]])
  				H[i] += Hp[p];
  		for (ll i = 1; i <= n; i++) H[i] += H[i - 1];
  		for (ll i = 1; i <= q; i++)
  			if ((H[R[i]] - H[L[i] - 1]) % 3) 
  				ans[i] = false;
  	}
  	for (ll i = 1; i <= q; i++) {
  		if (ans[i]) cout << "YES\n";
  		else cout << "NO\n";
  	}
  	return 0;
  }
  ```

# 线段树

**注:** `segment_tree` 均采用 `1-Index` 访问； `segment_tree::reset(vector&)` 中`vector`为`0-Index`

## 区间延迟（Lazy）修改模版

- C++ 风格实现

```c++
template<typename T> struct segment_tree {
    struct node {
        ll l, r; // 区间[l,r]
        T sum_v;
        T max_v;
        // lazy值
        T lazy_add;
        optional<T> lazy_set;
        ll length() const { return r - l + 1; }
        ll mid() const { return (l + r) / 2; }
    };
    vector<node> tree;
private:
    ll begin = 1, end = 1;
    void push_up(ll o) {
        // 向上传递
        ll lc = o * 2, rc = o * 2 + 1;
        tree[o].sum_v = tree[lc].sum_v + tree[rc].sum_v;
        tree[o].max_v = max(tree[lc].max_v, tree[rc].max_v);
    }
    void push_down(ll o) {
        // 向下传递
        ll lc = o * 2, rc = o * 2 + 1;
        if (tree[o].lazy_set.has_value()) {
            tree[lc].lazy_add = tree[rc].lazy_add = 0;
            tree[lc].lazy_set = tree[rc].lazy_set = tree[o].lazy_set;
            // 可差分操作
            tree[lc].max_v = tree[o].lazy_set.value();
            tree[rc].max_v = tree[o].lazy_set.value();
            // 求和贡献与长度有关
            tree[lc].sum_v = tree[o].lazy_set.value() * tree[lc].length();
            tree[rc].sum_v = tree[o].lazy_set.value() * tree[rc].length();
            tree[o].lazy_set.reset();
        }
        if (tree[o].lazy_add) {
            tree[lc].lazy_add += tree[o].lazy_add, tree[rc].lazy_add += tree[o].lazy_add;
            // 同上
            tree[lc].max_v += tree[o].lazy_add;
            tree[rc].max_v += tree[o].lazy_add;
            tree[lc].sum_v += tree[o].lazy_add * tree[lc].length();
            tree[rc].sum_v += tree[o].lazy_add * tree[rc].length();
            tree[o].lazy_add = {};
        }
    }
    void update(ll o, ll l, ll r, optional<T> const& set_v = {}, T const& add_v = 0) {
        ll lc = o * 2, rc = o * 2 + 1;
        if (tree[o].l == l && tree[o].r == r) { // 定位到所在区间 - 同下
            if (set_v.has_value()) {
                // set
                tree[o].max_v = set_v.value();
                tree[o].sum_v = set_v.value() * tree[o].length();
                tree[o].lazy_set = set_v; tree[o].lazy_add = {};
            }
            else {
                // add
                tree[o].max_v += add_v;
                tree[o].sum_v += add_v * tree[o].length();
                tree[o].lazy_add += add_v;
            }
            return;
        }
        push_down(o);
        ll mid = tree[o].mid();
        if (r <= mid) update(lc, l, r, set_v, add_v);
        else if (mid < l) update(rc, l, r, set_v, add_v);
        else {
            update(lc, l, mid, set_v, add_v);
            update(rc, mid + 1, r, set_v, add_v);
        }
        push_up(o);
    }
    node query(ll o, ll l, ll r) {
        ll lc = o * 2, rc = o * 2 + 1;
        if (tree[o].l == l && tree[o].r == r) return tree[o];
        push_down(o);
        ll mid = tree[o].mid();
        if (r <= mid) return query(lc, l, r);
        else if (mid < l) return query(rc, l, r);
        else {
            node p = query(lc, l, mid);
            node q = query(rc, mid + 1, r);
            return {
                l, r,
                p.sum_v + q.sum_v,
                max(p.max_v, q.max_v),
            };
        }
    }
    void build(ll o, ll l, ll r, const T* src = nullptr) {
        ll lc = o * 2, rc = o * 2 + 1;
        tree[o] = {};
        tree[o].l = l, tree[o].r = r;
        if (l == r) {
            if (src) tree[o].sum_v = tree[o].max_v = src[l];
            return;
        }
        ll mid = (l + r) / 2;
        build(lc, l, mid, src);
        build(rc, mid + 1, r, src);
        push_up(o);
    }
    void build(const T* src = nullptr) { build(begin, begin, end, src); }
public:
    void range_add(ll l, ll r, T const& v) { update(begin, l, r, {}, v); }
    void range_set(ll l, ll r, T const& v) { update(begin, l, r, v, 0); }
    node range_query(ll l, ll r) { return query(begin, l, r); }
    T range_sum(ll l, ll r) { return range_query(l, r).sum_v; }
    T range_max(ll l, ll r) { return range_query(l, r).max_v; }
    void reserve(const ll n) { tree.reserve(n); }
    void reset(const ll n) { end = n; tree.resize(end << 2); build(); }
    // 注意：src[0]会被省略
    void reset(const vector<T>& src) {
        end = src.size() - 1; tree.resize(end << 2);
        build(src.data());
    }
    explicit segment_tree() {};
    explicit segment_tree(const ll n) : begin(1), end(n) { reset(n); }
};
```

- https://codeforces.com/contest/2014/submission/282795544 （D，区间改+单点查询和）
- https://codeforces.com/contest/339/submission/282875335 （D，单点改+区间查询）

## 可持久化线段树（主席树）

- https://zhuanlan.zhihu.com/p/762284607
- https://ac.nowcoder.com/acm/contest/91177/F （找第$k$小）
- https://www.luogu.com.cn/problem/P3834

```c++
template <typename T> struct segment_tree {
    constexpr static ll root = 1; // 根节点编号
    ll node_id = 1; // 当前最新节点编号
public:
    struct node {
        ll lc, rc; // 左右子节点**编号**；非区间
        ll l, r; // 区间
        T sum{};
    };
    vector<node> tree;
    // 向上传递
    void push_up(ll o) {
        tree[o].sum = tree[tree[o].lc].sum + tree[tree[o].rc].sum;
    }
    // 初始版本
    void build(ll o, ll l, ll r) {
        if (l == r) return;
        ll mid = (l + r) / 2;
        ll lc = tree[o].lc = ++node_id, rc = tree[o].rc = ++node_id;
        tree[o].l = l, tree[o].r = r;
        build(lc, l, mid);
        build(rc, mid + 1, r);
        push_up(o);
    }
    void update(ll pos, ll l, ll r, ll prev /*旧版本复制源点*/, ll curr /*新版本新建点*/, T v) {
        ll mid = (l + r) / 2;
        if (l == r) {
            // 到达叶子点
            // 修改只在新点及剪出来的枝上体现
            tree[curr].sum = tree[prev].sum + v;
        } else {
            // 到叶子点路上；默认复用
            tree[curr] = tree[prev];
            if (pos <= mid) {
                // 新点会在左子树开，途径有必要持久化（复制）
                // 每个点都要开新点
                tree[curr].lc = ++node_id;
                update(pos, l, mid, tree[prev].lc, tree[curr].lc, v);
            } else {
                // 右子树 - 同上，交换左右
                tree[curr].rc = ++node_id;
                update(pos, mid + 1, r, tree[prev].rc, tree[curr].rc, v);
            }
            push_up(curr);
        }
    }
    explicit segment_tree(ll n) : tree(n) {};
};
segment_tree<ll> seg(DIM);
// 树上二分找[l,r]区间第k小
int query_kth(ll l, ll r, ll prev /*旧版本同位置点*/, ll curr /*新版本同位置点*/, ll kth_small) {
    if (l == r) return l;
    ll mid = (l + r) / 2;
    // 我们的每一个版本（根节点上点）线段树存的为*权值*（或直方图的高度，即数字的数目）
    // 找第k小即为找*离散化后*数x对应 \sum_{i=1}^{x} tree[i].sum < kth_small 的上限
    // 在[l,r]区间内找，可以看成是*两个*版本树的差分
    // 区间内的数目即为：
    ll d = seg.tree[seg.tree[curr].lc].sum - seg.tree[seg.tree[prev].lc].sum;
    // 树上二分
    if (d < kth_small) {
        // x更大在右子树
        // 在右边找；注意左区间数*不能*统计
        return query_kth(mid + 1, r, seg.tree[prev].rc, seg.tree[curr].rc, kth_small - d);
    } else {
        // x更小在左子树
        return query_kth(l, mid, seg.tree[prev].lc, seg.tree[curr].lc, kth_small);
    }
}
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll n, q; cin >> n >> q;
    vec a(n + 1), mp;
    for (ll i = 1; i<=n;i++)
        cin >> a[i], mp.push_back(a[i]);
    sort(mp.begin(), mp.end());
    mp.erase(unique(mp.begin(), mp.end()), mp.end());

    ll m = mp.size(); // 离散化后位置i对应数字
    seg.build(seg.root, 1, m);
    vec roots(n+1, seg.root);
    for (ll i = 1; i<=n;i++) {
        // 新版本
        roots[i] = ++seg.node_id;
        // 从上一个版本转移；这里在mp[i]上多一个数
        ll pos = lower_bound(mp.begin(), mp.end(), a[i]) - mp.begin() + 1;
        seg.update(pos, 1, m, roots[i - 1], roots[i], 1);
    }
    while (q--) {
        ll l,r; cin >> l >> r;
        ll mid = (r - l + 2) / 2; // \ceil
        // 注意我们求的是*上限*
        ll pos = query_kth(1, m, roots[l - 1], roots[r], mid);
        cout << mp[pos - 1] << endl;
    }
    return 0;
}
```

## 242E. XOR on Segment

区间二进制改+lazy传递+二进制trick

> You've got an array $a$, consisting of $n$ integers $a_1, a_2, ..., a_n$. You are allowed to perform two operations on this array:
>
> 1. Calculate the sum of current array elements on the segment $[l,r]$, that is, count value $a_l + a_{l+1} + ... + a_{r}$
>
> 2. Apply the xor operation with a given number *x* to each array element on the segment $[l,r]$, that is, execute $a_l = a_l \oplus x, a_{l+1} = a_{l+1} \oplus x,...,a_r = a_r \oplus x$ This operation changes exactly $r - l + 1$ array elements.
>
> Expression $x \oplus y$ means applying bitwise xor operation to numbers *x* and *y*. The given operation exists in all modern programming languages, for example in language *C++* and *Java* it is marked as "^", in *Pascal* — as "xor".
> You've got a list of *m* operations of the indicated type. Your task is to perform all given operations, for each sum query you should print the result you get.

```c++
template<typename T> struct segment_tree {
	struct node {
		ll l, r; // 区间[l,r]        
		T sum;
		// lazy值
		bool lazy_set; // xor项
		ll length() const { return r - l + 1; }
		ll mid() const { return (l + r) / 2; }
	};
	vector<node> tree;
private:
	ll begin = 1, end = 1;
	void flip(node& n) { n.sum = n.length() - n.sum, n.lazy_set ^= 1; }
	void push_up(ll o) {
		// 向上传递
		ll lc = o * 2, rc = o * 2 + 1;
		tree[o].sum = tree[lc].sum + tree[rc].sum;
	}
	void push_down(ll o) {
		// 向下传递
		ll lc = o * 2, rc = o * 2 + 1;
		if (tree[o].lazy_set) {			
			flip(tree[lc]), flip(tree[rc]);
			tree[o].lazy_set = false;
		}
	}
	void update(ll o, ll l, ll r) {
		ll lc = o * 2, rc = o * 2 + 1;
		if (!tree[o].l) return;
		if (tree[o].l == l && tree[o].r == r) { // 定位到所在区间 - 同下
			// set				
			flip(tree[o]);
			return;
		}
		push_down(o);
		ll mid = tree[o].mid();
		if (r <= mid) update(lc, l, r);
		else if (mid < l) update(rc, l, r);
		else {
			update(lc, l, mid);
			update(rc, mid + 1, r);
		}
		push_up(o);
	}
	node query(ll o, ll l, ll r) {
		ll lc = o * 2, rc = o * 2 + 1;
		if (!tree[o].l) return {};
		if (tree[o].l == l && tree[o].r == r) return tree[o];
		push_down(o);
		ll mid = tree[o].mid();
		if (r <= mid) return query(lc, l, r);
		else if (mid < l) return query(rc, l, r);
		else {
			node p = query(lc, l, mid);
			node q = query(rc, mid + 1, r);
			return {
				l, r,
				p.sum + q.sum
			};
		}
	}
	void build(ll o, ll l, ll r, const T* src = nullptr, ll depth = 1) {
		ll lc = o * 2, rc = o * 2 + 1;
		tree[o] = {};
		tree[o].l = l, tree[o].r = r;
		if (l == r) {
			if (src) tree[o].sum = src[l];
			return;
		}
		ll mid = (l + r) / 2;
		build(lc, l, mid, src, depth + 1);
		build(rc, mid + 1, r, src, depth + 1);
		push_up(o);
	}
	void build(const T* src = nullptr) { build(begin, begin, end, src); }
public:
	void range_set(ll l, ll r) { update(begin, l, r); }
	node range_query(ll l, ll r) { return query(begin, l, r); }
	void reserve(const ll n) { tree.reserve(n); }
	void reset(const ll n) { end = n; tree.resize(end << 2); build(); }
	void reset(const vector<T>& src) {
		end = src.size(); tree.resize(end << 2);
		build(src.data() - 1);
	}
	explicit segment_tree() {};
	explicit segment_tree(const ll n) : begin(1) { reset(n); }

	void debug() {
		ll d = 1;
		for (auto& n : tree) {
			if (n.depth == 0) continue;
			if (n.depth != d) d = n.depth, cout << endl;
			n.print();
		}
		cout << endl;
	}
};


int main() {
	fast_io();
	/* El Psy Kongroo */
	segment_tree<unsigned int> s[20];
	ll n; cin >> n;
	vector<unsigned int> arr(n); for (auto& x : arr) cin >> x;
	vector<unsigned int> bits(n);
	for (ll i = 0; i < 20; ++i) {
		for (ll j = 0; j < n; j++) bits[j] = (arr[j] & (1ll << i)) != 0;
		s[i].reset(bits);
	}
	ll m; cin >> m;
	while (m--) {
		ll op; cin >> op;
		switch (op)
		{
		case 1:
		{
			// sum
			ll l, r, ans = 0; cin >> l >> r;
			for (ll i = 0; i < 20; ++i) {
				ans += s[i].range_query(l, r).sum * (1ll << i);
			}
			cout << ans << endl;
			break;
		}
		case 2:
		{
			// xor
			ll l, r, x; cin >> l >> r >> x;
			for (ll i = 0; i < 20; ++i) {
				if (x & (1ll << i)) s[i].range_set(l, r); // mark as flip
			}
			break;
		}
		default:
			break;
		}
	}
	return 0;
}
```

## 920F. SUM and REPLACE

数论、单点改+剪枝

>Let $D(x)$ be the number of positive divisors of a positive integer $x$. For example, $D(2)= 2$ (2 is divisible by 1 and 2), $D(6) = 4$ (6 is divisible by 1, 2, 3 and 6).
>You are given an array $a$ of $n$ integers. You have to process two types of queries:
>
>1. `REPLACE` $l,r$ - for every $i \in [l,r]$, replace $a_i$ with $D(a_i)$
>2. `SUM` $l,r$ - calculate $\sum_{i=l}^{r}{a_i}$
>     Print the answer for each `SUM` query.

```c++
namespace eratosthenes_sieve_d {...}; // 见 板子整理
using namespace eratosthenes_sieve_d;
template<typename T> struct segment_tree {
    struct node {
        ll l, r; // 区间[l,r]
        T sum_v;
        T max_v;
        ll length() const { return r - l + 1; }
        ll mid() const { return (l + r) / 2; }
    };
    vector<node> tree;
private:
    ll begin = 1, end = 1;
    void push_up(ll o) {
        // 向上传递
        ll lc = o * 2, rc = o * 2 + 1;
        tree[o].sum_v = tree[lc].sum_v + tree[rc].sum_v;
        tree[o].max_v = max(tree[lc].max_v, tree[rc].max_v);
    }
    void update(ll o, ll l, ll r) {
        ll lc = o * 2, rc = o * 2 + 1;
        if (tree[o].max_v <= 2) return; // 剪掉！！
        if (tree[o].length() == 1 && tree[o].l == l && tree[o].r == r) {
            tree[o].sum_v = tree[o].max_v = D[tree[o].sum_v];
            return;
        }
        ll mid = tree[o].mid();
        if (r <= mid) update(lc, l, r);
        else if (mid < l) update(rc, l, r);
        else {
            update(lc, l, mid);
            update(rc, mid + 1, r);
        }
        push_up(o);
    }
    node query(ll o, ll l, ll r) {
        ll lc = o * 2, rc = o * 2 + 1;
        if (tree[o].l == l && tree[o].r == r) return tree[o];
        ll mid = tree[o].mid();
        if (r <= mid) return query(lc, l, r);
        else if (mid < l) return query(rc, l, r);
        else {
            node p = query(lc, l, mid);
            node q = query(rc, mid + 1, r);
            return {
                l, r,
                p.sum_v + q.sum_v,
                max(p.max_v, q.max_v),
            };
        }
    }
    void build(ll o, ll l, ll r, const T* src = nullptr) {
        ll lc = o * 2, rc = o * 2 + 1;
        tree[o] = {};
        tree[o].l = l, tree[o].r = r;
        if (l == r) {
            if (src) tree[o].sum_v = tree[o].max_v = src[l];
            return;
        }
        ll mid = tree[o].mid();
        build(lc, l, mid, src);
        build(rc, mid + 1, r, src);
        push_up(o);
    }
    void build(const T* src = nullptr) { build(begin, begin, end, src); }
public:
    void range_set(ll l, ll r) { update(begin, l, r); }
    node range_query(ll l, ll r) { return query(begin, l, r); }
    T range_sum(ll l, ll r) { return range_query(l, r).sum_v; }
    T range_max(ll l, ll r) { return range_query(l, r).max_v; }
    void reserve(const ll n) { tree.reserve(n); }
    void reset(const ll n) { end = n; tree.resize(end << 2); build(); }
    void reset(const vector<T>& src) {
        end = src.size(); tree.resize(end << 2);
        build(src.data() - 1);
    }
    explicit segment_tree() {};
    explicit segment_tree(const ll n) : begin(1), end(n) { reset(n); }
};

int main() {
    fast_io();
    /* El Psy Kongroo */
    init();
    // 1 -> 1, 2 -> 2 无需修改
    // 2 以上的区间趋近D[n]可以非常快 - log n次内可以暴力解决
    ll n, m; cin >> n >> m; vec arr(n);
    for (ll& x : arr) cin >> x;
    segment_tree<ll> st; st.reset(arr);
    while (m--) {
        ll op; cin >> op;
        switch (op)
        {
        case 1: {
            // REPLACE
            ll l, r; cin >> l >> r;
            ll mx = st.range_max(l, r);
            if (mx > 2) 
                st.range_set(l, r);
            break;
        }
        case 2: {
            // SUM
            ll l, r; cin >> l >> r;
            cout << st.range_sum(l, r) << endl;
            break;
        }
        default:
            break;
        }
    }
    return 0;
}
```

## 1234D. Distinct Characters Queries

串转换`bitset`求独特值数

>You are given a string $s$ consisting of lowercase Latin letters and $q$ queries for this string.
>Recall that the substring $s[l; r]$ of the string $s$ is the string $s_l s_{l + 1} \dots s_r$. For example, the substrings of "codeforces" are "code", "force", "f", "for", but not "coder" and "top".
>There are two types of queries:
>
>-   $1~ pos~ c$ ($1 \le pos \le |s|$, $c$ is lowercase Latin letter): replace $s_{pos}$ with $c$ (set $s_{pos} := c$);
>-   $2~ l~ r$ ($1 \le l \le r \le |s|$): calculate the number of distinct characters in the substring $s[l; r]$.

```c++
template<typename T> struct segment_tree {
	struct node {
		ll l, r; // 区间[l,r]        
		T value; // a-z 标记
		// lazy值        
		optional<T> lazy_set;
		ll length() const { return r - l + 1; }
		ll mid() const { return (l + r) / 2; }
	};
	vector<node> tree;
private:
	ll begin = 1, end = 1;
	void push_up(ll o) {
		// 向上传递
		ll lc = o * 2, rc = o * 2 + 1;
		tree[o].value = tree[lc].value | tree[rc].value;
	}
	void push_down(ll o) {
		// 向下传递
		ll lc = o * 2, rc = o * 2 + 1;
		if (tree[o].lazy_set.has_value()) {
			tree[lc].lazy_set = tree[rc].lazy_set = tree[o].lazy_set;
			// 可差分操作            
			tree[lc].value = tree[rc].value = tree[o].lazy_set.value();
			tree[o].lazy_set.reset();
		}
	}
	void update(ll o, ll l, ll r, optional<T> const& set_v = {}, T const& add_v = 0) {
		ll lc = o * 2, rc = o * 2 + 1;
		if (tree[o].l == l && tree[o].r == r) { // 定位到所在区间 - 同下
			if (set_v.has_value()) {
				// set
				tree[o].value = set_v.value();
				tree[o].lazy_set = set_v;
			}
			return;
		}
		push_down(o); // 单点其实没必要...
		ll mid = tree[o].mid();
		if (r <= mid) update(lc, l, r, set_v, add_v);
		else if (mid < l) update(rc, l, r, set_v, add_v);
		else {
			update(lc, l, mid, set_v, add_v);
			update(rc, mid + 1, r, set_v, add_v);
		}
		push_up(o);
	}
	node query(ll o, ll l, ll r) {
		ll lc = o * 2, rc = o * 2 + 1;
		if (tree[o].l == l && tree[o].r == r) return tree[o];
		push_down(o);
		ll mid = tree[o].mid();
		if (r <= mid) return query(lc, l, r);
		else if (mid < l) return query(rc, l, r);
		else {
			node p = query(lc, l, mid);
			node q = query(rc, mid + 1, r);
			return { l, r, p.value | q.value };
		}
	}
	void build(ll o, ll l, ll r, const T* src = nullptr) {
		ll lc = o * 2, rc = o * 2 + 1;
		tree[o] = {};
		tree[o].l = l, tree[o].r = r;
		if (l == r) {
			if (src) tree[o].value = src[l];
			return;
		}
		ll mid = (l + r) / 2;
		build(lc, l, mid, src);
		build(rc, mid + 1, r, src);
		push_up(o);
	}
	void build(const T* src = nullptr) { build(begin, begin, end, src); }
public:
	void range_set(ll l, ll r, T const& v) { update(begin, l, r, v, 0); }
	node range_query(ll l, ll r) { return query(begin, l, r); }
	/****/
	void reserve(const ll n) { tree.reserve(n); }
	void reset(const ll n) { end = n; tree.resize(end << 2); build(); }
	// src: 0-based input array
	void reset(const vector<T>& src) {
		end = src.size(); tree.resize(end << 2);
		build(src.data() - 1);
	}
	explicit segment_tree() {};
	explicit segment_tree(const ll n) : begin(1) { reset(n); }
};
typedef bitset<32> bs;
bs from_char(char c) { return bs(1 << (c - 'a')); }
int main() {
	fast_io();
	/* El Psy Kongroo */
	string s; cin >> s;
	vector<bs> arr(s.size());
	for (ll i = 0; i < s.size(); ++i) arr[i] = from_char(s[i]);
	segment_tree<bs> st; st.reset(arr);
	ll q; cin >> q;
	while (q--) {
		ll op; cin >> op;
		if (op == 1) {
			ll pos; char c; cin >> pos >> c;
			st.range_set(pos, pos, from_char(c));
		}
		else {
			ll l, r; cin >> l >> r;
			auto ans = st.range_query(l, r);
			auto bits = ans.value;
			cout << bits.count() << '\n';
		}
	}
	return 0;
}
```

## P11373 「CZOI-R2」天平

- 正解转 https://mos9527.com/posts/cp/gcd-problems/#p11373-czoi-r2%E5%A4%A9%E5%B9%B3，此处为Subtask 3解法
- TL；DR 区间维护$gcd$；同时将**区间改**操作化为**单点改**操作省去`push_down`
  - 给定数组$a$定义$gcd(a) = gcd(a_1,a_2,...a_n)$
  - 由[引理](https://mos9527.com/posts/cp/gcd-problems/#691c-row-gcd)知$gcd(x,y) = gcd(x,y-x)$，可拓展为$gcd(a) = gcd(a_1, a_2 - a_1, ..., a_n - a_{n-1})$
  - 记差分数组为$b$,$\forall b_i \in b, b_i = a_i - a_{i-1}$,既有$gcd(a) = gcd(a_1, b_2,...,b_n)$
  - 鉴于题目只要求**区间加**，即等效于**差分数组单点改**，维护$b$数组RMQ后`push_up`即可

```c++
template<typename T> struct segment_tree {
  struct node {
    ll l, r; // 区间[l,r]
    T sum, gcd; // 差分和，差分gcd
    ll length() const { return r - l + 1; }
    ll mid() const { return (l + r) / 2; }
  };
  vector<node> tree;
private:
  ll begin = 1, end = 1;
  void push_up(ll o) {
    // 向上传递
    ll lc = o * 2, rc = o * 2 + 1;
      tree[o].sum = tree[lc].sum + tree[rc].sum;
    tree[o].gcd = gcd(tree[lc].gcd, tree[rc].gcd);
  }
  void update(ll o, ll l, ll r, ll v) {
    ll lc = o * 2, rc = o * 2 + 1;
    if (tree[o].l == l && tree[o].r == r && tree[o].length() == 1) { // 定位单点
      tree[o].sum += v, tree[o].gcd = tree[o].sum;
      return;
    }
    ll mid = tree[o].mid();
    if (r <= mid) update(lc, l, r, v);
    else if (mid < l) update(rc, l, r, v);
    else {
      update(lc, l, mid, v);
      update(rc, mid + 1, r, v);
    }
    push_up(o);
  }
  node query(ll o, ll l, ll r) {
    ll lc = o * 2, rc = o * 2 + 1;
    if (tree[o].l == l && tree[o].r == r) return tree[o];
    ll mid = tree[o].mid();
    if (r <= mid) return query(lc, l, r);
    else if (mid < l) return query(rc, l, r);
    else {
      node p = query(lc, l, mid);
      node q = query(rc, mid + 1, r);
      return { l, r, p.sum + q.sum, gcd(p.gcd, q.gcd) };
    }
  }
  void build(ll o, ll l, ll r, const T* src = nullptr) {
    ll lc = o * 2, rc = o * 2 + 1;
    tree[o] = {};
    tree[o].l = l, tree[o].r = r;
    if (l == r) {
      if (src) tree[o].sum = tree[o].gcd = src[l];
      return;
    }
    ll mid = (l + r) / 2;
    build(lc, l, mid, src);
    build(rc, mid + 1, r, src);
    push_up(o);
  }
  void build(const T* src = nullptr) { build(begin, begin, end, src); }
public:
  void add(ll p, T const& v) { update(begin, p,p, v); }
  node range_query(ll l, ll r) { return query(begin, l, r); }
  /****/
  void reserve(const ll n) { tree.reserve(n); }
  void reset(const ll n) { end = n; tree.resize(end << 2); build(); }
  // src: 0-based input array
  void reset(const vector<T>& src) {
    end = src.size(); tree.resize(end << 2);
    build(src.data() - 1);
  }
  explicit segment_tree() {};
  explicit segment_tree(const ll n) : begin(1) { reset(n); }
};
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll n,q; cin >> n >> q;
    vec src(n); for (ll& x : src) cin >> x;
    for (ll i = n - 1;i >= 1;i--) src[i] -= src[i-1];
    segment_tree<ll> seg(n); seg.reset(src);
    while (q--) {
        char op; cin >> op;
        switch (op) {
            case 'D': {
                ll x; cin >> x;
                break;
            }
            case 'I': {
                ll x,y; cin >> x>>y;
                break;
            }
            case 'A': {
                ll l,r,v; cin >> l >> r >> v;
                seg.add(l,v);
                if (r != n) seg.add(r+1,-v);
                break;
            }
            case 'Q':
            default:{
                ll l,r,v; cin >> l >> r >> v;
                ll a = seg.range_query(1,l).sum; // 差分和->a_l
                ll b_gcd = seg.range_query(l + 1,r).gcd;
                ll range_gcd = gcd(a,b_gcd);
                if (v % range_gcd == 0) cout << "YES" << endl;
                else cout << "NO" << endl;
                break;
            }
        }
    }
    return 0;
}
```

  # DP

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

### 南昌 2025 邀请赛 - G Exploration

递推式很简单；记$dp_{i,j}$为从点$i$走$j$步可达最大边权积；实现上由于原图有（自）环DFS/BFS处理这个dp不好写

```c++
vector<vec> dp(n + 1, vec(33, 1));
for (ll i = 1;i <= 32;i++)
    for (ll u = 1; u <= n;u++)
        for (auto [v,d] : G[u])
            dp[u][i] = max(dp[u][i], dp[v][i-1] * d);
```

