---
author: mos9527
lastmod: 2025-05-09T19:37:44.543488
title: 算竞笔记 - 题集/板子整理（C++）
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---
# Preface

参考主要来自[《算法竞赛入门经典：训练指南》](https://cread.jd.com/read/startRead.action?bookId=30133704&readType=1)、[OIWiki](https://oi-wiki.org/)、[CP Algorithms](https://cp-algorithms.com/)等资源和多方博客、课程，在自己的码风下所著

**注：** 部分实现可能用到较新语言特性，烦请修改后在较老OJ上使用；**原则上提供的代码兼容符合Cpp20及以上标准的编译器**

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
# Misc

- 开启GCC debug容器: `add_compile_definitions(-D_GLIBCXX_DEBUG)`

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
template<typename T> struct vec2 {
    T x, y;
    ///
    inline T length_sq() const { return x * x + y * y; }
    inline T length() const { return sqrt(length_sq()); }
    inline vec2& operator+=(vec2 const& other) { x += other.x, y += other.y; return *this; }
    inline vec2& operator-=(vec2 const& other) { x -= other.x, y -= other.y; return *this; }
    inline vec2& operator*=(T const& other) { x *= other, y *= other; return *this; }
    inline vec2& operator/=(T const& other) { x /= other, y /= other; return *this; }
    inline vec2 operator+(vec2 const& other) const { vec2 v = *this; v += other; return v; }
    inline vec2 operator-(vec2 const& other) const { vec2 v = *this; v -= other; return v; }
    inline vec2 operator*(T const& other) const { vec2 v = *this; v *= other; return v; }
    inline vec2 operator/(T const& other) const { vec2 v = *this; v /= other; return v; }
    ///
    inline static T dist_sq(vec2 const& a, vec2 const& b) {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    }
    inline static T dist(vec2 const& a, vec2 const& b) {
        return sqrt(vec2::dist_sq(a, b));
    }
    inline static T cross(vec2 const& a, vec2 const& b) {
        return a.x * b.y - a.y * b.x;
    }
    inline static T dot(vec2 const& a, vec2 const& b) {
        return a.x * b.x + a.y * b.y;
    }
    ///
    inline friend bool operator< (vec2 const& a, vec2 const& b) {
        if (a.x - b.x < EPS) return true;
        if (a.x - b.x > EPS) return false;
        if (a.y - b.y < EPS) return true;
        return false;
    }
    inline friend ostream& operator<< (ostream& s, const vec2& v) {
        s << '(' << v.x << ',' << v.y << ')'; return s;
    }
    inline friend istream& operator>> (istream& s, vec2& v) {
        s >> v.x >> v.y; return s;
    }
};
typedef vec2<lf> point;
typedef vec2<ll> Ipoint;
```

#### 二维凸包

```c++
struct convex_hull : vector<point> {
    bool is_inside(point const& p) {
        for (ll i = 0; i < size() - 1; i++) {
            point a = (*this)[i], b = (*this)[i + 1];
            point e = b - a, v = p - a;
            // 全在边同一侧
            if (point::cross(e, v) < EPS) return false;
        }
        return true;
    }
    lf min_dis(point const& p) {
        lf dis = 1e100;
        for (ll i = 0; i < size() - 1; i++) {
            point a = (*this)[i], b = (*this)[i + 1];
            point e = b - a, v = p - a;
            // 垂点在边上
            if (point::dot(p - a, b - a) >= 0 && point::dot(p - b, a - b) >= 0)
                dis = min(dis, abs(point::cross(e, v) / e.length()));
            // 垂点在边外 - 退化到到顶点距离min
            else
                dis = min(dis, min((p - a).length(), (p - b).length()));
        }
        return dis;
    }
    void build(vector<point>& p) { // Andrew p368
        sort(p.begin(), p.end());
        resize(p.size());
        ll m = 0;
        for (ll i = 0; i < p.size(); i++) {
            while (m > 1 && point::cross((*this)[m - 1] - (*this)[m - 2], p[i] - (*this)[m - 2]) < EPS) m--;
            (*this)[m++] = p[i];
        }
        ll k = m;
        for (ll i = p.size() - 2; i >= 0; i--) {
            while (m > k && point::cross((*this)[m - 1] - (*this)[m - 2], p[i] - (*this)[m - 2]) < EPS) m--;
            (*this)[m++] = p[i];
        }
        if (p.size() > 1) m--;
        resize(m);
    }
};
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
#define INF 1e10
struct edge { ll to, weight; };
struct vert { ll vtx, dis; };
struct graph {
	vector<vector<edge>> edges;
	vector<bool> vis;
	vector<ll> dis;
	graph(const size_t verts) : edges(verts + 1), vis(verts + 1), dis(verts + 1) {};
	void add_edge(ll u, ll v, ll w = 1) {
		edges[u].emplace_back(edge{ v,w });
	}
	const auto& dijkstra(ll start) {
		fill(dis.begin(), dis.end(), INF);
        fill(vis.begin(), vis.end(), false);
		const auto pp = PRED(vert, lhs.dis > rhs.dis);
		priority_queue<vert, vector<vert>, decltype(pp)> T{ pp }; // 最短路点
		T.push(vert{ start, 0 });
		dis[start] = 0;
		while (!T.empty())
		{
			vert from = T.top(); T.pop();
			if (!vis[from.vtx]) {
				vis[from.vtx] = true;
				for (auto e : edges[from.vtx]) { // 松弛出边
					if (dis[e.to] > dis[from.vtx] + e.weight) {
						dis[e.to] = dis[from.vtx] + e.weight;
						T.push(vert{ e.to, dis[e.to] });
					}
				}
			}
		}
		return dis;
	}
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

### Hierholzer

```c++
struct edge { ll to, weight; };
struct vert { ll vtx, dis; };
template<size_t Size> struct graph {
	bool G[Size][Size]{};
	ll in[Size]{};

	ll n;
	graph(const size_t verts) : n(verts) {};
	void add_edge(ll u, ll v) {
		G[u][v] = G[v][u] = true;
		in[v]++;
		in[u]++;
	}

	v euler_road_ans;
	v& euler_road(ll pa) {
		euler_road_ans.clear();
		ll odds = 0;
		for (ll i = 1; i <= n; i++) {
			if (in[i] % 2 != 0) 
				odds++;
		}
		if (odds != 0 && odds != 2) return euler_road_ans;
		const auto hierholzer = [&](ll x, auto& func) -> void {
			for (ll i = 1; i <= n; i++) {
				if (G[x][i]) {
					G[x][i] = G[i][x] = 0;
					func(i, func);
				}
			}
			euler_road_ans.push_back(x);
		};
		hierholzer(pa, hierholzer);
        reverse(euler_road_ans.begin(),euler_road_ans.end()
		return euler_road_ans;
	}
};
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

## Dinic 最大流

```c++
struct graph {
    ll n, cnt = 0;
    vec V, W, Next, Head;
    graph(ll n, ll e = DIM) : V(e), W(e), Next(e, -1), Head(e, -1), n(n) {}
    void add_edge(ll u, ll v, ll w) {
        Next[cnt] = Head[u];
        V[cnt] = v, W[cnt] = w;
        Head[u] = cnt;
        cnt++;
    }
    void dinic_add_edge(ll u, ll v, ll w) {
        add_edge(u, v, w); // W[i]
        add_edge(v, u, 0); // W[i^1]
    }
private:
    vec dinic_depth, dinic_cur;
    bool dinic_bfs(ll s, ll t) /* 源点，汇点 */ {
        queue<ll> Q;
        dinic_depth.assign(n + 1, 0);        
        dinic_depth[s] = 1; Q.push(s);
        while (!Q.empty()){
            ll u = Q.front(); Q.pop();
            for (ll i = Head[u]; i != -1; i = Next[i]) {
                if (W[i] && dinic_depth[V[i]] == 0) {                    
                    dinic_depth[V[i]] = dinic_depth[u] + 1;
                    Q.push(V[i]);
                }
            }
        }
        return dinic_depth[t];
    }
    ll dinic_dfs(ll u, ll t, ll flow = INF) {
        if (u == t) return flow;
        for (ll& i = dinic_cur[u] /* 维护掉已经走过的弧 */; i != -1; i = Next[i]) {
            if (W[i] && dinic_depth[V[i]] == dinic_depth[u] + 1) {
                ll d = dinic_dfs(V[i], t, min(flow, W[i]));
                W[i] -= d, W[i^1] += d; // i^1 是 i 的反向边; 原边i%2==0, 反边在之后；故反边^1->原边 反之亦然
                if (d) return d;
            }
        }
        return 0;
    }
public:
    ll dinic(ll s, ll t) {
        ll ans = 0;
        while (dinic_bfs(s, t)) {
            dinic_cur = Head;
            while (ll d = dinic_dfs(s, t)) ans += d;
        }
        return ans;
    }
};
```

- https://www.cnblogs.com/SYCstudio/p/7260613.html

- https://codeforces.com/gym/105336/submission/280592598 (G. 疯狂星期六)

  ```c++
  int main() {
      fast_io();
      /* El Psy Kongroo */
      ll n, m; cin >> n >> m;
      ll t = n + m + 1;
      vec A(n + 1), V(n + 1);
      for (ll i = 1; i <= n; i++) cin >> A[i] >> V[i];
      graph G(n + m + 1);
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

 ## 强连通分量 / SCC

### Tarjan
```c++
struct SCC {
    ll n, dfn_cnt;
    vec dfn, low, vis, sta, dis;
    vec in_loop;
    stack<ll> stk;
    vector<vector<II>> G;
    SCC(ll n) : n(n), sta(n), dfn(n), low(n), G(n), dis(n), dfn_cnt(0), in_loop(n) {};

    void add_edge(ll u, ll v, ll w) { G[u].push_back({v, w}); }

    map<ll, vec> scc;
    void tarjan(ll u) {
        if (dfn[u]) return;
        dfn[u] = low[u] = ++dfn_cnt, stk.push(u), sta[u] = 1;
        for (auto const& [v, w] : G[u]) {
            if (!dfn[v])
                dis[v] = dis[u] + w, tarjan(v), low[u] = min(low[u], low[v]);
            else if (sta[v]) {
                low[u] = min(low[u], dfn[v]);
                // coming from another node. we have a loop in v's component
				// note that components of only one node / zero weight are not considered here
				if (dis[v] != dis[u] + w) in_loop[v] = 1;
            }
			in_loop[u] |= in_loop[v];
        }
        if (dfn[u] == low[u]) {            
            ll v; do {
                v = stk.top(), stk.pop();
                sta[v] = 0;
                scc[u].push_back(v);
                in_loop[v] |= in_loop[u];
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

# 动态规划 / DP

移步 [DP 类型专题](https://mos9527.github.io/posts/cp/dp-problems/)

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

### 线段树

移步 [线段树专题](https://mos9527.github.io/posts/cp/segment-tree-problems/)

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
const ll lowbit(ll x) { return x & (-x); }
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

## 离散化杂谈

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


## MSVC也要用万能头!!

- `bits/stdc++.h`
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
