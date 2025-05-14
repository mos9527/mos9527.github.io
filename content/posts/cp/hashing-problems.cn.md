---
author: mos9527
lastmod: 2025-05-14T10:08:15.029000+08:00
title: 算竞笔记 - 哈希类型专题
tags: ["哈希","ACM","算竞","XCPC","板子","题集","Codeforces","C++","数学"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

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
