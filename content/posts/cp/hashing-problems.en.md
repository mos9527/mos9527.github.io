---
author: mos9527
lastmod: 2025-05-19T20:32:37.762536
title: Arithmetic Notes - Hash Types Topic
tags: ["Hashing","ACM","Competeive Programming","XCPC","(Code) Templates","Problem sets","Codeforces","C++","Mathematics"]
categories: ["Problem Solutions", "Competeive Programming", "Collection/compilation"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

## [1. Candy Rush](https://codeforces.com/gym/104736/problem/C)

TL;DR - string hash accelerated comparison + big hash trick

[2023-2024 ACM-ICPC Latin American Regional Programming Contest](https://codeforces.com/gym/104736)

Official solution (a math. puzzle)：https://codeforces.com/gym/104736/attachments/download/22730/editorial.pdf

---

1. After taking the frequency of occurrence of $C_i$ as a vector and doing a prefix sum, we can find that the frequency difference between valid intervals is $k \cdot 1_n$; the leveling complexity of the direct comparison is $O(nk\log{n})$, with $O(k)$ coming from the comparison itself

2. And the comparison can be converted to the $O(1)$ form, trick as follows：

- If $a,b$ corresponds to a valid interval, there must be: $freq_b = k \cdot 1_n + freq_a$
- And in doing the prefix sum, one can eliminate the $k \cdot 1_n$ term in time: i.e., all $-1$ for all non-$0$
- At this point, the comparison becomes $ hash(freq_b) = hash(freq_a) $
- Making a $hash$ that can be updated $O(1)$ allows the final complexity to be a good $O(nlogn)$

The solution $hash$ is $(base^i \cdot c_i)\%MOD$ - to avoid collision you can use larger int type or modulus, or the same code to open an `array` in a different base and then compare it.

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

TL;DR - randomized hash

https://atcoder.jp/contests/abc238

Official solution (a math. puzzle)：https://atcoder.jp/contests/abc238/editorial/3372

---

(Believed to be) very canonical hash question (but counts as a first look for myself...) The following is a brief summary

1. Noting that $freq_i$ is the number of prime factors after decomposing $A_i$ into prime factors here
   - The valid interval $a,b$ corresponds to a comparison that is $f =\sum_{a}^{b}{freq_i}$ with $\forall x \in f,x \equiv0\mod 3$
   - $freq$ can have $O(kn\log n)$ complexity after doing prefix sums, with $k$ being the number of prime factor varieties; space time complexity is clearly not good enough
2. Unlike the previous question: the need to maintain differentiable prefix intervals is known from the form $f$; such a hash is designed as follows：
  - is the prime factor $p$ taken as a random integer $H_p \in [0,2]$
  - Denote the set of prime factors of each number as $S_i$, and compute its hash as $hash_i = \sum_{}{H_j}, \forall j \in S_i $
  - Clearly, **such a hash has sufficiency**: $ A_1 \cdot A_2 \cdot ... {number\ of\ prime\ factors\ of\ each\ of\ A_n\ is\ a\ multiple\ of\ 3} \implies hash_n \equiv 0 \mod 3 $
  - **but necessity doesn't exist** but pre-existing adequacy (no false negatives) can be exploited; iterative validation based on randomization (+ luck) can K questions
3. hash do prefix and after due to the modulus nature can still on $\mod 3 $ directly determine the interval OK; see code

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
