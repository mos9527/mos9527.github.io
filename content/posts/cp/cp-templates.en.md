---
author: mos9527
lastmod: 2025-05-14T10:08:15.026000+08:00
title: Competitive Programming - Algorithm Templates And Problem Sets (C++)
tags: ["ACM","Competeive Programming","XCPC","(Code) Templates","Problem sets","Codeforces","C++"]
categories: ["Problem Solutions", "Competeive Programming", "Collection/compilation"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---
# Preface

Reference primarily from [Introductory Classics for Algorithmic Competition: A Training Guide (https://cread.jd.com/read/startRead.action?bookId=30133704&readType=1)、[OIWiki](https://oi-wiki.org/)、[CP Algorithms](https://cp-algorithms.com/) and other resources and multiple blogs and courses, authored under their own code breeze

**Note:** Some implementations may use newer language features, so please modify them for use on older OJs; **In principle, the code provided is compatible with compilers that comply with the Cpp20 standard and above**.

# Header
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
const ll DIM = 1e5;
const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const lf EPS = 1e-8;
int main() {
    fast_io();
    /* El Psy Kongroo */

    return 0;
}
```
# Misc

- Open the GCC debug container: `add_compile_definitions(-D_GLIBCXX_DEBUG)`

# Mathematics

## Binary Exponentiation

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
```

## Linear algebra
### Matrix
- https://codeforces.com/gym/105170/submission/261977724
- https://codeforces.com/gym/105336/submission/280576093 (D encoder-decoder)
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

### Linear bases

- https://oi.men.ci/linear-basis-notes/
- https://www.luogu.com.cn/article/zo12e4s5
- https://codeforces.com/gym/105336/submission/280570848（J 找最小）

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

## Miscellaneous number theory
### Pisano cycle

- https://codeforces.com/contest/2033/submission/287844746

*Retrieved from https://oi-wiki.org/math/combinatorics/fibonacci/#%E7%9A%AE%E8%90%A8%E8%AF%BA%E5%91%A8%E6%9C%9F*

The minimum positive period of the Fibonacci series in the sense of mode $m$ is called the [Pisano cycle](https://en.wikipedia.org/wiki/Pisano_period)
The Pisano period is always no more than $6m$ and the equality sign is taken only if it satisfies the form $m=2\times 5^k$.

When it is necessary to calculate the value of the $n$th Fibonacci mode $m$, if $n$ is very large, it is necessary to calculate the period of the Fibonacci mode $m$. Of course, only the period needs to be calculated, not necessarily the least positive period.
It is easy to verify that the least positive period of the Fibonacci numbers modulo $2$ is $3$ and the least positive period modulo $5$ is $20$.
Clearly, if $a$ and $b$ are mutually prime, the Pisano period of $ab$ is the least common multiple of the Pisano period of $a$ and the Pisano period of $b$.

Conclusion 2: On the odd prime $p\equiv 2,3 \pmod 5$, $2p+2$ is the period of the Fibonacci modulus $p$. That is, the Pisano period of the odd prime $p$ divides $2p+2$.

Conclusion 3: For a prime $p$, $M$ is the cycle of the Fibonacci modulus $p^{k-1}$, which is equivalent to $Mp$ being the cycle of the Fibonacci modulus $p^k$. In particular, $M$ is a Pisano cycle of modulus $p^{k-1}$, which is equivalent to $Mp$ being a Pisano cycle of modulus $p^k$.

---
** Thus it is also equivalent that $Mp$ is the period of the Fibonacci modulus $p^k$. **
** Because the periods are equivalent, the least positive period is also equivalent. **

## Computational geometry

### Two-dimensional geometry

- https://codeforces.com/gym/104639/submission/281132024

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
    inline static lf dist_sq(vec2 const& a, vec2 const& b) {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    }
    inline static lf dist(vec2 const& a, vec2 const& b) {
        return sqrt(vec2::dist_sq(a, b));
    }
    inline static lf cross(vec2 const& a, vec2 const& b) {
        return a.x * b.y - a.y * b.x;
    }
    inline static lf dot(vec2 const& a, vec2 const& b) {
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
```

#### 2D Convex Packet

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



## Number of combinations

Lucas：$$\binom{n}{m}\bmod p = \binom{\left\lfloor n/p \right\rfloor}{\left\lfloor m/p\right\rfloor}\cdot\binom{n\bmod p}{m\bmod p}\bmod p$$​
```c++
namespace comb {
	ll fac[MOD], ifac[MOD]; // x!, 1/x!
	void prep(ll N = MOD - 1) {
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


## Number theory
### Multiplying inverse elements
- https://acm.hdu.edu.cn/showproblem.php?pid=7437

Given a prime $m$, find the inverse of $a$, $a^{-1}$.

- Euler's theorem knows that $a^{\phi (m)} \equiv 1 \mod m$
- For prime $m$, $\phi (m) = m - 1$
- This scenario is Fermat's Little Theorem, i.e. $a^{m - 1} \equiv 1 \mod m$
- Multiplying left and right simultaneously by $a^{-1}$, gives $a ^ {m - 2} \equiv a ^ {-1} \mod m$
- i.e. `a_inv = binpow_mod(a, m - 2, m)`

### Eratosthenes sieve

- https://oi-wiki.org/math/number-theory/sieve
- https://www.luogu.com.cn/problem/P2158 (Euler function)

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

### Decompose the prime factor

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

# Graphology

## Topological sorting
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
## Shortest circuit

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

- https://codeforces.com/group/bAbX7h3CX1/contest/554012/submission/285834927 (jump points/validation route points)

```c++
#define INF 1e18
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

## Minimum spanning tree

### Kruskal 

```c++
struct dsu {
	vector<ll> pa;
	dsu(const ll size) : pa(size) { iota(pa.begin(), pa.end(), 0); }; // 初始时，每个集合都是自己的父亲
	inline bool is_root(const ll leaf) { return pa[leaf] == leaf; }
	inline ll find(const ll leaf) { return is_root(leaf) ? leaf : find(pa[leaf]); } // 路径压缩
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

## Euler's circuit

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

- RMQ (ST table)

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

- Multiplying Ideas
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

## Tree diameter

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

## Dinic Maximum Flow

- https://www.cnblogs.com/SYCstudio/p/7260613.html
- https://codeforces.com/gym/105336/submission/280592598 (G. Saturday Madness)

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

## Tree chain dissection / HLD

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

# Dynamic Programming / DP

Move to [DP type topic](https://mos9527.github.io/posts/cp/dp-problems/)

# Data Structures / DS

## RMQ Series
### Sliding window (monotonic queue)

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

### The line tree

Move to [Line Tree topic](https://mos9527.github.io/posts/cp/segment-tree-problems/)

### ST table

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



### Tree arrays
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
#### Support for non-differentiable query templates

- Explanation: https://oi-wiki.org/ds/fenwick/#树状数组维护不可差分信息
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

#### Interval Templates

- Explanation: https://oi-wiki.org/ds/fenwick/#区间加区间和
- Title: https://hydro.ac/d/ahuacm/p/Algo0304
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

## Priority queue (binary heap)

> ```c++
> auto pp = PRED( elem, lhs.w > rhs.w);
> priority_queue < elem, vector<elem>, decltype(pp)> Q {pp};
> ```

## DSU

- No consideration of side rights

```C++
struct dsu {
    vector<ll> pa;
    dsu(const ll size) : pa(size) { iota(pa.begin(), pa.end(), 0); }; // 初始时，每个集合都是自己的父亲
    inline bool is_root(const ll leaf) { return pa[leaf] == leaf; }
    inline ll find(const ll leaf) { return is_root(leaf) ? leaf : find(pa[leaf]); } // 路径压缩
    inline void unite(const ll x, const ll y) { pa[find(x)] = find(y); }
};
```

- Need to calculate the distance to the root
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

# String
## AC automatics
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



## String hash
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

# Miscellaneous

## Two points

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

## Replacement ring

- https://www.cnblogs.com/TTS-TTS/p/17047104.html

The length $n$-arrangement $p$ in which elements $i,j$ are exchanged $k$ times so that it becomes an arrangement $p'$, find the minimum $k$?

- The two permutations are sequentially connected to the edge; clearly there are $n$ unitary rings in the graph when the permutations coincide
- Swapping once in a ring divides one more ring; remember that the size of the ring is $s$
- Clearly, the division into $n$ unitary rings is the division into rings $s-1$ times; remember that there are $m$ rings
- can be obtained as $k = \sum_{1}^{m}{s - 1} = n - m$

Attachment: https://codeforces.com/contest/2033/submission/287844212

- Unlike general sorting problems, the arrangements do not need to be identical here; $p_i = i, p_i = p_{{i}_{i}}$ are both possible
- means that the final ring size to be wanted can also be $2$, when clearly the size is better; changing the computation of $k$ to $k = \sum_{1}^{m}{\frac{s - 1}{2}}$ is sufficient

## Discrete

For large $a_i$ but small $n$ cases

- Online `map` writing

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



- Offline `map` writing

```c++
map<ll, ll> R;
for (auto& ai : a) R[ai] = 1;
vec Ri; // kth-big
ll cnt = 0; for (auto& [x, i] : R) i = cnt++, Ri.push_back(x);
for (auto& [ai, bi] : a) ai = R[ai], bi = R[bi];
```

- Offline `set` writing
  - Note that the complexity ($R(x)$) of this `set`, if it is an STL set, is actually $O(n)$
    - See https://codeforces.com/blog/entry/123961 for details
    - [TL;DR `std::distance`** operates $O(1)$ for *random*** iterators only](https://en.cppreference.com/w/cpp/iterator/distance), $O(n)$ for all other iterators (if applicable)
    - Generating TLE can be seen at https://codeforces.com/contest/2051/submission/298511255
      - `map` solution (AC): https://codeforces.com/contest/2051/submission/298511985

```c++
set<ll> Rs;
vector<II> a(n);
for (auto& ai : a) Rs.insert(ai);
vec Ri(R.begin(), R.end()); // kth-big
auto R = [&](ll x) -> ll { return distance(Rs.begin(), Rs.lower_bound(x)); };
```



## MSVC needs a universal header too!!!!

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
