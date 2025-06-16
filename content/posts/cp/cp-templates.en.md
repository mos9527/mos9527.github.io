---
author: mos9527
lastmod: 2025-06-08T22:50:14.951000+08:00
title: Competitive Programming - Algorithm Templates And Problem Sets (C++)
tags: ["ACM","Competeive Programming","XCPC","(Code) Templates","Problem sets","Codeforces","C++"]
categories: ["Problem Solutions", "Competeive Programming", "Collection/compilation"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---
# Preface

Reference primarily from [Introductory Classics for Algorithmic Competition: A Training Guide](https://cread.jd.com/read/startRead.action?bookId=30133704&readType=1)、[OIWiki](https://oi-wiki.org/)、[CP Algorithms](https://cp-algorithms.com/) and other resources and multiple blogs and courses, authored under their own code breeze

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
- https://codeforces.com/gym/105336/submission/280570848 (J 找最小）

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
- Topic：https://acm.hdu.edu.cn/showproblem.php?pid=7463

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

# Treap

**ATTENTION:** Out the door and to the left. https://caterpillow.github.io/byot Thanks meow.

AKA **Cartesian Tree**, **Randomized BST**; supports $log n$ insertion, deletion, lookup and closure operations (`push_up`)

- https://cp-algorithms.com/data_structures/treap.html
- https://oi.baoshuo.ren/fhq-treap

## A. `std::set` class containers

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

## B. Lazy Marker Interval Treap

- 1. Provide **delete** operation, interval modification; do not support lookup; support RMQ

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

- 2. Single point of modification (no lazy tagging)

 See Above/Reference https://mos9527.com/posts/cp/gcd-problems/#p11373-czoi-r2%E5%A4%A9%E5%B9%B3



# Preface

References mainly from https://cp-algorithms.com/algebra/fft.html, https://en.wikipedia.org/wiki/Discrete_Fourier_transform, https://oi.wiki/math/ poly/fft/

~~to take care of a certain OJ~~ this article routines (except miscellaneous) C++ standard only needs `11`; **[board portal](#reference)**,[topic portal](#problems)

## define

- The $DFT$ of a polynomial $A$ is the value of $A$ in each unit root $w_{n, k} = w_n^k = e^{\frac{2 k \pi i}{n}}$

$$
\begin{align}
\text{DFT}(a_0, a_1, \dots, a_{n-1}) &= (y_0, y_1, \dots, y_{n-1}) \newline
&= (A(w_{n, 0}), A(w_{n, 1}), \dots, A(w_{n, n-1})) \newline
&= (A(w_n^0), A(w_n^1), \dots, A(w_n^{n-1}))
\end{align}
$$

- $IDFT$ ($InverseDFT$) i.e., the coefficients of the polynomial $A$ are recovered from these values $(y_0, y_1, \dots, y_{n-1})$

$$
\text{IDFT}(y_0, y_1, \dots, y_{n-1}) = (a_0, a_1, \dots, a_{n-1})
$$

- The unit root has the following properties

  - productive
    $$
    w_n^n = 1 \newline
    w_n^{\frac{n}{2}} = -1 \newline
    w_n^k \ne 1, 0 \lt k \lt n
    $$
    
  - All unit roots sum to $0$
    $$
    \sum_{k=0}^{n-1} w_n^k = 0
    $$
    This is obvious when looking at the $n$-side symmetry using Euler's formula $e^{ix} = cos x + i\ sin x$

## applications

Consider two polynomials $A, B$ multiplied together
$$
(A \cdot B)(x) = A(x) \cdot B(x)
$$

- Obviously applying $DFT$ yields

$$
DFT(A \cdot B) = DFT(A) \cdot DFT(B)
$$

- The coefficients of $A \cdot B$ are easy to find

$$
A \cdot B = IDFT(DFT(A \cdot B)) = IDFT(DFT(A) \cdot DFT(B))
$$

## Inverse operation (IDFT)

Recall the definition of $DFT$
$$
\text{DFT}(a_0, a_1, \dots, a_{n-1}) = (A(w_n^0), A(w_n^1), \dots, A(w_n^{n-1}))
$$
- Written in [matrix form](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT) that is

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
- Then the $DFT$ operation is

$$
F\begin{pmatrix}
a_0 \newline a_1 \newline a_2 \newline a_3 \newline \vdots \newline a_{n-1}
\end{pmatrix} = \begin{pmatrix}
y_0 \newline y_1 \newline y_2 \newline y_3 \newline \vdots \newline y_{n-1}
\end{pmatrix}
$$
- The simplification has

$$
y_k = \sum_{j=0}^{n-1} a_j w_n^{k j},
$$

where the van der Monde array $M$ ranks are orthogonal in all terms, [conclusions can be made](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT).

$$
F^{-1} = \frac{1}{n} F^\star, F_{i,j}^\star = \overline{F_{j,i}}
$$

existing
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
- Then the $IDFT$ operation is

$$
\begin{pmatrix}
a_0 \newline a_1 \newline a_2 \newline a_3 \newline \vdots \newline a_{n-1}
\end{pmatrix} = F^{-1} \begin{pmatrix}
y_0 \newline y_1 \newline y_2 \newline y_3 \newline \vdots \newline y_{n-1}
\end{pmatrix}
$$
- The simplification has

$$
a_k = \frac{1}{n} \sum_{j=0}^{n-1} y_j w_n^{-k j}
$$
### reach a verdict

- **Note that $w_i$ uses the conjugate i.e. $n \cdot \text{IDFT}$**
- The $DFT,IDFT$ operation can be realized at the same time with a few tweaks in the implementation; it will be used next.

## Realization (FFT)

The plain envelope time complexity is $O(n^2)$ and is not elaborated here

The process of $FFT$ is as follows

- Let $A(x) = a_0 x^0 + a_1 x^1 + \dots + a_{n-1} x^{n-1}$, split into two sub-polynomials by parity

$$
\begin{align}
A_0(x) &= a_0 x^0 + a_2 x^1 + \dots + a_{n-2} x^{\frac{n}{2}-1} \newline
A_1(x) &= a_1 x^0 + a_3 x^1 + \dots + a_{n-1} x^{\frac{n}{2}-1}
\end{align}
$$
- Apparently.

$$
A(x) = A_0(x^2) + x A_1(x^2).
$$

- found 
$$
\left(y_k^0 \right)_{k=0}^{n/2-1} = \text{DFT}(A_0)
$$

$$
\left(y_k^1 \right)_{k=0}^{n/2-1} = \text{DFT}(A_1)
$$

$$
y_k = y_k^0 + w_n^k y_k^1, \quad k = 0 \dots \frac{n}{2} - 1.
$$
- For the second half $\frac{n}{2}$ there are

$$
\begin{align}
y_{k+n/2} &= A\left(w_n^{k+n/2}\right) \newline
&= A_0\left(w_n^{2k+n}\right) + w_n^{k + n/2} A_1\left(w_n^{2k+n}\right) \newline
&= A_0\left(w_n^{2k} w_n^n\right) + w_n^k w_n^{n/2} A_1\left(w_n^{2k} w_n^n\right) \newline
&= A_0\left(w_n^{2k}\right) - w_n^k A_1\left(w_n^{2k}\right) \newline
&= y_k^0 - w_n^k y_k^1
\end{align}
$$

- That is, $y_{k+n/2} = y_k^0 - w_n^k y_k^1$, which is formally very close to $y_k$. Summarize:

$$
\begin{align}
y_k &= y_k^0 + w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1, \newline
y_{k+n/2} &= y_k^0 - w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1.
\end{align}
$$

This is known as **"butterfly optimization ”**.

### reach a verdict

- Clearly the merger cost is $O(n)$; by $T_{\text{DFT}}(n) = 2 T_{\text{DFT}}\left(\frac{n}{2}\right) + O(n)$ it is known that $FFT$ can solve the problem in $O(nlogn)$ time
- The subsumption implementation will also be simple

### Code (consolidation)

Also known as **Cooley-Tukey algorithm**; Partitioned Solution

- If $w_n$ is implemented using `std::complex` it is straightforward to find $w_n = e^{\frac{2\pi i}{n}}$ using [`std::exp` with its own specialization](https://en.cppreference.com/w/cpp/numeric/complex/exp)
- Alternatively, using Euler's formula $e^{ix} = cos x + i\ sin x$ one can construct `Complex w_n{ .real = cos(2 * PI / n), .imag = sin(2 * PI / n) }`
- Combining the $DFT$, $IDFT$ relationship described previously, use $w_n = -e^{\frac{2\pi i}{n}}$ and divide by $n$ to find $IDFT$.
- **Time complexity $O(n\log n)$**, due to merging after halving, **Space complexity $O(n)$**

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
    // Note that dividing log2(n) times 2 is dividing 2^log2(n) = n
    if (invert) 
      A[k] /= 2, A[k + n / 2] /= 2;
		w_k *= w_n;
	}   
}
void FFT(cvec& a) { FFT(a, false); }
void IFFT(cvec& y) { FFT(y, true); }
```

### Code (multiplication)

The extra space introduced by the subsumption method can actually be optimized away - the multiplicative recursive solution is introduced next.

- Observe the order of final backtracking in subsumption (with $n=8$)
  -   Initial sequence is $\{x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7\}$
  -   After one bisection $\{x_0, x_2, x_4, x_6\},\{x_1, x_3, x_5, x_7 \}$
  -   After two bisections $\{x_0,x_4\} \{x_2, x_6\},\{x_1, x_5\},\{x_3, x_7 \}$
  -   After three times bisection $\{x_0\}\{x_4\}\{x_2\}\{x_6\}\{x_1\}\{x_5\}\{x_3\}\{x_7 \}$

- If you pay enough attention you can see the pattern as follows

```python
In [17]: [int(bin(i)[2:].rjust(3,'0')[::-1],2) for i in range(8)]
Out[17]: [0, 4, 2, 6, 1, 5, 3, 7]

In [18]: [bin(i)[2:].rjust(3,'0')[::-1] for i in range(8)]
Out[18]: ['000', '100', '010', '110', '001', '101', '011', '111']

In [19]: [bin(i)[2:].rjust(3,'0') for i in range(8)]
Out[19]: ['000', '001', '010', '011', '100', '101', '110', '111']
```

- i.e., the binary inverse (symmetric) order, noting that the inverse order is $R(x)$

```c++
auto R = [n](ll x) {
    ll msb = ceil(log2(n)), res = 0;
    for (ll i = 0;i < msb;i++)
        if (x & (1 << i))
            res |= 1 << (msb - 1 - i);
    return res;
};
```

- From bottom to top, recursively in lengths $2,4,6,\cdots,n$, keeping this order will accomplish the task accomplished by the method of subsumption
- Again, because of the symmetry, the reordering can also be done in $O(n)$; ** time complexity $O(n\log n)$, space complexity $O(1)$**

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
    // From bottom to top n_i = 2, 4, 6, ... ,n directly recursive
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

## Number Theoretic Transformations (NTT)

Calculations in the imaginary domain inevitably have accuracy problems; the larger the number the greater the error and because $exp$ (or $sin, cos$) is used it is extremely difficult to correct. The following describes number-theoretic transformations (or fast number-theoretic transformations) to allow absolutely correct $O(nlogn)$ envelopes to be accomplished in the modulus domain.

- DFT in the domain of primes $p$, $F={\mathbb {Z}/p}$; note that the nature of the unit root is preserved under modulus

- It is also obvious that there is $$(w_n^m)^2 = w_n^n = 1 \pmod{p}, m = \frac{n}{2}$$; using this property we can find $$w_n^k$$ by using the fast power

- Of course, we need to find such $g$ of $g_n^n \equiv 1 \mod p$ such that $g_n$ is equivalent to $w_n$


### original root

> The following is taken from: https://cp-algorithms.com/algebra/primitive-root.html#algorithm-for-finding-a-primitive-root, 

Definition:** For any $a$ and the existence of $a$, $n$ mutually prime and $g^k \equiv a \mod n$, $g$ is said to be the original root of mod $n$. **
CONCLUSION: **The original root of $n$, $g$,$g^k \equiv 1 \pmod n$, $k=\phi(n)$ is the minimal solution of $k$ **
An algorithm for finding the original root is described below:

  - Euler definition: if $\gcd(a, n) = 1$, then $a^{\phi(n)} \equiv 1 \pmod{n}$
  - For the exponent $p$, the parsimonious solution is the $O(n^2)$ time check $g^d, d \in [0,\phi(n)] \not\equiv 1 \pmod n$

  - There exist such $O(\log \phi (n) \cdot \log n)$ solutions:
    - Find $\phi(n)$ factor $p_i \in P$ and check $g \in [1, n]$
    - For all $p_i \in P$, $g ^ { \frac {\phi (n)} {p_i}} \not\equiv 1\pmod n $, this root is an original root
  - For proof, please refer to the original article

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

### Realization (multiplication)

In summary, there are pairs of primes $p$ and their primitive roots $g$ that can do the unit root property in the moduli domain; commonly used ones are ($p=7 \times 17 \times 2^{23}+1=998244353, g=3$,$p=7 \times 2^{20} + 1 =7340033$)

The Euler function for these numbers satisfies the form $\phi(p) = p - 1 = c \times 2^k$, recalling that the Euler function $g^{p-1} \equiv 1 \pmod n$, which obviously fits nicely with what we're going to do next: traversing up to length $n_i$, $w_{n_i} = e^{\frac{2\pi}{n_i}}}$ i.e., the equivalent of $g^{\frac{p-1}{n_i}}$. Since $n_i$ is multiplied, $\frac{p-1}{n_i}$ is simply shifted, and also integer division will be error-free.


### Code (multiplication)

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
  // From bottom to top n_i = 2, 4, 6, ... ,n directly recursive
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

## Cosine Transform (DCT)

See below for the implementation; the following $\text{DCT-II, DCT-III}$ forms are used:

- DCT-2 and its regularization factor

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
  

## Reference

#### lib/poly.hpp

The $\text{DFT/FFT/(F)NTT}$ magic mentioned in this article is summarized below, out-of-the-box.

If you're in a good mood (link under `tbb`? Or is it `msvc` you're using...) The 2D FFT in this implementation allows for parallelism (`execution = std::execution::par_unseq`)

```c++
/*** POLY.H - 300LoC Single header Polynomial transform library
 * - Supports 1D/2D (I)FFT, (I)NTT, DCT-II & DCT-III with parallelism guarantees on 2D workloads.
 * - Battery included. Complex, Real and Integer types supported with built-in convolution helpers;
 * ...Though in truth, use something like FFTW instead. This is for reference and educational purposes only. */
#pragma once
#define _POLY_HPP
#include <cassert>
#include <cmath>
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include <execution>
namespace Poly {
    using ll = long long;   using lf = double;  using II = std::pair<ll, ll>;
    const lf PI = std::acos(-1);
    const ll NTT_Mod = 998244353, NTT_Root = 3;
    using Complex = std::complex<lf>;
    using CVec = std::vector<Complex>;
    using RVec = std::vector<lf>;
    using IVec = std::vector<ll>;
    using CVec2 = std::vector<CVec>;
    using RVec2 = std::vector<RVec>;
    using IVec2 = std::vector<IVec>;
#if __cplusplus >= 202002L
    template <typename T> concept ExecutionPolicy = std::is_execution_policy_v<T>;
    template <typename T> concept Vec1D = std::is_same_v<T, CVec> || std::is_same_v<T, RVec> || std::is_same_v<T, IVec>;
    template <typename T> concept Vec2D = std::is_same_v<T, CVec2> || std::is_same_v<T, RVec2> || std::is_same_v<T, IVec2>;
#else
#define ExecutionPolicy class
#define Callable class
#define Vec2D class
#endif
    namespace utils {
        inline RVec as_real(CVec const& a) {
            RVec res(a.size());
            for (ll i = 0; i < a.size(); i++)
                res[i] = a[i].real();
            return res;
        }
        inline RVec2 as_real(CVec2 const& a) {
            RVec2 res(a.size());
            for (ll i = 0; i < a.size(); i++)
                res[i] = as_real(a[i]);
            return res;
        }
        inline CVec as_complex(RVec const& a) {
            return {a.begin(), a.end()};
        }
        inline CVec2 as_complex(RVec2 const& a) {
            CVec2 res(a.size());
            for (ll i = 0; i < a.size(); i++)
                res[i] = as_complex(a[i]);
            return res;
        }
        inline bool is_pow2(ll x) {
            return (x & (x - 1)) == 0;
        }
        inline ll to_pow2(ll n) {
            n = ceil(log2(n)), n = 1ll << n;
            return n;
        }
        inline ll to_pow2(ll a, ll b) {
            return to_pow2(a + b);
        }
        inline II to_pow2(II const& a, II const& b) {
            return { to_pow2(a.first + b.first), to_pow2(a.second + b.second)};
        }
        template<typename T> inline void resize(T& a, ll n) { a.resize(n); }
        template<typename T> inline void resize(T& a, II nm) {
            a.resize(nm.first);
            for (auto& row : a) row.resize(nm.second);
        }
        template<typename T, typename Ty> inline void resize(T& a, II nm, Ty fill) {
            auto [N,M] = nm;
            ll n = a.size(), m = a.size() ? a[0].size() : 0;
            resize(a, nm);
            if (M > m) {
                for (ll i = 0;i < n;++i)
                    for (ll j = m; j < M; ++j)
                        a[i][j] = fill;
            }
            if (N > n) {
                for (ll i = n; i < N; ++i)
                    for (ll j = 0; j < M; ++j)
                        a[i][j] = fill;
            }
        }
    }
    namespace details {
        inline ll qpow(ll a, ll b, ll m) {
            a %= m;
            ll res = 1;
            while (b > 0) {
                if (b & 1) res = res * a % m;
                a = a * a % m;
                b >>= 1;
            }
            return res;
        }
        inline ll bit_reverse_perm(ll n, ll x) {
            ll msb = ceil(log2(n)), res = 0;
            for (ll i = 0; i < msb; i++)
                if (x & (1ll << i))
                    res |= 1ll << (msb - 1 - i);
            return res;
        }
        // Cooley-Tukey FFT
        inline CVec& FFT(CVec& a, bool invert) {
            const ll n = a.size();
            assert(utils::is_pow2(n));
            for (ll i = 0, r; i < n; i++)
                if (i < (r = bit_reverse_perm(n, i)))
                    swap(a[i], a[r]);
            for (ll n_i = 2; n_i <= n; n_i <<= 1) {
                lf w_ang = 2 * PI / n_i;
                // Complex w_n = exp(Complex{ 0, ang });
                Complex w_n = { std::cos(w_ang), std::sin(w_ang) };
                if (invert) w_n = conj(w_n);
                for (ll i = 0; i < n; i += n_i) {
                    Complex w_k = Complex{ 1, 0 };
                    for (ll j = 0; j < n_i / 2; j++) {
                        Complex u = a[i + j], v = a[i + j + n_i / 2] * w_k;
                        a[i + j] = u + v;
                        a[i + j + n_i / 2] = u - v;
                        if (invert)
                            a[i + j] /= 2, a[i + j + n_i / 2] /= 2;
                        w_k *= w_n;
                    }
                }
            }
            return a;
        }
        // Cooley-Tukey FFT in modular arithmetic / Number Theoretic Transform
        inline IVec& NTT(IVec& a, ll p, ll g, bool invert) {
            const ll n = a.size();
            assert(utils::is_pow2(n));
            for (ll i = 0, r; i < n; i++)
                if (i < (r = bit_reverse_perm(n, i)))
                    swap(a[i], a[r]);
            const ll inv_2 = qpow(2, p - 2, p);
            for (ll n_i = 2; n_i <= n; n_i <<= 1) {
                ll w_n = qpow(g, (p - 1) / n_i, p);
                if (invert)
                    w_n = qpow(w_n, p - 2, p);
                for (ll i = 0; i < n; i += n_i) {
                    ll w_k = 1;
                    for (ll j = 0; j < n_i / 2; j++) {
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
        // (Normalized Output) Discrete Cosine Transform (DCT-II), aka DCT
        inline RVec& DCT2(RVec& a) {
            // https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
            // https://zh.wikipedia.org/wiki/离散余弦变换#方法一[8]
            const ll n = a.size(), N = 2 * n;
            const lf k2N = std::sqrt(N), k4N = std::sqrt(2.0 * N);
            assert(utils::is_pow2(n));
            CVec a_n2 = utils::as_complex(a);
            a_n2.resize(N);
            std::copy(a_n2.begin(), a_n2.begin() + n, a_n2.begin() + n);
            std::reverse(a_n2.begin() + n, a_n2.end());
            FFT(a_n2, false);
            for (ll m = 0; m < n;m++) {
                lf w_ang = PI * m / N;
                Complex w_n = { std::cos(w_ang), std::sin(w_ang) };
                a[m] = (a_n2[m] * w_n).real(); // imag = 0
                a[m] /= (m == 0 ? k4N : k2N);
            }
            return a;
        }
        // (Normalized Input) Discrete Cosine Transform (DCT-III), aka IDCT
        inline RVec& DCT3(RVec& a) {
            // https://dsp.stackexchange.com/questions/51311/computation-of-the-inverse-dct-idct-using-dct-or-ifft
            // https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
            const ll n = a.size(), N = 2 * n;
            const lf k2N = std::sqrt(N);
            assert(utils::is_pow2(n));
            CVec a_n = utils::as_complex(a);
            a[0] /= std::sqrt(2.0);
            for (ll m = 0; m < n;m++) {
                lf w_ang = -PI * m / N;
                Complex w_n = { std::cos(w_ang), std::sin(w_ang) };
                a[m] *= k2N;
                a_n[m] = a[m] * w_n;
            }
            FFT(a_n, true);
            for (ll m = 0; m < n/2;m++)
                a[m * 2] = a_n[m].real(),
                a[m * 2 + 1] = a_n[n - m - 1].real();
            return a;
        }
    }
    namespace transform {
        template<Vec2D T, ExecutionPolicy Execution, class Transform> T& __transform2D(T& a, Transform const& transform, Execution const& execution) {
            const ll n = a.size(), m = a[0].size();
            IVec mn(max(m, n)); iota(mn.begin(), mn.end(), 0);
            for_each(execution, mn.begin(), mn.begin() + n, [&](ll row) {
                transform(a[row]);
            });
            for_each(execution, mn.begin(), mn.begin() + m, [&](ll col){
                typename T::value_type c(n);
                for (ll row = 0; row < n; row++)
                    c[row] = a[row][col];
                transform(c);
                for (ll row = 0; row < n; row++)
                    a[row][col] = c[row];
            });
            return a;
        }
        inline CVec& DFT(CVec& a) {
            return details::FFT(a, false);
        }
        inline CVec& IDFT(CVec& a) {
            return details::FFT(a, true);
        }
        inline IVec& NTT(IVec& a, ll p, ll g) {
            return details::NTT(a, p, g, false);
        }
        inline IVec& INTT(IVec& a, ll p, ll g) {
            return details::NTT(a, p, g, true);
        }
        inline RVec& DCT(RVec& a) {
            return details::DCT2(a);
        }
        inline RVec& IDCT(RVec& a) {
            return details::DCT3(a);
        }
        template<ExecutionPolicy Exec>
        CVec2& DFT2(CVec2& a, Exec execution) {
            return __transform2D(a, DFT, execution);
        }
        template<ExecutionPolicy Exec>
        CVec2& IDFT2(CVec2& a, Exec execution) {
            return __transform2D(a, IDFT, execution);
        }
        template<ExecutionPolicy Exec>
        RVec2& DCT2(RVec2& a, Exec execution) {
            return __transform2D(a, DCT, execution);
        }
        template<ExecutionPolicy Exec>
        RVec2& IDCT2(RVec2& a, Exec execution) {
            return __transform2D(a, IDCT, execution);
        }
    }
    namespace conv {
        template<Vec1D T, class Transform, class InvTransform>
        T& __convolve(T& a, T& b, Transform const& transform, InvTransform const& inv_transform) {
            ll n = utils::to_pow2(a.size(), b.size());
            utils::resize(a, n), utils::resize(b, n);
            transform(a), transform(b);
            for (ll i = 0; i < n; i++) a[i] *= b[i];
            inv_transform(a);
            return a;
        }
        template<Vec2D T, class Transform, class InvTransform, ExecutionPolicy Exec>
        T& __convolve2D(T& a, T& b, Transform const& transform, InvTransform const& inv_transform, Exec const& execution) {
            ll n = a.size(), m = a[0].size();
            ll k = b.size(), l = b[0].size();
            II NM = utils::to_pow2({ n,m },{ k,l });
            auto [N, M] = NM;
            utils::resize(a, NM), utils::resize(b, NM);
            transform(a, execution), transform(b, execution);
            for (ll i = 0; i < N; ++i) for (ll j = 0; j < M; ++j) a[i][j] *= b[i][j];
            inv_transform(a, execution);
            a.resize(n + k - 1);
            for (auto& row : a) row.resize(m + l - 1);
            return a;
        }
        // Performs complex convolution with DFT
        CVec& convolve(CVec& a, CVec& b) {
            return __convolve(a, b,transform::DFT,transform::IDFT);
        }
        // Performs modular convolution with NTT
        IVec& convolve(IVec& a, IVec& b, ll mod=NTT_Mod, ll root=NTT_Root) {
            return __convolve(a, b,[=](IVec& x){return transform::NTT(x,mod,root);},[=](IVec& x){return transform::INTT(x,mod,root);});
        }
        // Performs real-valued convolution with DCT
        RVec& convolve(RVec& a, RVec& b) {
            return __convolve(a, b, transform::DCT, transform::IDCT);
        }
        // Performs complex 2D convolution with DFT
        template<ExecutionPolicy Exec> CVec2& convolve2D(CVec2& a, CVec2& b, Exec const& execution) {
            return __convolve2D(a, b, transform::DFT2<Exec>, transform::IDFT2<Exec>, execution);
        }
        // Performs real-valued 2D convolution with DCT
        template<ExecutionPolicy Exec> RVec2& convolve2D(RVec2& a, RVec2& b, Exec const& execution) {
            return __convolve2D(a, b, transform::DCT2<Exec>, transform::IDCT2<Exec>, execution);
        }
    }
}
```

## Problems

### A * B

- https://acm.hdu.edu.cn/showproblem.php?pid=1402

- large integer multiplication

- The $10$ progression, with each digit from lowest to highest being $d_i$ can be seen as the polynomial $A(x) = x^n \times d_n + ... + x^1 \times d_1 + x^0 \times d_0$ at $x=10$.

- Two decimal numbers can be seen as $A(x), B(x)$, to find $A(x) * B(x)$ that is, to find $AB(x)$, by the above mentioned $\text{DFT,IDFT}$ relationship is known we can use this by $\text{FFT}$ in $O(n\log n)$ time to compute such numbers

- Since it is $10$-decimal, the coefficients of the final polynomial correspond to the $x=10$ solution; note the rounding.

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

### A + B Frequency

- https://open.kattis.com/problems/aplusb

- Given a sequence of integers $A$,$B$, find the possible and number of outcomes of $a \in A, b \in B, a + b$

- Consider this transformation into a polynomial problem: let $ P_a(x) = \sum x^{A_i}, P_b(x) = \sum x^{B_i} $

- Given the examples $a = [1,~ 2,~ 3], b = [2,~ 4]$, the $P_aP_b$ so constructed have
  $$
  (1 x^1 + 1 x^2 + 1 x^3) (1 x^2 + 1 x^4) = 1 x^3 + 1 x^4 + 2 x^5 + 1 x^6 + 1 x^7
  $$

- In this way the index is found to correspond to the coefficients, i.e. the various possible quantities

### cyclic multiplication of numbers

- Given a sequence of long $n$ integers $A$,$B$ such that $C_{p,i} = B_{(i + p) \% n}$, find any $A \cdot C_p$ value

- Recall that the coefficients of the polynomial multiplication i.e. such an envelope
  $$
  c[k] = \sum_{i+j=k} a[i] b[j]
  $$
  
- Let $A$ be in reverse order, and then complement $n$ $0$; let $B$ complement $B$ itself

- i.e., $A_i = 0 (i \gt n - 1)$, so that at this point we have

$$
c[k] = \sum_{i+j=k} a[i] b[j] = \sum_{i=0}^{n-1} a[i] b[k-i]
$$

- For $i + k > n$, $b[(i + k) \% n] = b[i + k - n + 1]$; the above equation is the result when $p = k - n + 1

- That is, $c[p + n - 1]$ corresponds to the original $A \cdot C_p$-value at $p$.

### string match

- Given a string $S$ and a pattern string $P$, for each character $C_i\in[0,26]$, count the total number of occurrences of $P$ in $S$.
  - Construct the polynomial $A(x) = \sum a_i x^i$, where $a_i = e^{\frac{2 \pi S_i}{26}}$
  - Let $S$ be its inverse order and construct the polynomial $B(x)=\sum b_i x^i$, where $b_i = e^{-\frac{2 \pi P_i}{26}}$
- Note that after the envelope

$$
c_{m-1+i} = \sum_{j = 0}^{m-1} a_{i+j} \cdot b_{m-1-j} = \sum_{j=0}^{m-1}e^{\frac{2 \pi S_{i+j} - 2\pi P_j}{26}}
$$

Clearly if the match then $e^{\frac{2 \pi S_{i+j} - 2\pi P_j}{26}} = e^0 = 1$, then all matches if and only if $c_{m-1+i} = m$, and the pattern string $P$ has occurrences at $S_i

#### Attachment: partial match

- Let some of the characters in $P$ be arbitrary, then inverting the order makes these positions polynomial coefficients $b_i=0$; with $x$ such positions
- Recalling the above equation, it is easy to see that there are $c_i = \sum_{j=0}^{m-1-x} e^{\cdots} + \sum_0^x 0$ when and only when these coefficients are matched.
- Clearly, when $c_{m-1+i} = m - x$, the pattern string $P$ with an arbitrary matching pattern has occurrences at $S_i$

## Image processing?

> Normal people should use [FFTW](https://www.fftw.org/) - but alas you are an ACM player.

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
        ll nchn = img.size(), h = img[0].size(), w = img[0][0].size();
        return { nchn, h, w };
    }
    // Assuming 8bit sRGB space
    template<typename Texel> Image from_texels(const Texel* img_data, int w, int h, int nchn) {
        Image chns(nchn, Poly::RVec2(h, Poly::RVec(w)));
        for (ll y = 0; y < h; ++y)
            for (ll x = 0; x < w; ++x)
                for (ll c = 0; c < nchn; ++c)
                    chns[c][y][x] = img_data[(y * w + x) * nchn + c];
        return chns;
    }
    vector<Texel> to_texels(const Image& res, int& w, int& h, int& nchn) {
        std::tie(nchn, h, w) = image_size(res);
        vector<Texel> texels(w * h * nchn);
        for (ll y = 0; y < h; ++y)
            for (ll x = 0; x < w; ++x)
                for (ll c = 0; c < nchn; ++c) {
                    ll t = std::round(res[c][y][x]);
                    texels[(y * w + x) * nchn + c] = max(min(255ll, t),0ll);
                }
        return texels;
    }
    inline Image from_file(const char* filename, bool hdr=false) {
        int w, h, nchn;
        Texel* img_data = stbi_load(filename, &w, &h, &nchn, 0);
        assert(img_data && "cannot load image");
        auto chns = from_texels(img_data, w, h, nchn);
        stbi_image_free(img_data);
        return chns;
    }
    inline void to_file(const Image& res, const char* filename, bool hdr=false) {
        int w, h, nchn;
        auto texels = to_texels(res, w, h, nchn);
        int success = stbi_write_png(filename, w, h, nchn, texels.data(), w * nchn);
        assert(success && "image data failed to save!");
    }
    inline Image create(int nchn, int h, int w, lf fill){
        Image image(nchn);
        for (auto& ch : image)
            Poly::utils::resize(ch, {h,w}, fill);
        return image;
    }
    inline Poly::RVec2& to_grayscale(Image& image) {
        auto [nchn, h, w] = image_size(image);
        auto& ch0 = image[0];
        // L = R * 299/1000 + G * 587/1000 + B * 114/1000
        for (ll c = 0;c <= 2;c++) {
            for (ll i = 0;i < h;i++) {
                for (ll j = 0;j < w;j++) {
                    if (c == 0) ch0[i][j] *= 0.299;
                    if (c == 1) ch0[i][j] += image[1][i][j] * 0.587;
                    if (c == 2) ch0[i][j] += image[2][i][j] * 0.144;
                }
            }
        }
        return ch0;
    }
}
```

### two-dimensional envelope (math.)

> Want to play around with a mega kernel and not wait half a year?

- Let the original image $A[N,M]$, and the envelope kernel $B[K,L]$ space to perform the envelope has time complexity $O(N * M * K * L)$
- Using $\text{FFT}$ it is $O(N * M * log(N * M))$

#### Gaussian blur

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

- test sample

  | importation                                                  | exports                                                      |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![input](/image-github/434868266-52c8860a-c118-406c-9ef1-2211b9e5ecc9.png) | ![output](/image-github/434868287-7f7bfe51-db49-4295-ab3a-76751c395c1b.png) |

#### Wiener deconvolution (inverse envelope)

> 2025, Codeforces 4.1 H question see

- https://en.wikipedia.org/wiki/Wiener_deconvolution
- Wiener deconvolution can be expressed as

$$
\ F(f) = \frac{H^\star(f)}{ |H(f)|^2 + N(f) }G(f)= \frac{H^\star(f)}{ H(f)\times H^\star(f) + N(f) }G(f)
$$

- are in the frequency domain, where $F$ is the original image, $G$ is the post-envelope image, $H$ is the convolution kernel, and $N$ is the noise function

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
        // Window required
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
- test sample

  | importation                                                  | exports                                                      |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![output](/image-github/435036278-13695e56-aa4e-4352-a90d-07ca14620479.png) | ![deblur](/image-github/435036293-38ad63d7-a12a-4032-8d08-3fd7e872d752.png) |



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

## 691C. Row GCD

> You are given two positive integer sequences $a_1, \ldots, a_n$ and $b_1, \ldots, b_m$. For each $j = 1, \ldots, m$ find the greatest common divisor of $a_1 + b_j, \ldots, a_n + b_j$.

- **Lemma:** $gcd(x,y) = gcd(x,y-x)$
- ** Lemma: ** can be extended to ** the array $gcd$ is numerically equal to the array difference $gcd$ **; the proof is obvious, omitted
  - Note that the proposition does not hold on **arrays and subarrays taken on their differences**, as in 991F.
    - How does Typora quickly add in-page links...

- Remember that $g_{pfx} = gcd(a_2-a_1,a_3-a_2,.... .a_n-a_{n-1})$
- In this problem use $gcd(a_1+b_1,a_2+b_1,... .a_n+b_1) = gcd(a_1+b1,a_2-a_1,a_3-a_2,... .a_n-a_{n-1}) = gcd(a_1 + b_1, g_{pfx})$ is sufficient

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

# 991F. Maximum modulo equality

>You are given an array $a$ of length $n$ and $q$ queries $l$, $r$.
>For each query, find the maximum possible $m$, such that all elements $a_l$, $a_{l+1}$, ..., $a_r$ are equal modulo $m$. In other words, $a_l \bmod m = a_{l+1} \bmod m = \dots = a_r \bmod m$, where $a \bmod b$ — is the remainder of division $a$ by $b$. In particular, when $m$ can be infinite, print $0$.

- **Lemma:** Equivalence in the sense of mod $m$ ($x \mod m = y \mod m$) $\iff$ $|x-y| \mod m = 0$
- 故本题$a_l \bmod m = a_{l+1} \bmod m = \dots = a_r \bmod m \iff |a_{l+1} - a_{l}| \mod m = |a_{l+2} - a_{l}| \mod m = ... = |a_{r} - a_{r-1}| \mod m = 0$
- It is clear that the largest $m$ here is the $gcd$ of the difference array.
- Just process the query to realize $gcd$ RMQ; note that the boundary by (2) should be $[l+1,r]$; the $l=r$ case is $m$ desirable $\inf$.

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

## P11373 “CZOI-R2” Balance

> You have $n$ **weight sets** numbered $1$ through $n$. For the $i$th **weight group** the weights in the **weight group** have a common positive integer mass $a_i$, and there are an infinite number of **weights** in each **weight group**.
> where there are $q$ operations:
>
> - `I x v`: a new set of individual **weights** with a mass of $v$ is added after the $x$th **weight set**, when $x=0$ it means that it is added at the top;
> - `D x`: delete the $x$th **weight set**;
> - `A l r v`: add $v$ to the mass of the weights in all **weight sets** from $l$ to $r$;
> - `Q l r v`: Determine if the mass $v$ can be weighed using weights from the **set of weights** from $l$ to $r$. Any number of weights in each weight set may or may not be used.
>   For operations `I` and `D`, the number and the value of $n$ change automatically after the operation.
>   Weighing some **weights** can result in a mass $v$, if and only if there exists a method of placing these weights on each side of the balance such that placing $1$ objects with mass $v$ on one side balances the balance.

- **Lemma:** **Bézout's identity** (English: Bézout's identity), or **Thoupantu's equation for one special case**: Let $a_1, \cdots a_n$ be $n$ integers, and $d$ be their greatest common divisor, then there exist integers $x_1, \cdots x_n$ such that $x_1\cdot a_1 + \ cdots x_n\cdot a_n = d$

- For operation `Q`, which asks whether the integers $x_i$ in $l,r$ can form $x_1\cdot a_1 + \cdots x_n\cdot a_n = kd = v \to v \mod gcd(a_1,... ,a_n) = 0 $

- Maintain a balanced tree/treap for the operation `I,D,A`.
  - The idea is basically the same as [Line Tree Subtask 3](https://mos9527.com/posts/cp/segment-tree-problems/#p11373-czoi-r2%E5%A4%A9%E5%B9%B3)
  - Additional considerations for maintenance of **addition, deletion**; simple manipulation of neighboring differentials is sufficient
  - Since it's a single point of modification, again there's no need for `push_down` to pass the lazy marker
    - Logu b review why doesn't it show compilation warnings ==; `insert` didn't `return` directly RTE for countless hair...


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

**Note:** `segment_tree` are accessed using `1-Index`; `vector` is `0-Index` in `segment_tree::reset(vector&)`.

## 242E. XOR on Segment

Interval binary change + lazy pass + binary trick

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

Number theory, single point change + pruning

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

String conversion `bitset` to find the number of unique values

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

## P6492. STEP

> Given a sequence of characters $a$ of length $n$, initially the sequence contains all characters `L`.
>
> There are $q$ modifications, and each time, given an $x$, if $a_x$ is `L`, then $a_x$ is modified to `R`, otherwise $a_x$ is modified to `L`.
>
> For a string $s$ containing only the characters `L`, `R`, $s$ is said to be satisfied if there are no consecutive `L` and `R` in it.
>
> After each modification, output the length of the longest consecutive substring in the current sequence $a$ that satisfies the requirement.

```C++
int len[4*N],L[4*N],R[4*N],S[4*N],H[4*N],ans[4*N];
// 原数组，节点长度，左端点，右端点，符合条件的前缀，符合条件的后缀，符合条件的最大长度

void work(int o,int k){//更新第o个节点 
	S[o]=H[o]=ans[o]=1;
	L[o]=R[o]=k;
}

void maintain(int o){
	int lc=o<<1,rc=o<<1|1;
	if(L[rc]^R[lc]==0){
		ans[o]=max(ans[lc],ans[rc]);
	}
	else{
		ans[o]=max(H[lc]+S[rc],max(ans[lc],ans[rc]));
	}
	L[o]=L[lc],R[o]=R[rc];
	if(S[lc]==len[lc]&&L[rc]^R[lc])S[o]=S[lc]+S[rc];
	else S[o]=S[lc];
	if(H[rc]==len[rc]&&L[rc]^R[lc])H[o]=H[rc]+H[lc];
	else H[o]=H[rc];
} 

void build(int o,int l,int r){
	len[o]=r-l+1;
	if(l==r){
		work(o,0);
		return;
	}
	int lc=o*2,rc=o*2+1,mid=l+r>>1;
	build(lc,l,mid);
	build(rc,mid+1,r);
	maintain(o);
}

void change(int o,int l,int r,int x)
{
	if(l==r)                                
	{
		work(o,!L[o]);                //0变成1,1变成0
		return;
	}
	int lc=o*2,rc=o*2+1,mid=l+r>>1;
	if(x<=mid) change(lc,l,mid,x);
	else change(rc,mid+1,r,x);
	maintain(o);
}

int main(){
	int n,q;
	cin>>n>>q;
	build(1,1,n);
	while(q--){
		int x;cin>>x;
		change(1,1,n,x);
		cout<<ans[1]<<endl;
	}
	return 0;
}


```

## P11373 「CZOI-R2」scales

- Positive solution to https://mos9527.com/posts/cp/gcd-problems/#p11373-czoi-r2%E5%A4%A9%E5%B9%B3, here is the Subtask 3 solution
- TL; DR interval maintenance $gcd$; also **interval change** operation to **single point change** operation omitting `push_down`
  - Given an array $a$ define $gcd(a) = gcd(a_1,a_2,.... .a_n)$
  - By [引理](https://mos9527.com/posts/cp/gcd-problems/#691c-row-gcd) $gcd(x,y) = gcd(x,y-x)$, which can be expanded to $gcd(a) = gcd(a_1, a_2 - a_1, ... , a_n - a_{n-1})$
  - Remember that the difference array is $b$,$\forall b_i \in b, b_i = a_i - a_{i-1}$, with both $gcd(a) = gcd(a_1, b_2, ... ,b_n)$
  - Given that the topic only requires **interval plus **, i.e., it is equivalent to **differential array single-point change**, `push_up` is sufficient after maintaining the $b$-array RMQ

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

  

## Reference

- C++ style implementation

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
        tree[o].max_v = max_v(tree[lc].max_v, tree[rc].max_v);
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
    // src -> 0开始的叶子节点
    void reset(const vector<T>& src) {
        end = src.size(); tree.resize(end << 2);
        build(src.data() - 1); // 邪恶指针trick - 毕竟我们的访问从1开始
    }
    explicit segment_tree() {};
    explicit segment_tree(const ll n) : begin(1), end(n) { reset(n); }
};
```

## RMQ General

- https://codeforces.com/contest/2014/submission/282795544 （D，Zone Change + Single Point Inquiry and）
- https://codeforces.com/contest/339/submission/282875335 （D，Single-point change + zone query）
