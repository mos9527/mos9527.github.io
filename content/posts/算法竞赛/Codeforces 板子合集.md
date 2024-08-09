---
author: mos9527
lastmod: 2077-01-01T12:00:00.000000+08:00
title: Codeforces 板子合集
tags: ["ACM","算竞","Codeforces"]
categories: ["合集","杂项"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---
## Header
```c++
#pragma GCC optimize("O3","unroll-loops")
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <queue>
#include <iomanip>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <set>
#include <array>
#include <random>
#include <cassert>

using namespace std;
typedef double lf;
typedef long long ll;
typedef unsigned long long ull;
typedef vector<ll> v;
typedef vector<ull> uv;
#define EPS 0.0001F
#define PRED(X) [](auto const& lhs, auto const& rhs) {return X;}
#define PREDT(T,X) [](T const& lhs, T const& rhs) {return X;}
#define PAIR2(T) pair<T,T>
typedef PAIR2(ll) II;
typedef PAIR2(ull) uII;
#define DIMENSION 3e5+1
#define BIT(X) (1LL << X)
#define LOWBIT(X) (X & (-X))
#define DIM (size_t)(DIMENSION)
#define MOD (ll)(1e9 + 7)
#ifdef _DEBUG
#define LOG cerr
#define REDIRECT_IN assert(std::freopen("input.in", "r", stdin));
#define REDIRECT_OUT assert(std::freopen("input.out", "r", stdout));
#else
#define LOG if (false) cerr
#define REDIRECT_IN
#define REDIRECT_OUT
#endif // _DEBUG

int main() {
    std::ios::sync_with_stdio(false); std::cin.tie(0); std::cout.tie(0);
    /* El Psy Kongroo */
    return 0;
}
```
# 数学
## 矩阵

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

- https://codeforces.com/gym/105170/problem/C

## 快速幂

```c++
// 注：爆int64考虑__int128或相关intrinsic
// MSVC: https://codeforces.com/blog/entry/106396
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
		if (b & 1) res = res * a % m;
		a = a * a % m;
		b >>= 1;
	}
	return res;
}
```

## 组合数（递推）

```c++
ll C[DIM][DIM];
for (ll i = 0; i < DIM; i++)
    for (ll j = 0; j <= i; j++)
        C[i][j] = j ? ((C[i - 1][j] + C[i - 1][j - 1]) % MOD) : 1;
```

- Lucas：$$\binom{n}{m}\bmod p = \binom{\left\lfloor n/p \right\rfloor}{\left\lfloor m/p\right\rfloor}\cdot\binom{n\bmod p}{m\bmod p}\bmod p$$​

## 乘法逆元
- https://acm.hdu.edu.cn/showproblem.php?pid=7437

给定质数$m$,求$a$的逆元$a^{-1}$​

- 欧拉定理知 $a^{\phi (m)} \equiv 1 \mod m$
- 对质数 $m$, $\phi (m) = m - 1$
- 此情景即为费马小定理，i.e. $a^{m - 1} \equiv 1 \mod m$
- 左右同时乘$a^{-1}$,可得 $a ^ {m - 2} \equiv a ^ {-1} \mod m$
- 即 `a_inv = binpow_mod(a, m - 2, m)`

## 数论

### Euler 筛

```c++
namespace euler_sieve { // 欧拉筛法 + 区间筛
    v primes;
    bool not_prime[DIM];

    void init(ll N) {
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
            if (p * p > r) break;
            for (ll j = max((ll)ceil(1.0 * l / p), p) * p; j <= r; j += p) not_prime[j] = true;
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
    for (ll a : {2, 3, 5, 7, 11, 13, 17, 19, 23}) {
        if (p == a) return 1;
        ll x = binpow_mod(a, d, p);
        if (x == 1 || x == p - 1) continue;
        for (int i = 0; i < r - 1; ++i) {
            x = x * x % p;
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
    ll s = 0, t = 0;
    ll c = ll(rng()) % (x - 1) + 1;
    ll val = 1;
    for (ll goal = 1; ; goal *= 2, s = t, val = 1) {
        for (ll step = 1; step <= goal; step++) {
            t = (t * t + c) % x;
            val = val * abs(t - s) % x;
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
	void Prime_Factor(ll x, v& res) {
        for (auto p : primes) {
            if (p * p > x) break;
            if (x % p == 0) {
                res.push_back(p);
                while (x % p == 0) x /= p;
            }
            if (x != 1) res.push_back(x);
        }
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
struct graph {
	bool G[1001][1001]{};
	v in;
	v L;
	ll n;

	v dp;

	graph(ll dimension) : n(dimension), in(dimension + 1), dp(dimension + 1) {};
	void add_edge(ll from, ll to) {
		if (!G[from][to]) {
			G[from][to] = 1;
			in[to]++;
		}
	}
	bool topsort() {
		ll cnt = 0;
		queue<ll> S;
		ll ans = 0;
		for (ll i = 1; i <= n; i++) {
			if (in[i] == 0) S.push(i), dp[i] = 1;
		}
		while (!S.empty()) {
			ll v = S.front(); S.pop();
			cnt++;
			for (ll out = 1; out <= n; out++) {
				if (G[v][out]) {
					if (--in[out] == 0)
						S.push(out);
					ll dist = (dp[v] + 1);
					dp[out] = max(dp[out], dist);
				}
			}
		}
		return true; // ((cnt == n) ? true : false);
	}

};
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
		const auto pp = PREDT(vert, lhs.dis > rhs.dis);
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

```c++
struct edge { ll to, cost; };
struct graph {
	ll n;

	vector<vector<edge>> G;

	vector<v> fa;
	v depth, dis;

	graph(ll n) : n(n), fa(ceil(log2(n)) + 1, v(n)), depth(n), G(n), dis(n) {}

	void add_edge(ll from, ll to, ll cost = 1) {
		G[from].push_back({ to, cost });
	}

	void lca_prep(ll root) {
		auto dfs = [&](ll u, ll pa, ll dep, auto& dfs) -> void {
			fa[0][u] = pa, depth[u] = dep;
			for (ll i = 1; i < fa.size(); i++) {
				// u 的第 2^i 的祖先是 u 第 (2^(i-1)) 个祖先的第 (2^(i-1)) 个祖先
				fa[i][u] = fa[i - 1][fa[i - 1][u]];
			}
			for (auto& e : G[u]) {
				if (e.to == pa) continue;
				dis[e.to] = dis[u] + e.cost;
				dfs(e.to, u, dep + 1, dfs);
			}
		};
		dfs(root, root, 0, dfs);
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
};
```

## 树的直径

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

# 数据结构 / DS
## 优先队列（二叉堆）

> ```c++
> #define PREDT(T,X) [](T const& lhs, T const& rhs) {return X;}    
> auto pp = PREDT( elem, lhs.w > rhs.w);
> priority_queue < elem, vector<elem>, decltype(pp)> Q {pp};
> ```

## DSU

- 不考虑边权

```C++
struct dsu {
    vector<ll> pa;
    dsu(const ll size) : pa(size) { iota(pa.begin(), pa.end(), 0); }; // 初始时，每个集合都是自己的父亲
    inline bool is_root(const ll leaf) { return pa[leaf] == leaf; }
    inline ll find(const ll leaf) { return is_root(leaf) ? leaf : find(pa[leaf]); } // 路径压缩
    inline void unite(const ll x, const ll y) { pa[find(x)] = find(y); }
};
```

- 需要计算到根距离

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
## 树状数组
```c++
struct fenwick : public v {
	using v::v;
	void init(v const& a) {
		for (ll i = 0; i < a.size(); i++) {
			(*this)[i] += a[i]; // 求出该子节点
			ll j = i + LOWBIT(i);
			if (j < size()) (*this)[j] += (*this)[i]; // ...后更新父节点
		}
	}
	// \sum_{i=1}^{n} a_i
	ll sum(ll n) {
		ll s = 0;
		for (; n; n -= LOWBIT(n)) s += (*this)[n];
		return s;
	};
	ll query(ll l, ll r) {
		return sum(r) - sum(l - 1);
	}
	void add(ll n, ll k) {
		for (; n < size(); n += LOWBIT(n)) (*this)[n] += k;
	};
};
```
### 支持不可差分查询模板

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

### 区间模板

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

