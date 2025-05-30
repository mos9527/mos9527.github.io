---
author: mos9527
lastmod: 2025-05-30T08:50:48.135148
title: 算竞笔记 - GCD及相关专题
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++","GCD","数学"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

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
For each query, find the maximum possible $m$, such that all elements $a_l$, $a_{l+1}$, ..., $a_r$ are equal modulo $m$. In other words, $a_l \bmod m = a_{l+1} \bmod m = \dots = a_r \bmod m$, where $a \bmod b$ — is the remainder of division $a$ by $b$. In particular, when $m$ can be infinite, print $0$.

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
其中，有 $q$ 次操作：
>- `I x v`：在第 $x$ 个**砝码组**后新增一组单个**砝码**质量为 $v$ 的**砝码组**，当 $x=0$ 时表示在最前面新增；
>- `D x`：删除第 $x$ 个**砝码组**；
>- `A l r v`：把从 $l$ 到 $r$ 的所有**砝码组**中的砝码质量加 $v$；
>- `Q l r v`：判断能否用从 $l$ 到 $r$ 的**砝码组**中的砝码，称出质量 $v$。每个砝码组中的砝码可以使用任意个，也可以不用。
对于操作 `I` 和 `D`，操作后编号以及 $n$ 的值自动变化。
称一些**砝码**可以称出质量 $v$，当且仅当存在将这些砝码分别放在天平两边的摆放方法，使得将 $1$ 个质量为 $v$ 的物体摆放在某边可以让天平平衡。

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

  
