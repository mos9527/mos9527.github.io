---
author: mos9527
lastmod: 2025-04-18T12:15:44.873000+08:00
title: Computing Notes - GCD and Related Topics
tags: ["ACM","Competeive Programming","XCPC","(Code) Templates","Problem sets","Codeforces","C++","GCD","Mathematics"]
categories: ["Problem Solutions", "Competeive Programming", "Collection"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

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
For each query, find the maximum possible $m$, such that all elements $a_l$, $a_{l+1}$, ..., $a_r$ are equal modulo $m$. In other words, $a_l \bmod m = a_{l+1} \bmod m = \dots = a_r \bmod m$, where $a \bmod b$ — is the remainder of division $a$ by $b$. In particular, when $m$ can be infinite, print $0$.

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
> For operations `I` and `D`, the number and the value of $n$ change automatically after the operation.
> Weighing some **weights** can result in a mass $v$, if and only if there exists a method of placing these weights on each side of the balance such that placing $1$ objects with mass $v$ on one side balances the balance.

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

