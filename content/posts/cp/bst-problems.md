---
author: mos9527
lastmod: 2024-12-10T08:46:17.203000+08:00
title: 算竞笔记：平衡树相关专题
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++","GCD","数学","DS","data structures"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Treap

**ATTENTION:** WIP. *没调完呢（* 

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

提供**删除**操作；不支持查找；支持RMQ

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

