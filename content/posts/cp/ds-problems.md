---
author: mos9527
lastmod: 2024-11-16T22:29:05.808000+08:00
title: 算竞笔记：数据结构相关专题
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++","GCD","数学","DS","data structures"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Treap 

**WIP**; 又名**笛卡尔树**，**随机BST**；支持$log n$插入，删除，查找及闭包操作(`push_up`)

- https://cp-algorithms.com/data_structures/treap.html
- https://oi.baoshuo.ren/fhq-treap

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
    II split(ll o, T const& key) { // 偏序 [<, >=]
        if (!o) return {0,0};
        if (tree[o].key < key) { // 左大右小
            auto [ll,rr] = split(tree[o].r, key);
            tree[o].r = ll;
            push_up(o);
            return {o,rr};
        } else {
            auto [ll,rr] = split(tree[o].l, key);
            tree[o].l = rr;
            push_up(o);
            return {ll,o};
        }
    }
    ll merge(ll l, ll r) {
        if (!l || !r) return l + r;
        if (tree[l].priority > tree[r].priority) // 保持堆序; 优先级大的在上
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
        auto [l,r] = split(o, key);
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

