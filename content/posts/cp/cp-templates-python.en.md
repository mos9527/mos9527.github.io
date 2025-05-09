---
author: mos9527
lastmod: 2025-05-09T19:37:44.543273
title: Competitive Programming - Algorithm Templates And Problem Sets (Python)
tags: ["ACM","Competeive Programming","XCPC","(Code) Templates","Problem Solutions","Codeforces","Python"]
categories: ["Problem Solutions", "Competeive Programming", "Collection/compilation"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Preface

- The following Code is a direct port of [Cpp version](https://mos9527.github.io/posts/%E7%AE%97%E6%B3%95%E7%AB%9E%E8%B5%9B/cp-templates/); please consider the former as the upstream and first reference source
- Python implementation, generally **preferring** to consider dynamic memory opening and **hashing** lookups (e.g. `defaultdict` instead of linear storage)
- Code section is not categorized, please make good use of [Cpp version](https://mos9527.github.io/posts/%E7%AE%97%E6%B3%95%E7%AB%9E%E8%B5%9B/cp-templates/) + Ctrl-F.

# Header

```python
from collections import defaultdict
```

# graph theory

## receiver gauge

```python
class graph(defaultdict):
    def __init__(self):
        super().__init__(list)

    def add_edge(self, u, v):
        self[u].append(v)
```

## DSU

```python
class dsu(dict):
    def __getitem__(self, key):
        if not key in self:
            super().__setitem__(key, key)
        return super().__getitem__(key)

    def find(self, u):
        if self[u] != u:
            return self.find(self[u])
        return self[u]

    def join(self, u, v):
        self[self.find(u)] = self.find(v)

    def same(self, u, v):
        return self.find(u) == self.find(v)

```

## HLD

```python
class HLD:
    def __init__(self, g: graph = None):
        self.dfn_cnt = 0
        self.sizes = defaultdict(int)
        self.depth = defaultdict(int)
        self.top = defaultdict(int)
        self.parent = defaultdict(int)
        self.dfn = defaultdict(int)
        self.dfn_out = defaultdict(int)
        self.heavy = defaultdict(int)
        self.inv_dfn = list()
        self.G = g if g is not None else graph()

    def __dfs1(self, u):
        self.heavy[u] = -1
        self.sizes[u] = 1
        for v in self.G[u]:
            if self.depth[v]:
                continue
            self.depth[v] = self.depth[u] + 1
            self.parent[v] = u
            self.__dfs1(v)
            self.sizes[u] += self.sizes[v]
            if self.heavy[u] == -1 or self.sizes[v] > self.sizes[self.heavy[u]]:
                self.heavy[u] = v

    def __dfs2(self, u, v_top):
        self.top[u] = v_top
        self.dfn[u] = self.dfn_cnt + 1
        while len(self.inv_dfn) <= self.dfn[u]:
            self.inv_dfn.append(0)
        self.inv_dfn[self.dfn[u]] = u
        self.dfn_cnt += 1
        if self.heavy[u] != -1:
            self.__dfs2(self.heavy[u], v_top)
            for v in self.G[u]:
                if v != self.heavy[u] and v != self.parent[u]:
                    self.__dfs2(v, v)

        self.dfn_out[u] = self.dfn_cnt

    def add_edge(self, u, v):
        self.G.add_edge(u, v)
        self.G.add_edge(v, u)

    def prep(self, root):
        self.depth[root] = 1
        self.__dfs1(root)
        self.__dfs2(root, root)

    def lca(self, u, v):
        """lowest common ancestor"""
        while self.top[u] != self.top[v]:
            if self.depth[self.top[u]] < self.depth[self.top[v]]:
                u, v = v, u
            u = self.parent[self.top[u]]
        return u if self.depth[u] < self.depth[v] else v

    def lca_multi(self, a, b, c):
        """lca(a, b) ^ lca(b, c) ^ lca(c, a)"""
        return self.lca(a, b) ^ self.lca(b, c) ^ self.lca(c, a)

    def dist(self, u, v):
        """distance between u and v"""
        return self.depth[u] + self.depth[v] - 2 * self.depth[self.lca(u, v)] + 1

    def path_sum(self, u, v, query):
        """query: callable(dfn_l, dfn_r)"""
        while self.top[u] != self.top[v]:
            if self.depth[self.top[u]] < self.depth[self.top[v]]:
                u, v = v, u
            query(self.dfn[self.top[u]], self.dfn[u])
            u = self.parent[self.top[u]]

        if self.dfn[v] > self.dfn[u]:
            u, v = v, u
        query(self.dfn[v], self.dfn[u])

    def subtree(self, u):
        """iterable of the subtree of u"""
        return (self.inv_dfn[i] for i in range(self.dfn[u], self.dfn_out[u] + 1))

    def is_child_of(self, u, v):
        """v is a child of/part of the sub tree of u"""
        return self.dfn[u] <= self.dfn[v] <= self.dfn_out[u]

```

