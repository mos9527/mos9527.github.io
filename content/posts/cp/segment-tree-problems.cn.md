---
author: mos9527
lastmod: 2025-05-14T10:08:15.031000+08:00
title: 算竞笔记 - 线段树专题
tags: ["线段树",ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

**注:** `segment_tree` 均采用 `1-Index` 访问； `segment_tree::reset(vector&)` 中`vector`为`0-Index`
## 区间延迟（Lazy）修改模版

- C++ 风格实现

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
        tree[o].max_v = max(tree[lc].max_v, tree[rc].max_v);
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
    // 注意：src[0]会被省略
    void reset(const vector<T>& src) {
        end = src.size() - 1; tree.resize(end << 2);
        build(src.data());
    }
    explicit segment_tree() {};
    explicit segment_tree(const ll n) : begin(1), end(n) { reset(n); }
};
```

- https://codeforces.com/contest/2014/submission/282795544 （D，区间改+单点查询和）
- https://codeforces.com/contest/339/submission/282875335 （D，单点改+区间查询）

## 可持久化线段树（主席树）

- https://zhuanlan.zhihu.com/p/762284607
- https://ac.nowcoder.com/acm/contest/91177/F （找第$k$小）
- https://www.luogu.com.cn/problem/P3834

```c++
template <typename T> struct segment_tree {
    constexpr static ll root = 1; // 根节点编号
    ll node_id = 1; // 当前最新节点编号
public:
    struct node {
        ll lc, rc; // 左右子节点**编号**；非区间
        ll l, r; // 区间
        T sum{};
    };
    vector<node> tree;
    // 向上传递
    void push_up(ll o) {
        tree[o].sum = tree[tree[o].lc].sum + tree[tree[o].rc].sum;
    }
    // 初始版本
    void build(ll o, ll l, ll r) {
        if (l == r) return;
        ll mid = (l + r) / 2;
        ll lc = tree[o].lc = ++node_id, rc = tree[o].rc = ++node_id;
        tree[o].l = l, tree[o].r = r;
        build(lc, l, mid);
        build(rc, mid + 1, r);
        push_up(o);
    }
    void update(ll pos, ll l, ll r, ll prev /*旧版本复制源点*/, ll curr /*新版本新建点*/, T v) {
        ll mid = (l + r) / 2;
        if (l == r) {
            // 到达叶子点
            // 修改只在新点及剪出来的枝上体现
            tree[curr].sum = tree[prev].sum + v;
        } else {
            // 到叶子点路上；默认复用
            tree[curr] = tree[prev];
            if (pos <= mid) {
                // 新点会在左子树开，途径有必要持久化（复制）
                // 每个点都要开新点
                tree[curr].lc = ++node_id;
                update(pos, l, mid, tree[prev].lc, tree[curr].lc, v);
            } else {
                // 右子树 - 同上，交换左右
                tree[curr].rc = ++node_id;
                update(pos, mid + 1, r, tree[prev].rc, tree[curr].rc, v);
            }
            push_up(curr);
        }
    }
    explicit segment_tree(ll n) : tree(n) {};
};
segment_tree<ll> seg(DIM);
// 树上二分找[l,r]区间第k小
int query_kth(ll l, ll r, ll prev /*旧版本同位置点*/, ll curr /*新版本同位置点*/, ll kth_small) {
    if (l == r) return l;
    ll mid = (l + r) / 2;
    // 我们的每一个版本（根节点上点）线段树存的为*权值*（或直方图的高度，即数字的数目）
    // 找第k小即为找*离散化后*数x对应 \sum_{i=1}^{x} tree[i].sum < kth_small 的上限
    // 在[l,r]区间内找，可以看成是*两个*版本树的差分
    // 区间内的数目即为：
    ll d = seg.tree[seg.tree[curr].lc].sum - seg.tree[seg.tree[prev].lc].sum;
    // 树上二分
    if (d < kth_small) {
        // x更大在右子树
        // 在右边找；注意左区间数*不能*统计
        return query_kth(mid + 1, r, seg.tree[prev].rc, seg.tree[curr].rc, kth_small - d);
    } else {
        // x更小在左子树
        return query_kth(l, mid, seg.tree[prev].lc, seg.tree[curr].lc, kth_small);
    }
}
int main() {
    fast_io();
    /* El Psy Kongroo */
    ll n, q; cin >> n >> q;
    vec a(n + 1), mp;
    for (ll i = 1; i<=n;i++)
        cin >> a[i], mp.push_back(a[i]);
    sort(mp.begin(), mp.end());
    mp.erase(unique(mp.begin(), mp.end()), mp.end());

    ll m = mp.size(); // 离散化后位置i对应数字
    seg.build(seg.root, 1, m);
    vec roots(n+1, seg.root);
    for (ll i = 1; i<=n;i++) {
        // 新版本
        roots[i] = ++seg.node_id;
        // 从上一个版本转移；这里在mp[i]上多一个数
        ll pos = lower_bound(mp.begin(), mp.end(), a[i]) - mp.begin() + 1;
        seg.update(pos, 1, m, roots[i - 1], roots[i], 1);
    }
    while (q--) {
        ll l,r; cin >> l >> r;
        ll mid = (r - l + 2) / 2; // \ceil
        // 注意我们求的是*上限*
        ll pos = query_kth(1, m, roots[l - 1], roots[r], mid);
        cout << mp[pos - 1] << endl;
    }
    return 0;
}
```

## 242E. XOR on Segment

区间二进制改+lazy传递+二进制trick

> You've got an array $a$, consisting of $n$ integers $a_1, a_2, ..., a_n$. You are allowed to perform two operations on this array:
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

数论、单点改+剪枝

>Let $D(x)$ be the number of positive divisors of a positive integer $x$. For example, $D(2)= 2$ (2 is divisible by 1 and 2), $D(6) = 4$ (6 is divisible by 1, 2, 3 and 6).
You are given an array $a$ of $n$ integers. You have to process two types of queries:
>1. `REPLACE` $l,r$ - for every $i \in [l,r]$, replace $a_i$ with $D(a_i)$
>2. `SUM` $l,r$ - calculate $\sum_{i=l}^{r}{a_i}$
Print the answer for each `SUM` query.

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

串转换`bitset`求独特值数

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

## P11373 「CZOI-R2」天平
- 正解转 https://mos9527.com/posts/cp/gcd-problems/#p11373-czoi-r2%E5%A4%A9%E5%B9%B3，此处为Subtask 3解法
- TL；DR 区间维护$gcd$；同时将**区间改**操作化为**单点改**操作省去`push_down`
  - 给定数组$a$定义$gcd(a) = gcd(a_1,a_2,...a_n)$
  - 由[引理](https://mos9527.com/posts/cp/gcd-problems/#691c-row-gcd)知$gcd(x,y) = gcd(x,y-x)$，可拓展为$gcd(a) = gcd(a_1, a_2 - a_1, ..., a_n - a_{n-1})$
  - 记差分数组为$b$,$\forall b_i \in b, b_i = a_i - a_{i-1}$,既有$gcd(a) = gcd(a_1, b_2,...,b_n)$
  - 鉴于题目只要求**区间加**，即等效于**差分数组单点改**，维护$b$数组RMQ后`push_up`即可
  
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

  

