---
author: mos9527
lastmod: 2024-11-08T11:05:50.210000+08:00
title: 算竞笔记：DP类型专题
tags: ["DP","ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

**注:** WIP. 主要是OIWiki搬砖

**TODO:** 添加题目

# 基础

>https://oi-wiki.org/dp/basic

## LCS （最长公共子序列）

设 $f(i,j)$ 表示只考虑 $A$ 的前 $i$ 个元素，$B$ 的前 $j$ 个元素时的最长公共子序列的长度

- 转移方程

$$
f(i,j)=\begin{cases}f(i-1,j-1)+1&A_i=B_j\\\max(f(i-1,j),f(i,j-1))&A_i\ne B_j\end{cases}
$$

- 实现

```c++
int a[MAXN], b[MAXM], f[MAXN][MAXM];

int dp() {
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= m; j++)
      if (a[i] == b[j])
        f[i][j] = f[i - 1][j - 1] + 1;
      else
        f[i][j] = std::max(f[i - 1][j], f[i][j - 1]);
  return f[n][m];
}
```

## LIS（最长不下降子序列）

> 设 $f(i)$ 表示以 $A_i$ 为结尾的最长不下降子序列的长度，则所求为 $\max_{1 \leq i \leq n} f(i)$。计算 $f(i)$ 时，尝试将 $A_i$ 接到其他的最长不下降子序列后面，以更新答案。

- 转移方程

$$
f(i)=\max_{1 \leq j < i, A_j \leq A_i} (f(j)+1)
$$

- 实现

```c++
int a[MAXN], d[MAXN];

int dp() {
  d[1] = 1;
  int ans = 1;
  for (int i = 2; i <= n; i++) {
    d[i] = 1;
    for (int j = 1; j < i; j++)
      if (a[j] <= a[i]) {
        d[i] = max(d[i], d[j] + 1);
        ans = max(ans, d[i]);
      }
  }
  return ans;
}
```
- 优化

> 再看一下之前的转移：$(j, l - 1) \rightarrow (i, l)$，就可以判断某个 $(i, l)$ 是否合法。
初始时 $(1, 1)$ 肯定合法。
> 那么，只需要找到一个 $l$ 最大的合法的 $(i, l)$，就可以得到最终最长不下降子序列的长度了。
> 那么，根据上面的方法，我们就需要维护一个可能的转移列表，并逐个处理转移。

> 所以可以定义 $a_1 \dots a_n$ 为原始序列，$d_i$ 为所有的长度为 $i$ 的不下降子序列的末尾元素的最小值，$len$ 为子序列的长度。
> 初始化：$d_1=a_1,len=1$。
> 现在我们已知最长的不下降子序列长度为 1，那么我们让 $i$ 从 2 到 $n$ 循环，依次求出前 $i$ 个元素的最长不下降子序列的长度，循环的时候我们只需要维护好 $d$ 这个数组还有 $len$ 就可以了。**关键在于如何维护。**

> 考虑进来一个元素 $a_i$：
> 
> 1.  元素大于等于 $d_{len}$，直接将该元素插入到 $d$ 序列的末尾。
> 2.  元素小于 $d_{len}$，找到 **第一个** 大于它的元素，用 $a_i$ 替换它。

> -   对于步骤 1：
>     由于我们是从前往后扫，所以说当元素大于等于 $d_{len}$ 时一定会有一个不下降子序列使得这个不下降子序列的末项后面可以再接这个元素。如果 $d$ 不接这个元素，可以发现既不符合定义，又不是最优解。

- 实现
```c++
for (int i = 0; i < n; ++i) scanf("%d", a + i);
memset(dp, 0x1f, sizeof dp);
mx = dp[0];
for (int i = 0; i < n; ++i) {
  *std::upper_bound(dp, dp + n, a[i]) = a[i];
}
ans = 0;
while (dp[ans] != mx) ++ans;
```

> 对于最长 **上升** 子序列问题，类似地，可以令 $d_i$ 表示所有长度为 $i$ 的最长上升子序列的末尾元素的最小值。
> 需要注意的是，在步骤 2 中，若 $a_i \leq d_{len}$，由于最长上升子序列中相邻元素不能相等，需要在 $d$ 序列中找到 **第一个** **不小于** $a_i$ 的元素，用 $a_i$ 替换之。
在实现上（以 C++ 为例），需要将 `upper_bound` 函数改为 `lower_bound`。

# 背包

## 01背包

> https://oi.wiki/dp/knapsack/

设 DP 状态 $f_{i,j}$ 为在只能放前 $i$ 个物品的情况下，容量为 $j$ 的背包所能达到的最大总价值。

- 转移方程

$$
f_{i,j}=\max(f_{i-1,j},f_{i-1,j-w_{i}}+v_{i})
$$

- 滚动数组优化

$$
f_j=\max \left(f_j,f_{j-w_i}+v_i\right)
$$

- 实现

```cpp
for (int i = 1; i <= n; i++)
  for (int l = W; l >= w[i]; l--) f[l] = max(f[l], f[l - w[i]] + v[i]);
```

## 完全背包
> https://oi.wiki/dp/knapsack/

设 DP 状态 $f_{i,j}$ 为在只能放前 $i$ 个物品的情况下，容量为 $j$ 的背包所能达到的最大总价值。

- 转移方程

> 可以考虑一个朴素的做法：对于第 $i$ 件物品，枚举其选了多少个来转移。这样做的时间复杂度是 $O(n^3)$ 的。

> 状态转移方程如下：

$$
f_{i,j}=\max_{k=0}^{+\infty}(f_{i-1,j-k\times w_i}+v_i\times k)
$$

> 考虑做一个简单的优化。可以发现，对于 $f_{i,j}$，只要通过 $f_{i,j-w_i}$ 转移就可以了。因此状态转移方程为：

$$
f_{i,j}=\max(f_{i-1,j},f_{i,j-w_i}+v_i)
$$

-  滚动数组优化
$$
f_j=\max(f_j,f_{j-w_i}+v_i)
$$

- 实现

```cpp
for (int i = 1; i <= n; i++)
  for (int l = w[i]; l <= W; l++) f[l] = max(f[l], f[l - w[i]] + v[i]);
```

# 区间

令状态 $f(i,j)$ 表示将下标位置 $i$ 到 $j$ 的所有元素合并能获得的价值的最大值，那么 $f(i,j)=\max\{f(i,k)+f(k+1,j)+cost\}$，$cost$ 为将这两组元素合并起来的价值。

- 转移方程

$$
f(i,j)=\max\{f(i,k)+f(k+1,j)+\sum_{t=i}^{j} a_t \}~(i\le k<j)
$$
> 令 $sum_i$ 表示 $a$ 数组的前缀和，状态转移方程变形为

$$
f(i,j)=\max\{f(i,k)+f(k+1,j)+sum_j-sum_{i-1} \}
$$

> 由于计算 $f(i,j)$ 的值时需要知道所有 $f(i,k)$ 和 $f(k+1,j)$ 的值，而这两个中包含的元素的数量都小于 $f(i,j)$，所以我们以 $len=j-i+1$ 作为 DP 的阶段。首先从小到大枚举 $len$，然后枚举 $i$ 的值，根据 $len$ 和 $i$ 用公式计算出 $j$ 的值，然后枚举 $k$，时间复杂度为 $O(n^3)$

- 实现

```cpp
for (len = 2; len <= n; len++)
  for (i = 1; i <= n - len; i++) {
    int j = len + i - 1;
    for (k = i; k < j; k++)
      f[i][j] = max(f[i][j], f[i][k] + f[k + 1][j] + sum[j] - sum[i - 1]);
  }
```

# 状压

**注:** WIP. 目前仅为 https://oi-wiki.org/dp/state/ 片段

>  状压 DP 是动态规划的一种，通过将状态压缩为整数来达到优化转移的目的。

## 例题
在 $N\times N$ 的棋盘里面放 $K$ 个国王（$1 \leq N \leq 9, 1 \leq K \leq N \times N$），使他们互不攻击，共有多少种摆放方案。

国王能攻击到它上下左右，以及左上左下右上右下八个方向上附近的各一个格子，共 $8$ 个格子。

### 解释

> 设 $f(i,j,l)$ 表示前 $i$ 行，第 $i$ 行的状态为 $j$，且棋盘上已经放置 $l$ 个国王时的合法方案数。

> 对于编号为 $j$ 的状态，我们用二进制整数 $sit(j)$ 表示国王的放置情况，$sit(j)$ 的某个二进制位为 $0$ 表示对应位置不放国王，为 $1$ 表示在对应位置上放置国王；用 $sta(j)$ 表示该状态的国王个数，即二进制数 $sit(j)$ 中 $1$ 的个数。例如，状态可用二进制数 $100101$ 来表示（棋盘左边对应二进制低位），则有 $sit(j)=100101_{(2)}=37, sta(j)=3$。

> 设当前行的状态为 $j$，上一行的状态为 $x$，可以得到下面的状态转移方程：$f(i,j,l) = \sum f(i-1,x,l-sta(j))$。

>  设上一行的状态编号为 $x$，在保证当前行和上一行不冲突的前提下，枚举所有可能的 $x$ 进行转

- 转移方程：

$$
f(i,j,l) = \sum f(i-1,x,l-sta(j))
$$

```c++
long long sta[2005], sit[2005], f[15][2005][105];
int n, k, cnt;

void dfs(int x, int num, int cur) {
  if (cur >= n) {  // 有新的合法状态
    sit[++cnt] = x;
    sta[cnt] = num;
    return;
  }
  dfs(x, num, cur + 1);  // cur位置不放国王
  dfs(x + (1 << cur), num + 1,
      cur + 2);  // cur位置放国王，与它相邻的位置不能再放国王
}

bool compatible(int j, int x) {
  if (sit[j] & sit[x]) return false;
  if ((sit[j] << 1) & sit[x]) return false;
  if (sit[j] & (sit[x] << 1)) return false;
  return true;
}

int main() {
  cin >> n >> k;
  dfs(0, 0, 0);  // 先预处理一行的所有合法状态
  for (int j = 1; j <= cnt; j++) f[1][j][sta[j]] = 1;
  for (int i = 2; i <= n; i++)
    for (int j = 1; j <= cnt; j++)
      for (int x = 1; x <= cnt; x++) {
        if (!compatible(j, x)) continue;  // 排除不合法转移
        for (int l = sta[j]; l <= k; l++) f[i][j][l] += f[i - 1][x][l - sta[j]];
      }
  long long ans = 0;
  for (int i = 1; i <= cnt; i++) ans += f[n][i][k];  // 累加答案
  cout << ans << endl;
  return 0;
}
```

# 优化

## 斜率优化

> 有 $n$ 个玩具，第 $i$ 个玩具价值为 $c_i$。要求将这 $n$ 个玩具排成一排，分成若干段。对于一段 $[l,r]$，它的代价为 $(r-l+\sum_{i=l}^r c_i-L)^2$。其中 $L$ 是一个常量，求分段的最小代价。

### 朴素的 DP 做法

令 $f_i$ 表示前 $i$ 个物品，分若干段的最小代价。

状态转移方程：$f_i=\min_{j<i}\{f_j+(i-(j+1)+pre_i-pre_j-L)^2\}=\min_{j<i}\{f_j+(pre_i-pre_j+i-j-1-L)^2\}$。

其中 $pre_i$ 表示前 $i$ 个数的和，即 $\sum_{j=1}^i c_j$。

该做法的时间复杂度为 $O(n^2)$，无法解决本题。

### 优化

考虑简化上面的状态转移方程式：令 $s_i=pre_i+i,L'=L+1$，则 $f_i=\min_{j<i}\{f_j+(s_i-s_j-L')^2\}$。

将与 $j$ 无关的移到外面，我们得到

$$
f_i - (s_i-L')^2=\min_{j<i}\{f_j+s_j^2 + 2s_j(L'-s_i) \} 
$$

考虑一次函数的斜截式 $y=kx+b$，将其移项得到 $b=y-kx$。我们将与 $j$ 有关的信息表示为 $y$ 的形式，把同时与 $i,j$ 有关的信息表示为 $kx$，把要最小化的信息（与 $i$ 有关的信息）表示为 $b$，也就是截距。具体地，设

$$
\begin{aligned}
x_j&=s_j\\
y_j&=f_j+s_j^2\\
k_i&=-2(L'-s_i)\\
b_i&=f_i-(s_i-L')^2\\
\end{aligned}
$$

则转移方程就写作 $b_i = \min_{j<i}\{ y_j-k_ix_j \}$。我们把 $(x_j,y_j)$ 看作二维平面上的点，则 $k_i$ 表示直线斜率，$b_i$ 表示一条过 $(x_j,y_j)$ 的斜率为 $k_i$ 的直线的截距。问题转化为了，选择合适的 $j$（$1\le j<i$），最小化直线的截距。

![slope_optimization](https://oi-wiki.org/dp/images/optimization.svg)

如图，我们将这个斜率为 $k_i$ 的直线从下往上平移，直到有一个点 $(x_p,y_p)$ 在这条直线上，则有 $b_i=y_p-k_ix_p$，这时 $b_i$ 取到最小值。算完 $f_i$，我们就把 $(x_i,y_i)$ 这个点加入点集中，以做为新的 DP 决策。那么，我们该如何维护点集？

容易发现，可能让 $b_i$ 取到最小值的点一定在下凸壳上。因此在寻找 $p$ 的时候我们不需要枚举所有 $i-1$ 个点，只需要考虑凸包上的点。而在本题中 $k_i$ 随 $i$ 的增加而递增，因此我们可以单调队列维护凸包。

具体地，设 $K(a,b)$ 表示过 $(x_a,y_a)$ 和 $(x_b,y_b)$ 的直线的斜率。考虑队列 $q_l,q_{l+1},\ldots,q_r$，维护的是下凸壳上的点。也就是说，对于 $l<i<r$，始终有 $K(q_{i-1},q_i) < K(q_i,q_{i+1})$ 成立。

我们维护一个指针 $e$ 来计算 $b_i$ 最小值。我们需要找到一个 $K(q_{e-1},q_e)\le k_i< K(q_e,q_{e+1})$ 的 $e$（特别地，当 $e=l$ 或者 $e=r$ 时要特别判断），这时就有 $p=q_e$，即 $q_e$ 是 $i$ 的最优决策点。由于 $k_i$ 是单调递增的，因此 $e$ 的移动次数是均摊 $O(1)$ 的。

在插入一个点 $(x_i,y_i)$ 时，我们要判断是否 $K(q_{r-1},q_r)<K(q_r,i)$，如果不等式不成立就将 $q_r$ 弹出，直到等式满足。然后将 $i$ 插入到 $q$ 队尾。

这样我们就将 DP 的复杂度优化到了 $O(n)$。

概括一下上述斜率优化模板题的算法：

1.  将初始状态入队。
2.  每次使用一条和 $i$ 相关的直线 $f(i)$ 去切维护的凸包，找到最优决策，更新 $dp_i$。
3.  加入状态 $dp_i$。如果一个状态（即凸包上的一个点）在 $dp_i$ 加入后不再是凸包上的点，需要在 $dp_i$ 加入前将其剔除。

## 四边形

考虑最简单的情形，我们要解决如下一系列最优化问题。

$$
f(i) = \min_{1 \leq j \leq i} w(j,i) \qquad \left(1 \leq i \leq n\right) \tag{1}
$$

这里假定成本函数 $w(j,i)$ 可以在 $O(1)$ 时间内计算。

动态规划的状态转移方程经常可以写作一系列最优化问题的形式。以（1）式为例，这些问题含有参数 $i$，问题的目标函数和可行域都可以依赖于 $i$。每一个问题都是在给定参数 $i$ 时，选取某个可行解 $j$ 来最小化目标函数的取值。为表述方便，下文将参数为 $i$ 的最优化问题简称为「问题 $i$」，该最优化问题的可行解 $j$ 称为「决策 $j$」，目标函数在最优解处取得的值则称为「状态 $f(i)$」。同时，记问题 $i$ 对应的最小最优决策点为 $\mathop{\mathrm{opt}}(i)$。

在一般的情形下，这些问题总时间复杂度为 $O(n^2)$。这是由于对于问题 $i$，我们需要考虑所有可能的决策 $j$。而在满足决策单调性时，可以有效缩小决策空间，优化总复杂度。

-   **决策单调性**：对于任意 $i_1 < i_2$，必然成立 $\mathop{\mathrm{opt}}(i_1) \leq \mathop{\mathrm{opt}}(i_2)$。
        对于问题 $i$，最优决策集合未必是一个区间。决策单调性实际可以定义在最优决策集合上。对于集合 $A$ 和 $B$，可以定义 $A \leq B$ 当且仅当对于任意 $a\in A$ 和 $b\in B$，成立 $\min\{a,b\}\in A$ 和 $\max\{a,b\}\in B$。这蕴含最小（最大）最优决策点的单调性，即此处采取的定义。本文关于最小最优决策点叙述的结论，同样适用于最大最优决策点。但是，存在情形，某更大问题的最小最优决策严格小于另一更小问题的最大最优决策，亦即可能对某些 $i_1 < i_2$ 成立 $\mathop{\mathrm{optmax}}(i_1) > \mathop{\mathrm{optmin}}(i_2)$，所以在书写代码时，应保证总是求得最小或最大的最优决策点。
    -   另一方面，拥有相同最小最优决策的问题构成一个区间。这一区间，作为最小最优决策的函数，应严格递增。亦即，给定 $j_1 = \mathop{\mathrm{opt}}(i_1)$，$j_2 = \mathop{\mathrm{opt}}(i_2)$，如果 $j_1 < j_2$，那么必然有 $i_1 < i_2$。换言之，如果决策 $j_1 < j_2$ 能够成为最小最优决策的问题区间分别是 $[l_{j_1},r_{j_1}]$ 和 $[l_{j_2},r_{j_2}]$，那么必然有 $r_{j_1} < l_{j_2}$。


最常见的判断决策单调性的方法是通过四边形不等式（quadrangle inequality）。

-   **四边形不等式**：如果对于任意 $a\leq b\leq c\leq d$ 均成立

$$
w(a,c)+w(b,d) \leq w(a,d)+w(b,c),
$$

则称函数 $w$ 满足四边形不等式（简记为「交叉小于包含」）。若等号永远成立，则称函数 $w$ 满足 **四边形恒等式**。

如果没有特别说明，以下都会保证 $a\leq b\leq c\leq d$。四边形不等式给出了一个决策单调性的充分不必要条件。

+ 定理 1: 若 $w$ 满足四边形不等式，则问题 (1) 满足决策单调性。

  要证明这一点，可采用反证法。假设对某些 $c < d$，成立 $a = \mathop{\mathrm{opt}}(d) < \mathop{\mathrm{opt}}(c) = b$。此时有 $a < b \leq c < d$。根据最优化条件，$w(a,d) \leq w(b,d)$ 且 $w(b,c) < w(a,c)$，于是，$w(a,d) - w(b,d) \leq 0 < w(a,c) - w(b,c)$，这与四边形不等式矛盾。

四边形不等式可以理解在合理的定义域内，$w$ 的二阶混合差分 $\Delta_i\Delta_jw(j,i)$ 非正。
利用决策单调性，有两种常见算法可以将算法复杂度优化到 $O(n\log n)$。

### 分治

要求解所有状态，只需要求解所有最优决策点。为了对所有 $1 \leq i \leq n$ 求解 $\mathop{\mathrm{opt}}(i)$，首先计算 $\mathop{\mathrm{opt}}(n/2)$，而后分别计算 $1 \leq i < n/2$ 和 $n/2 < i \leq n$ 上的 $\mathop{\mathrm{opt}}(i)$，注意此时已知前半段的 $\mathop{\mathrm{opt}}(i)$ 必然位于 $1$ 和 $\mathop{\mathrm{opt}}(n/2)$ 之间（含端点），而后半段的 $\mathop{\mathrm{opt}}(i)$ 必然位于 $\mathop{\mathrm{opt}}(n/2)$ 和 $\mathop{\mathrm{opt}}(n)$ 之间（含端点）。对于两个子区间，也类似处理，直至计算出每个问题的最优决策。在分治的过程中记录搜索的上下边界，就可以保证算法复杂度控制在 $O(n\log n)$。递归树层数为 $O(\log n)$，而每层中，单个决策点至多计算两次，所以总的计算次数是 $O(n\log n)$。


```cpp
int w(int j, int i);

void DP(int l, int r, int k_l, int k_r) {
  int mid = (l + r) / 2, k = k_l;
  // 求状态f[mid]的最优决策点
  for (int j = k_l; j <= min(k_r, mid - 1); ++j)
    if (w(j, mid) < w(k, mid)) k = j;
  f[mid] = w(k, mid);
  // 根据决策单调性得出左右两部分的决策区间，递归处理
  if (l < mid) DP(l, mid - 1, k_l, k);
  if (r > mid) DP(mid + 1, r, k, k_r);
}
```
### 二分队列

注意到对于每个决策点 $j$，能使其成为最小最优决策点的问题 $i$ 必然构成一个区间。可以通过单调队列记录到目前为止每个决策点可以解决的问题的区间，这样，问题的最优解自然可以通过队列中记录的决策点计算得到。算法大致如下。

```cpp
int val(int j, int i);
int lt[N], rt[N], f[N];
deque<int> dq;
// 初始化队列
dq.emplace_back(1);
lt[1] = 1;
rt[n] = n;
// 顺次考虑所有问题和决策
for (int j = 1; j <= n; ++j) {
  // 出队
  while (!dq.empty() && rt[dq.front()] < j) {
    dq.pop_front();
  }
  // 计算
  f[j] = val(dq.front(), j);
  // 入队
  while (!dq.empty() && val(j, lt[dq.back()]) < val(dq.back(), lt[dq.back()])) {
    dq.pop_back();
  }
  if (dq.empty()) {
    dq.emplace_back(j);
    lt[j] = j + 1;
    rt[j] = n;
  } else if (val(j, rt[dq.back()]) < val(dq.back(), rt[dq.back()])) {
    if (rt[dq.back()] < n) {
      dq.emplace_back(j);
      lt[j] = rt[dq.back()] + 1;
      rt[j] = n;
    }
  } else {
    int ll = lt[dq.back()];
    int rr = rt[dq.back()];
    int i;
    // 二分
    while (ll <= rr) {
      int mm = (ll + rr) / 2;
      if (val(j, mm) < val(dq.back(), mm)) {
        i = mm;
        rr = mm - 1;
      } else {
        ll = mm + 1;
      }
    }
    rt[dq.back()] = i - 1;
    dq.emplace_back(j);
    lt[j] = i;
    rt[j] = n;
  }
}
```
