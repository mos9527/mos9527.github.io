---
author: mos9527
lastmod: 2025-04-16T11:10:42.502000+08:00
title: 算竞笔记 - FFT/多项式类型合集
tags: ["ACM","算竞","XCPC","板子","题集","Codeforces","C++"]
categories: ["题解", "算竞", "合集"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

# Preface

**注：**摘自 https://cp-algorithms.com/algebra/fft.html, https://en.wikipedia.org/wiki/Discrete_Fourier_transform

## 定义

- 多项式$A$的$DFT$即为$A$在各单位根$w_{n, k} = w_n^k = e^{\frac{2 k \pi i}{n}}$之值

$$
\begin{align}
\text{DFT}(a_0, a_1, \dots, a_{n-1}) &= (y_0, y_1, \dots, y_{n-1}) \\
&= (A(w_{n, 0}), A(w_{n, 1}), \dots, A(w_{n, n-1})) \\
&= (A(w_n^0), A(w_n^1), \dots, A(w_n^{n-1}))
\end{align}
$$

- $IDFT$ ($InverseDFT$) 即从这些值$(y_0, y_1, \dots, y_{n-1})$恢复多项式$A$的系数

$$
\text{IDFT}(y_0, y_1, \dots, y_{n-1}) = (a_0, a_1, \dots, a_{n-1})
$$

- 单位根有以下性质

  - 积性
    $$
    w^n = 1 \\
    w^k \ne 1, 0 \lt k \lt n
    $$

  - 所有单位根和为$0$
    $$
    \sum_{k=0}^{n-1} w^k = 0
    $$
    这点利用欧拉公式$e^{ix} = cos x + i\ sin x$看$n$边形对称性很显然

## 应用

考虑两个多项式$A, B$相乘
$$
(A \cdot B)(x) = A(x) \cdot B(x)
$$

- 显然运用$DFT$可得

$$
DFT(A \cdot B) = DFT(A) \cdot DFT(B)
$$

- $A \cdot B$的系数易求

$$
A \cdot B = IDFT(DFT(A \cdot B)) = IDFT(DFT(A) \cdot DFT(B))
$$

## 逆操作（IDFT）

回忆$DFT$的定义
$$
\text{DFT}(a_0, a_1, \dots, a_{n-1}) = (A(w_n^0), A(w_n^1), \dots, A(w_n^{n-1}))
$$
- 写成[矩阵形式](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT)即为

$$
F = \begin{pmatrix}
w_n^0 & w_n^0 & w_n^0 & w_n^0 & \cdots & w_n^0 \\
w_n^0 & w_n^1 & w_n^2 & w_n^3 & \cdots & w_n^{n-1} \\
w_n^0 & w_n^2 & w_n^4 & w_n^6 & \cdots & w_n^{2(n-1)} \\
w_n^0 & w_n^3 & w_n^6 & w_n^9 & \cdots & w_n^{3(n-1)} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
w_n^0 & w_n^{n-1} & w_n^{2(n-1)} & w_n^{3(n-1)} & \cdots & w_n^{(n-1)(n-1)}
\end{pmatrix} \\
$$
- 那么$DFT$操作即为

$$
F\begin{pmatrix}
a_0 \\ a_1 \\ a_2 \\ a_3 \\ \vdots \\ a_{n-1}
\end{pmatrix} = \begin{pmatrix}
y_0 \\ y_1 \\ y_2 \\ y_3 \\ \vdots \\ y_{n-1}
\end{pmatrix}
$$
- 化简有

$$
y_k = \sum_{j=0}^{n-1} a_j w_n^{k j},
$$
其中范德蒙德阵$M$行列各项正交，[可做出结论](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#The_unitary_DFT) $F^{-1} = \frac{1}{n}F^*$, $F_{i,j}^* = \overline{F_{j,i}}$，既有
$$
F^{-1} = \frac{1}{n}
\begin{pmatrix}
w_n^0 & w_n^0 & w_n^0 & w_n^0 & \cdots & w_n^0 \\
w_n^0 & w_n^{-1} & w_n^{-2} & w_n^{-3} & \cdots & w_n^{-(n-1)} \\
w_n^0 & w_n^{-2} & w_n^{-4} & w_n^{-6} & \cdots & w_n^{-2(n-1)} \\
w_n^0 & w_n^{-3} & w_n^{-6} & w_n^{-9} & \cdots & w_n^{-3(n-1)} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
w_n^0 & w_n^{-(n-1)} & w_n^{-2(n-1)} & w_n^{-3(n-1)} & \cdots & w_n^{-(n-1)(n-1)}
\end{pmatrix}
$$
- 那么$IDFT$操作即为

$$
\begin{pmatrix}
a_0 \\ a_1 \\ a_2 \\ a_3 \\ \vdots \\ a_{n-1}
\end{pmatrix} = F^{-1} \begin{pmatrix}
y_0 \\ y_1 \\ y_2 \\ y_3 \\ \vdots \\ y_{n-1}
\end{pmatrix}
$$
- 化简有

$$
a_k = \frac{1}{n} \sum_{j=0}^{n-1} y_j w_n^{-k j}
$$
### 结论

- **注意到$w_i$使用共轭即为$n \cdot \text{IDFT}$**
- 实现中稍作调整即可同时实现$DFT,IDFT$操作；接下来会用到

## 实现（FFT）

朴素包络时间复杂度为$O(n^2)$，这里不做阐述

$FFT$的过程如下

- 令$A(x) = a_0 x^0 + a_1 x^1 + \dots + a_{n-1} x^{n-1}$, 按奇偶拆成两个子多项式

$$
\begin{align}
A_0(x) &= a_0 x^0 + a_2 x^1 + \dots + a_{n-2} x^{\frac{n}{2}-1} \\
A_1(x) &= a_1 x^0 + a_3 x^1 + \dots + a_{n-1} x^{\frac{n}{2}-1}
\end{align}
$$
- 显然有

$$
A(x) = A_0(x^2) + x A_1(x^2).
$$
- 设$\left(y_k^0\right)_{k=0}^{n/2-1} = \text{DFT}(A_0), \left(y_k^1\right)_{k=0}^{n/2-1} = \text{DFT}(A_1)$，前$\frac{n}{2}$项即为

$$
y_k = y_k^0 + w_n^k y_k^1, \quad k = 0 \dots \frac{n}{2} - 1.
$$
- 对后半$\frac{n}{2}$有$$y_{k+n/2} = A\left(w_n^{k+n/2}\right) = A_0\left(w_n^{2k+n}\right) + w_n^{k + n/2} A_1\left(w_n^{2k+n}\right) = A_0\left(w_n^{2k} w_n^n\right) + w_n^k w_n^{n/2} A_1\left(w_n^{2k} w_n^n\right) = A_0\left(w_n^{2k}\right) - w_n^k A_1\left(w_n^{2k}\right) =  y_k^0 - w_n^k y_k^1$$

- **即$$y_{k+n/2} = y_k^0 - w_n^k y_k^1$$，形式上非常接近$y_k$。**综上：

$$
\begin{align}
y_k &= y_k^0 + w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1, \\
y_{k+n/2} &= y_k^0 - w_n^k y_k^1, &\quad k = 0 \dots \frac{n}{2} - 1.
\end{align}
$$

### 结论

- 很显然合并代价是$O(n)$；由$T_{\text{DFT}}(n) = 2 T_{\text{DFT}}\left(\frac{n}{2}\right) + O(n)$则知$FFT$可在$O(nlogn)$时间内解决问题
- 归并实现也将很简单

### Code （归并）

- 若使用`std::complex`实现$w_n$可以直接用[`std::exp`自带特化](https://en.cppreference.com/w/cpp/numeric/complex/exp)求得$w_n = e^{\frac{2\pi i}{n}}$
- 或者利用欧拉公式$e^{ix} = cos x + i\ sin x$可构造`Complex w_n{ .real = cos(2 * PI / n), .imag = sin(2 * PI / n) }`
- 结合之前所述的$DFT$, $IDFT$关系，使用$w_n = -e^{\frac{2\pi i}{n}}$并除$n$即求$IDFT$

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
        if (invert) {
			// 注意：除 log2(n) 次 2 即除 2^log2(n) = n
            A[k] /= 2;
            A[k + n / 2] /= 2;
        }
		w_k *= w_n;
	}   
}
void FFT(cvec& a) { FFT(a, false); }
void IFFT(cvec& y) { FFT(y, true); }
```

