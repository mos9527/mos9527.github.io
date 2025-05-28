---
author: mos9527
lastmod: 2025-05-23T17:28:54.741308
title: Arithmetic Competition Notes - FFT/Polynomial/Number Theory Topics
tags: ["ACM","Competeive Programming","XCPC","(Code) Templates","Problem sets","Codeforces","C++"]
categories: ["Problem Solutions", "Competeive Programming", "Collection/compilation"]
ShowToc: true
TocOpen: true
typora-root-url: ..\..\static
---

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
