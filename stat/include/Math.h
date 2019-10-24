#ifndef __MATH_H__
#define __MATH_H__

#include "Types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace stat {

template <typename T>
constexpr T Inf = std::numeric_limits<T>::infinity();

template <typename T>
constexpr T NaN = std::numeric_limits<T>::quiet_NaN();

constexpr double pi = 3.141592653589793238463;

template <typename T>
Vec<T> allocVec(uint32_t N, T v = 0) {
    Vec<T> vec(N, v);
    return vec;
}

template <typename T>
Mat<T> allocMat(uint32_t M, uint32_t N, T v = 0) {
    Mat<T> mat(M, allocVec<T>(N, v));
    return mat;
}

template <typename T>
double sign(T v) {
    return std::signbit(v) ? -1.0 : 1.0;
}

template <typename T1, typename T2>
double dot(const Vec<T1> &x1, const Vec<T2> &x2) {
    double sum = 0.0;
    auto m1 = x1.size(), m2 = x2.size();
    if (m1 != m2) {
        printf("ERROR: dot, dimensions are not aligned of two input vectors [%zu, %zu]\n", m1, m2);
        return sum;
    }
    for (auto i = 0; i < m1; ++i) { sum += (x1[i] * x2[i]); }
    return sum;
}

template <typename T1, typename T2>
Vec<double> dot(const Vec<T1> &x, T2 a) {
    auto v = allocVec<double>(x.size(), 0);
    for (auto i = 0; i < x.size(); ++i) { v[i] = x[i] * a; }
    return v;
}

template <typename T1, typename T2>
Vec<double> dot(T1 a, const Vec<T2> &x) {
    return dot(x, a);
}

template <typename T1, typename T2>
Vec<double> add(const Vec<T1> &v1, const Vec<T2> &v2) {
    auto m1 = v1.size(), m2 = v2.size();
    if (m1 != m2) {
        printf("ERROR: add, dimensions are not aligned of two input vectors [%zu, %zu]\n", m1, m2);
        return {};
    }
    auto v = allocVec<double>(m1);
    for (auto i = 0; i < m1; ++i) { v[i] = v1[i] + v2[i]; }
    return v;
}

template <typename T1, typename T2>
Vec<double> add(const Vec<T1> &v1, T2 a) {
    auto v2 = allocVec<T2>(v1.size(), a);
    return add(v1, v2);
}

template <typename T1, typename T2>
Vec<double> add(T1 a, const Vec<T2> &v2) {
    return add(v2, a);
}

template <typename T>
Mat<T> gram(const Mat<T> &X) {
    auto m = X.size();
    auto g = allocMat<T>(m, m, 0);
    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < m; ++j) { g[i][j] = dot(X[i], X[j]); }
    }
    return g;
}

template <typename T>
Mat<T> transpose(const Mat<T> &mat) {
    auto m = mat.size();
    if (m == 0) { printf("ERROR: transpose on empty matrix (rows = 0)\n"); }
    auto n = mat[0].size();
    if (n == 0) { printf("ERROR: transpose on empty matrix (cols = 0)\n"); }
    auto transMat = allocMat<T>(n, m, 0);
    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < n; ++j) { transMat[j][i] = mat[i][j]; }
    }
    return transMat;
}

template <typename T>
Vec<T> getRow(const Mat<T> &mat, uint32_t r) {
    if (r >= mat.size()) {
        printf("ERROR: row index out of bound\n");
        return {};
    }
    Vec<T> row = mat[r];
    return row;
}

template <typename T>
Vec<T> getCol(const Mat<T> &mat, uint32_t c) {
    auto m = mat.size(), n = mat[0].size();
    if (c >= n) {
        printf("ERROR: col index out of bound\n");
        return {};
    }
    Vec<T> col;
    for (auto i = 0; i < m; ++i) { col.emplace_back(mat[i][c]); }
    return col;
}

template <typename T1, typename T2>
double Lp(const Vec<T1> &x, const Vec<T2> &y, uint32_t p = 2) {
    auto mx = x.size(), my = y.size();
    double sum = 0.0;
    if (mx == my && mx > 0) {
        for (auto i = 0; i < mx; ++i) {
            sum += std::pow(std::abs(static_cast<double>(x[i] - y[i])), static_cast<double>(p));
        }
        return std::pow(sum, 1.0 / static_cast<double>(p));
    }
    return sum;
}

template <typename T>
T sum(const Vec<T> &X) {
    T sum = 0;
    // overflow risk
    for (const auto &x : X) sum += x;
    return sum;
}

template <typename T>
double mean(const Vec<T> &X) {
    if (X.size() == 0) {
        printf("ERROR: invalid input vector\n");
        return NaN<double>;
    }
    double sum = stat::sum(X);
    return sum / X.size();
}

// \sigma^2 = \frac{\sum{(X-\mu)^2}}{N}
template <typename T>
double stdev(const Vec<T> &X) {
    if (X.size() == 0) {
        printf("ERROR: invalid input vector\n");
        return NaN<double>;
    }
    auto mu = mean(X);
    double sum = 0.0;
    for (const auto &x : X) { sum += std::pow(x - mu, 2); }
    return std::sqrt(sum / X.size());
}

// gaussian probability, aka. normal distribution
// f(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
template <typename T>
double gaussian_prob(T x, double mu, double sigma) {
    if (sigma == 0) {
        printf("ERROR: sigma = 0\n");
        return NaN<double>;
    }
    double exp = std::exp(-(std::pow(x - mu, 2) / 2.0 / std::pow(sigma, 2)));
    return exp / std::sqrt(2 * pi) / sigma;
}

}  // namespace stat

#endif  // __MATH_H__
