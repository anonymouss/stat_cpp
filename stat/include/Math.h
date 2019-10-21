#ifndef __MATH_H__
#define __MATH_H__

#include "Types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace stat {

template <typename T>
Vec<T> allocVec(uint32_t N, T v = 0) {
    Vec<T> vec(N, v);
    return vec;
}

template <typename T>
Mat<T> allocMat(uint32_t M, uint32_t N, T v = 0) {
    Mat<T> mat(M, allocVec<T>(N));
    return mat;
}

template <typename T>
double sign(T v) {
    return std::signbit(v) ? -1.0 : 1.0;
}

template <typename T1, typename T2>
double dot(Vec<T1> x1, Vec<T2> x2) {
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
Vec<double> dot(Vec<T1> x, T2 a) {
    auto v = allocVec<double>(x.size(), 0);
    for (auto i = 0; i < x.size(); ++i) { v[i] = x[i] * a; }
    return v;
}

template <typename T1, typename T2>
Vec<double> dot(T1 a, Vec<T2> x) {
    return dot(x, a);
}

template <typename T1, typename T2>
Vec<double> add(Vec<T1> v1, Vec<T2> v2) {
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
Vec<double> add(Vec<T1> v1, T2 a) {
    auto v2 = allocVec<T2>(v1.size(), a);
    return add(v1, v2);
}

template <typename T1, typename T2>
Vec<double> add(T1 a, Vec<T2> v2) {
    return add(v2, a);
}

template <typename T>
Mat<T> gram(Mat<T> X) {
    auto m = X.size();
    auto g = allocMat<T>(m, m, 0);
    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < m; ++j) { g[i][j] = dot(X[i], X[j]); }
    }
    return g;
}

template <typename T>
Vec<T> getRow(Mat<T> mat, uint32_t r) {
    if (r >= mat.size()) {
        printf("ERROR: row index out of bound\n");
        return {};
    }
    Vec<T> row = mat[r];
    return row;
}

template <typename T>
Vec<T> getCol(Mat<T> mat, uint32_t c) {
    auto m = mat.size(), n = mat[0].size();
    if (c >= n) {
        printf("ERROR: col index out of bound\n");
        return {};
    }
    Vec<T> col;
    for (auto i = 0; i < m; ++i) { col.emplace_back(mat[i][c]); }
    return col;
}

template <typename DataType, typename LabelType>
double Lp(Vec<DataType> x, Vec<LabelType> y, uint32_t p = 2) {
    auto mx = x.size(), my = y.size();
    double sum = 0.0;
    if (mx == my && mx > 0) {
        for (auto i = 0; i < mx; ++i) {
            sum += std::pow(static_cast<double>(std::abs(x[i] - y[i])), static_cast<double>(p));
        }
        return std::pow(sum, 1.0 / static_cast<double>(p));
    }
    return sum;
}

}  // namespace stat

#endif  // __MATH_H__