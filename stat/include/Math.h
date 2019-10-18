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
        printf("ERROR: dimensions are not aligned of two input vectors\n");
        return sum;
    }
    for (auto i = 0; i < m1; ++i) {
        sum += (x1[i] * x2[i]);
    }
    return sum;
}

template <typename T1, typename T2>
Vec<double> dot(Vec<T1> x, T2 a) {
    auto v = allocVec<double>(x.size(), 0);
    for (auto i = 0; i < x.size(); ++i) {
        v[i] = x[i] * a;
    }
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
        printf("ERROR: dimensions are not aligned of two input vectors\n");
        return {};
    }
    auto v = allocVec<double>(m1);
    for (auto i = 0; i < m1; ++i) {
        v[i] = v1[i] + v2[i];
    }
    return v;
}

template <typename T>
Mat<T> gram(Mat<T> X) {
    auto m = X.size();
    auto g = allocMat<T>(m, m, 0);
    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < m; ++j) {
            g[i][j] = dot(X[i], X[j]);
        }
    }
    return g;
}

}  // namespace stat

#endif  // __MATH_H__