#ifndef __TYPES_H__
#define __TYPES_H__

#include <cstdint>
#include <vector>

namespace stat {

// 1-D vector type
template <typename T>
using Vec = std::vector<T>;

// 2-D matrix type
template <typename T>
using Mat = std::vector<Vec<T>>;

template <typename T = float>
struct Data {
    Mat<T> data;
    uint32_t m;
    uint32_t n;
};

}  // namespace stat

#endif  // __TYPES_H__