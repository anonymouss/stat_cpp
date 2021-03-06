# Statistical Learning Method

A simple implementation of book ***Statistical Learning Method*** (*aka.* ***统计学习方法***) with C++

## Progress

- [x] Perceptron (original form and dual form impl)
- [x] k-NN (simple knn and kdtree impl)
- [x] Naive Bayes (gaussian distribution model)
- [ ] Decision Tree
- [ ] Logisitic Regression
- [ ] SVM
- [ ] AdaBoost
- [ ] EM
- [ ] HMM
- [ ] CRF

## Description

### Data

Dataset:

1. [Mnist dataset](./data/mnist/) (http://yann.lecun.com/exdb/mnist/).

    multi-classes, 10 labels

2. [Iris dataset](./data/iris/) (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

    splited to 2 classes

data structure:

```cpp
template <typename T>
struct Data {
    Mat<T> data;    // 2-d matrix, std::vector<std::vector<T>>
    uint32_t m;     // data rows
    uint32_t n;     // data cols
};
```

to load dataset:

```C++
#include "Stat.h"

auto train = stat::mnist::loadTrainSet(); // or stat::iris::loadTrainSet()
auto X_train = std::get<0>(train);
auto y_train = std::get<1>(train);

auto test = stat::mnist::loadTestSet();
auto X_test = std::get<0>(train);
auto y_test = std::get<1>(train);

// if compile with C++17 onwards, tuple can be simplified by:
auto [X_train, y_train] = stat::mnist::loadTrainSet(); // structured binding
```

### Model

```cpp
#include "Stat.h"

// returns std::unique_ptr<Model>, model params is {key : value} map
auto model = stat::CreateModel<DataType, LabelType>(
    ModelType,
    {{"param_key", "param_value"},
    {"model_type", "kdtree"}} /* ... model params */);
model->train(X_train, y_train);
model->validate(X_test, y_test);

// Sample Outputs:
// INFO: creating perceptron model
// INFO: training original form
// INFO: training done.
//
// Perceptron:
//
// Model: $f(x) = sign(w \cdot x + b)$
//        w = [ -0.500000, -0.170000, 0.810000, 0.980000,  ]
//        b = -0.300000
//
// accuracy: 1.000000
//
// INFO: creating perceptron model
// INFO: training dual form
// INFO: training done.
//
// Perceptron:
//
// Model: $f(x) = sign(w \cdot x + b)$
//        w = [ -0.400000, -5.100000, 8.300000, 4.000000,  ]
//        b = -1.000000
//
// accuracy: 1.000000
```

### Reference

- Python impl of 'statistical learning method': https://github.com/fengdu78/lihang-code
- C++ KD-tree: https://github.com/junjiedong/KDTree
- Naive Bayes: 
    - https://www.cnblogs.com/sxron/p/5452821.html,            
    - https://blog.csdn.net/qq_27009517/article/details/80044431
    - https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/

### Extra

successfully compiled by Clang 8.0 with libc++ on Ubuntu 14.04, enabled C++17.

g++/msvc is not verified.
