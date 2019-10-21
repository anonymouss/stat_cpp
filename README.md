# Statistical Learning Method

A simple implementation of book ***Statistical Learning Method*** (*aka.* ***统计学习方法***) with C++

## Progress

- [x] Perceptron
- [ ] k-NN
- [ ] Naive Bayes
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

1. [Mnist dataset](./data/mnist/)(http://yann.lecun.com/exdb/mnist/).

    multi-classes, 10 labels

2. [Iris dataset](./data/iris/)(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

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
#include "Uitls.h"

auto train = stat::mnist::loadTrainSet();
auto X_train = std::get<0>(train);
auto y_train = std::get<1>(train);

// if compile with C++17 onwards, tuple can be simplified by:
auto [X_train, y_train] = stat::mnist::loadTrainSet(); // structured binding
```

### Model

```cpp
#include "Stat.h"

// returns std::unique_ptr<Model>, extra indicates to some extra infos some model requires.
auto model = stat::CreateModel<DataType, LabelType>(ModelType, extra);
model->train(X_train, y_train);
model->validate(X_test, y_test);

// Outputs:
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

### Extra

successfully compiled by Clang 8.0 with libc++ on Ubuntu 14.04, enabled C++17.

g++/msvc is not verified.

