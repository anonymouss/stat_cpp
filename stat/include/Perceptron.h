#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include "Model.h"
#include "Math.h"

namespace stat {

/**
 * Perceptron Model
 *
 * Model:   $f(x) = sign(w \cdot x + b)$
 *          $sign(x) = \left\{\begin{matrix} +1, x \ge 0 & \\-1, x < 0 & \end{matrix}\right$
 * Loss:    $L(w,b) = -\sum_{x_i\subseteq M}{y_i(w\cdot x_i + b)}$
 * Update by gradient descent:
 *          $w = w + \eta y_ix_i$
 *          $b = b + \eta y_i$
 */
template <typename DataType, typename LabelType>
class Perceptron : public Model<DataType, LabelType> {
public:
    Perceptron();
    virtual ~Perceptron() = default;

    virtual bool train(const Data<DataType> &X_train, const Data<LabelType> &y_train) final;

    virtual LabelType predict(const Vec<DataType> &X) final;

    virtual double validate(const Data<DataType> &X_test, const Data<LabelType> &y_test) final;

    virtual void describe() const final;

private:
    Vec<double> weight;
    double bias;
    double eta;

    double f(Vec<DataType> X);
};

template <typename DataType, typename LabelType>
Perceptron<DataType, LabelType>::Perceptron() : weight({}), bias(0.0), eta(0.1) {}

template <typename DataType, typename LabelType>
bool Perceptron<DataType, LabelType>::train(const Data<DataType> &X_train, const Data<LabelType> &y_train) {
    bool hasMisclassified = true;
    auto m = X_train.m, n = X_train.n;
    if (m == 0 || n == 0) {
        printf("ERROR: invalid training set\n");
        return false;
    }
    weight = allocVec<double>(n, 1);
    while (hasMisclassified) {
        int misclassified = 0;
        for (auto i = 0; i < m; ++i) {
            auto X = X_train.data[i];
            auto y = y_train.data[i][0];
            if (y * f(X) <= 0) {
                weight = add(weight, dot(eta, dot(y, X)));
                bias += eta * y;
                ++misclassified;
            }
        }
        if (misclassified == 0) hasMisclassified = false;
    }
    printf("INFO: training done.\n");
    describe();
    return true;
}

template <typename DataType, typename LabelType>
LabelType Perceptron<DataType, LabelType>::predict(const Vec<DataType> &X) {
    return static_cast<LabelType>(sign(f(X)));
}

template <typename DataType, typename LabelType>
double Perceptron<DataType, LabelType>::validate(const Data<DataType> &X_test, const Data<LabelType> &y_test) {
    double correct = 0.0;
    auto m = X_test.m;
    for (auto i = 0; i < m; ++i) {
        if (predict(X_test.data[i]) == y_test.data[i][0])
            ++correct;
    }
    double acc = correct / m;
    printf("accuracy: %f\n", acc);
    return acc;
}

template <typename DataType, typename LabelType>
double Perceptron<DataType, LabelType>::f(Vec<DataType> X) {
    return dot(X, weight) + bias;
}

template <typename DataType, typename LabelType>
void Perceptron<DataType, LabelType>::describe() const {
    printf("\nPerceptron:\n\n");
    printf("Model: $f(x) = sign(w \\cdot x + b)$\n");
    printf("       w = [ ");
    for (const auto &w : weight) printf("%f, ", w);
    printf(" ]\n");
    printf("       b = %f\n", bias);
}

}  // namespace stat

#endif  // __PERCEPTRON_H__