#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include "Math.h"
#include "Model.h"

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
 *
 * Dual form:
 * Model:   $f(x) = sign\left \{ \sum_{j=1}^N{\alpha_j y_j x_j\cdot x+b} \right \}$
 * Miss condition:
 *          $y_i\left \{ \sum_{j=1}^N{\alpha_jy_jx_j\cdot x_i + b} \right \} \le 0$
 * Gram Maxtrix:
 *          $G = \left [ x_i\cdot x_j \right ]_{N\times N}$
 * Update:
 *          $\alpha_i = \alpha_i + \eta$
 *          $b = b + \eta y_i$
 * finally, $w = \sum_{i=1}^N{\alpha_i y_i x_i}$
 */
template <typename DataType, typename LabelType>
class Perceptron : public Model<DataType, LabelType> {
public:
    explicit Perceptron(ModelParam param);
    virtual ~Perceptron() = default;

    virtual bool train(const Data<DataType> &X_train, const Data<LabelType> &y_train) final;

    virtual LabelType predict(const Vec<DataType> &X) final;

    virtual double validate(const Data<DataType> &X_test, const Data<LabelType> &y_test) final;

    virtual void describe() const final;

private:
    enum ModelType : uint32_t {
        ORIGNAL,
        DUAL,
    };

    uint32_t type;
    Vec<double> weight;
    double bias;
    double eta;
    Vec<double> alpha;
    Mat<DataType> gr;

    double f0(Vec<DataType> X);
    double f1(Vec<LabelType> y, Vec<DataType> g);
    virtual bool train_original(const Data<DataType> &X_train,
                                const Data<LabelType> &y_train) final;
    virtual bool train_dual(const Data<DataType> &X_train, const Data<LabelType> &y_train) final;
};

template <typename DataType, typename LabelType>
Perceptron<DataType, LabelType>::Perceptron(ModelParam param)
    : type(ModelType::ORIGNAL), weight({}), bias(0.0), eta(0.0), alpha({}), gr({{}}) {
    const auto &model_type = param.find("model_type");
    if (model_type != param.cend()) {
        if (model_type->second == "original")
            type = ModelType::ORIGNAL;
        else if (model_type->second == "dual")
            type = ModelType::DUAL;
    }
}

template <typename DataType, typename LabelType>
LabelType Perceptron<DataType, LabelType>::predict(const Vec<DataType> &X) {
    return static_cast<LabelType>(sign(f0(X)));
}

template <typename DataType, typename LabelType>
double Perceptron<DataType, LabelType>::validate(const Data<DataType> &X_test,
                                                 const Data<LabelType> &y_test) {
    Clock clk(__func__);

    double correct = 0.0;
    auto m = X_test.m;
    for (auto i = 0; i < m; ++i) {
        if (predict(X_test.data[i]) == y_test.data[i][0]) ++correct;
    }
    double acc = correct / m;
    printf("accuracy: %f\n\n", acc);
    return acc;
}

template <typename DataType, typename LabelType>
double Perceptron<DataType, LabelType>::f0(Vec<DataType> X) {
    return dot(X, weight) + bias;
}

template <typename DataType, typename LabelType>
double Perceptron<DataType, LabelType>::f1(Vec<LabelType> y, Vec<DataType> g) {
    double sum = 0.0;
    for (auto i = 0; i < y.size(); ++i) { sum += alpha[i] * y[i] * g[i]; }
    sum += bias;
    return sum;
}

template <typename DataType, typename LabelType>
bool Perceptron<DataType, LabelType>::train(const Data<DataType> &X_train,
                                            const Data<LabelType> &y_train) {
    if (type == ModelType::ORIGNAL) {
        return train_original(X_train, y_train);
    } else {
        return train_dual(X_train, y_train);
    }
}

template <typename DataType, typename LabelType>
bool Perceptron<DataType, LabelType>::train_original(const Data<DataType> &X_train,
                                                     const Data<LabelType> &y_train) {
    Clock clk(__func__);

    printf("INFO: training original form\n");
    bool hasMisclassified = true;
    auto m = X_train.m, n = X_train.n;
    if (m == 0 || n == 0) {
        printf("ERROR: invalid training set\n");
        return false;
    }
    weight = allocVec<double>(n, 1);
    eta = 0.1;
    while (hasMisclassified) {
        int misclassified = 0;
        for (auto i = 0; i < m; ++i) {
            auto X = X_train.data[i];
            auto y = y_train.data[i][0];
            if (y * f0(X) <= 0) {
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
bool Perceptron<DataType, LabelType>::train_dual(const Data<DataType> &X_train,
                                                 const Data<LabelType> &y_train) {
    Clock clk(__func__);

    printf("INFO: training dual form\n");
    bool hasMisclassified = true;
    auto m = X_train.m, n = X_train.n;
    if (m == 0 || n == 0) {
        printf("ERROR: invalid training set\n");
        return false;
    }
    eta = 1;
    alpha = allocVec<double>(m, 0);
    weight = allocVec<double>(n, 1);
    auto gr = gram(X_train.data);
    auto y = getCol(y_train.data, 0);
    while (hasMisclassified) {
        int misclassified = 0;
        for (auto i = 0; i < m; ++i) {
            auto g = getCol(gr, i);  // m dim vector
            if (y[i] * f1(y, g) <= 0) {
                alpha[i] += eta;
                bias += y[i] * eta;
                ++misclassified;
            }
        }
        if (misclassified == 0) hasMisclassified = false;
    }
    for (int i = 0; i < m; ++i) { weight = add(weight, dot((alpha[i] * y[i]), X_train.data[i])); }
    printf("INFO: training done.\n");
    describe();
    return true;
}

template <typename DataType, typename LabelType>
void Perceptron<DataType, LabelType>::describe() const {
    printf("\nPerceptron:\n\n");
    printf("Model: $f(x) = sign(w \\cdot x + b)$\n");
    printf("       w = [ ");
    for (const auto &w : weight) printf("%f, ", w);
    printf(" ]\n");
    printf("       b = %f\n\n", bias);
}

}  // namespace stat

#endif  // __PERCEPTRON_H__