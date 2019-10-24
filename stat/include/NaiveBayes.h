#ifndef __NAIVE_BAYES_H__
#define __NAIVE_BAYES_H__

#include "Math.h"
#include "Model.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace stat {

constexpr double smoothing = 0.1;

template <typename DataType, typename LabelType>
class NaiveBayes : public Model<DataType, LabelType> {
public:
    explicit NaiveBayes(ModelParam param);
    virtual ~NaiveBayes() = default;

    virtual bool train(const Data<DataType> &X_train, const Data<LabelType> &y_train) final;

    virtual LabelType predict(const Vec<DataType> &X) final;

    virtual double validate(const Data<DataType> &X_test, const Data<LabelType> &y_test) final;

    virtual void describe() const final;

private:
    enum NBType : uint32_t {
        GAUSSIAN,
        BERNOULLI,
        // OTHERS
    };

    struct GaussianParam {
        double mu;     // mean
        double sigma;  // standard deviation
    };

    bool isModelShow;
    NBType type;
    std::unordered_map<LabelType, std::vector<GaussianParam>> model;
    std::unordered_map<LabelType, double> priorprobabilities;

    std::vector<GaussianParam> summarize(const Mat<DataType> &data);
    bool train_gaussian(const Data<DataType> &X_train, const Data<LabelType> &y_train);
    bool train_bernoulli(const Data<DataType> &X_train, const Data<LabelType> &y_train);
    LabelType predict_gaussian(const Vec<DataType> &X);
    LabelType predict_bernoulli(const Vec<DataType> &X);
};

template <typename DataType, typename LabelType>
NaiveBayes<DataType, LabelType>::NaiveBayes(ModelParam param)
    : isModelShow(false), type(NBType::GAUSSIAN) {
    // TODO: only support gaussian model currently
    const auto &model_show = param.find("model_show");
    if (model_show != param.cend()) {
        if (model_show->second == "true") { isModelShow = true; }
    }

    const auto &model_type = param.find("model_type");
    if (model_type != param.cend()) {
        if (model_type->second == "gaussian") {
            type = NBType::GAUSSIAN;
        } else if (model_type->second == "bernoulli") {
            printf("TODO: not supproted currently\n");
            // type = NBType::BERNOULLI;
        }
    }
}

template <typename DataType, typename LabelType>
bool NaiveBayes<DataType, LabelType>::train(const Data<DataType> &X_train,
                                            const Data<LabelType> &y_train) {
    if (type == NBType::GAUSSIAN) {
        return train_gaussian(X_train, y_train);
    } else if (type == NBType::BERNOULLI) {
        return train_bernoulli(X_train, y_train);
    }
    return false;
}

template <typename DataType, typename LabelType>
bool NaiveBayes<DataType, LabelType>::train_gaussian(const Data<DataType> &X_train,
                                                     const Data<LabelType> &y_train) {
    Clock clk(__func__);

    auto m = X_train.m, n = X_train.n;
    if (m == 0 || n == 0) {
        printf("ERROR: invalid training set\n");
        return false;
    }
    std::unordered_map<LabelType, Mat<DataType>> dataByLabel;
    for (auto i = 0; i < m; ++i) {
        auto label = y_train.data[i][0];
        priorprobabilities[label] += 1.0 / m;  // calculate prior probabilities, P(y)
        // check if lable exsit in map
        if (dataByLabel.find(label) == dataByLabel.end()) {
            Mat<DataType> d;
            d.emplace_back(X_train.data[i]);
            dataByLabel.emplace(std::make_pair(label, d));
        } else {
            dataByLabel[label].emplace_back(X_train.data[i]);
        }
    }

    // calculate gaussian params which will be used to calculate P(X|y)
    for (const auto &label_data : dataByLabel) {
        auto param = summarize(label_data.second);
        model.emplace(std::make_pair(label_data.first, param));
    }

    printf("INFO: traning done\n");
    describe();
    return true;
}

template <typename DataType, typename LabelType>
bool NaiveBayes<DataType, LabelType>::train_bernoulli(const Data<DataType> &X_train,
                                                      const Data<LabelType> &y_train) {
    Clock clk(__func__);
    // TODO
    return false;
}

template <typename DataType, typename LabelType>
LabelType NaiveBayes<DataType, LabelType>::predict(const Vec<DataType> &X) {
    if (type == NBType::GAUSSIAN) {
        return predict_gaussian(X);
    } else if (type == NBType::BERNOULLI) {
        return predict_bernoulli(X);
    }  // TODO: more elses
    return 0;
}

template <typename DataType, typename LabelType>
LabelType NaiveBayes<DataType, LabelType>::predict_gaussian(const Vec<DataType> &X) {
    std::unordered_map<LabelType, double> probabilities;
    for (const auto &m : model) {
        probabilities[m.first] = std::log(priorprobabilities[m.first]);
        for (auto i = 0; i < X.size(); ++i) {
            auto param = m.second[i];
            // calculate P(X|y), the original form is multiplication (P(X1|y)*P(X2|y)...P(Xn|y)),
            // but prob may be very small >> 0. valeu by be 0 after multiplication (precision
            // issue), hence, change form to sum of log value. log(AB) = log(A) + log(B)
            probabilities[m.first] += std::log(gaussian_prob(X[i], param.mu, param.sigma));
        }
    }
    double maxProb = -Inf<double>;
    LabelType predicted = 0;
    for (const auto &prob : probabilities) {
        if (prob.second > maxProb) {
            maxProb = prob.second;
            predicted = prob.first;
        }
    }
    return predicted;
}

template <typename DataType, typename LabelType>
LabelType NaiveBayes<DataType, LabelType>::predict_bernoulli(const Vec<DataType> &X) {
    // TODO
    return 0;
}

template <typename DataType, typename LabelType>
double NaiveBayes<DataType, LabelType>::validate(const Data<DataType> &X_test,
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
void NaiveBayes<DataType, LabelType>::describe() const {
    return;
    if (!isModelShow) return;
    if (type == NBType::GAUSSIAN) {
        printf("Naive Bayes with Gaussian model\n");
        for (const auto &m : model) {
            printf("Label: %f\n\t{\n", m.first);
            for (const auto &param : m.second) {
                printf("\t\tmean: %f, std: %f,\n", param.mu, param.sigma);
            }
            printf("\t\n");
        }
    } else if (type == NBType::BERNOULLI) {
        // TODO
    }
}

template <typename DataType, typename LabelType>
std::vector<typename NaiveBayes<DataType, LabelType>::GaussianParam>
NaiveBayes<DataType, LabelType>::summarize(const Mat<DataType> &data) {
    std::vector<GaussianParam> param;
    for (auto i = 0; i < data[0].size(); ++i) {
        auto x = getCol(data, i);
        // smoothing is added to avoid sigma/variance being 0. denominator in calculating gaussian
        // probability
        GaussianParam p{mean(x), stdev(x) + smoothing};
        param.emplace_back(p);
    }
    return param;
}

}  // namespace stat

#endif  // __NAIVE_BAYES_H__