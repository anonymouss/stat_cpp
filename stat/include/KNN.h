#ifndef __KNN_H__
#define __KNN_H__

#include "Math.h"
#include "Model.h"

#include <map>
#include <set>
#include <unordered_set>
#include <utility>

namespace stat {

template <typename DataType, typename LabelType>
class KNN : public Model<DataType, LabelType> {
public:
    explicit KNN(ModelParam param);
    virtual ~KNN() = default;

    virtual bool train(const Data<DataType> &X_train, const Data<LabelType> &y_train) final;

    virtual LabelType predict(const Vec<DataType> &X) final;

    virtual double validate(const Data<DataType> &X_test, const Data<LabelType> &y_test) final;

    virtual void describe() const final;

private:
    enum KnnType : uint32_t {
        SIMPLE_KNN,
        KDTREE,
    };

    uint32_t k;
    uint32_t p;
    KnnType type;

    Data<DataType> xdata;
    Data<LabelType> ydata;
};

template <typename DataType, typename LabelType>
KNN<DataType, LabelType>::KNN(ModelParam param)
    : k(3), p(2), type(KnnType::SIMPLE_KNN), xdata({{}}), ydata({{}}) {
    const auto &model_k = param.find("k");
    if (model_k != param.cend()) {
        // trust user input, user code must ensure values are correct
        k = std::stol(model_k->second);
    }

    const auto &model_p = param.find("p");
    if (model_p != param.cend()) { p = std::stol(model_p->second); }

    const auto &model_type = param.find("model_type");
    if (model_type != param.cend()) {
        if (model_type->second == "knn") {
            type = KnnType::SIMPLE_KNN;
        } else if (model_type->second == "kdtree") {
            type = KnnType::KDTREE;
        }
    }
}

template <typename DataType, typename LabelType>
bool KNN<DataType, LabelType>::train(const Data<DataType> &X_train,
                                     const Data<LabelType> &y_train) {
    Clock clk(__func__);

    printf("INFO: k-NN has no training progress\n");
    auto m = X_train.m, n = X_train.n;
    if (m == 0 || n == 0) {
        printf("ERROR: invalid training set\n");
        return false;
    }
    xdata = X_train;
    ydata = y_train;

    describe();
    return true;
}

template <typename DataType, typename LabelType>
LabelType KNN<DataType, LabelType>::predict(const Vec<DataType> &X) {
    auto m = xdata.m;
    if (k > m) {
        printf("WARNING: improper k, set to k = m = %d\n", m);
        k = m;
    }
    std::multimap<double, LabelType> knnMap;
    for (auto i = 0; i < k; ++i) {
        auto dist = Lp(X, xdata.data[i], p);
        knnMap.emplace(std::make_pair(dist, ydata.data[i][0]));
    }
    for (auto i = k; i < m; ++i) {
        // update knn distance map. std::multimap is in lexicographical order by default.
        auto dist = Lp(X, xdata.data[i], p);
        auto last = knnMap.end();
        --last;
        if (last->first > dist) {
            knnMap.erase(last);
            knnMap.emplace(std::make_pair(dist, ydata.data[i][0]));
        }
    }
    std::unordered_set<LabelType> keys;
    std::multiset<LabelType> multiKeys;
    for (const auto &p : knnMap) {
        keys.emplace(p.second);
        multiKeys.emplace(p.second);
    }
    std::size_t maxCount = 0;
    LabelType predictedLabel = 0;
    for (auto label : keys) {
        auto count = multiKeys.count(label);
        if (count > maxCount) {
            maxCount = count;
            predictedLabel = label;
        }
    }
    return predictedLabel;
}

template <typename DataType, typename LabelType>
double KNN<DataType, LabelType>::validate(const Data<DataType> &X_test,
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
void KNN<DataType, LabelType>::describe() const {
    printf("\nKNN:\n\n");
    printf("with k = %u\n\n", k);
}

}  // namespace stat

#endif  // __KNN_H__