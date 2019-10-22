#ifndef __KNN_H__
#define __KNN_H__

#include "Math.h"
#include "Model.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
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

    // combine X (sample data) and y (sample lable) together. X will be sorted during building
    // kd-tree, this is to make sure that y and X are in the same order
    using Point = std::pair<Vec<DataType>, LabelType>;

    struct KdNode {
        Vec<DataType> data;  // a point from kd space
        LabelType label;
        uint32_t level;
        std::shared_ptr<KdNode> left;
        std::shared_ptr<KdNode> right;
        double nearest_dist;
        KdNode(const Vec<DataType> &_data, LabelType _label, uint32_t _level)
            : data(_data),
              label(_label),
              level(_level),
              left(nullptr),
              right(nullptr),
              nearest_dist(0.0) {}
    };

    uint32_t k;
    uint32_t p;
    KnnType type;

    Data<DataType> xdata;
    Data<LabelType> ydata;
    std::size_t feature_dim;

    std::shared_ptr<KdNode> root;
    std::shared_ptr<KdNode> nearest;

    // compare kNearestNodes, compare each node's nearest_dist
    using NodeCmp = std::function<bool(std::shared_ptr<KdNode>, std::shared_ptr<KdNode>)>;

    std::set<std::shared_ptr<KdNode>, NodeCmp> kNearestNodes;

    bool train_simple(const Data<DataType> &X_train, const Data<LabelType> &y_train);
    bool train_kdtree(const Data<DataType> &X_train, const Data<LabelType> &y_train);
    LabelType predict_simple(const Vec<DataType> &X);
    LabelType predict_kdtree(const Vec<DataType> &X);

    std::shared_ptr<KdNode> createKdTree(typename std::vector<Point>::iterator start,
                                         typename std::vector<Point>::iterator end,
                                         uint32_t depth = 0);
    void findNearest(std::shared_ptr<KdNode> currentNode, Vec<DataType> X);
};

template <typename DataType, typename LabelType>
KNN<DataType, LabelType>::KNN(ModelParam param)
    : k(3),
      p(2),
      type(KnnType::SIMPLE_KNN),
      xdata({{}}),
      ydata({{}}),
      feature_dim(0),
      root(nullptr),
      nearest(nullptr),
      kNearestNodes([](std::shared_ptr<KdNode> n1, std::shared_ptr<KdNode> n2) {
          return n1->nearest_dist < n2->nearest_dist;
      }) {
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
    if (type == KnnType::SIMPLE_KNN) {
        return train_simple(X_train, y_train);
    } else {
        return train_kdtree(X_train, y_train);
    }
}

template <typename DataType, typename LabelType>
bool KNN<DataType, LabelType>::train_simple(const Data<DataType> &X_train,
                                            const Data<LabelType> &y_train) {
    Clock clk(__func__);

    printf("INFO: simple k-NN has no training progress\n");
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
bool KNN<DataType, LabelType>::train_kdtree(const Data<DataType> &X_train,
                                            const Data<LabelType> &y_train) {
    Clock clk(__func__);

    printf("INFO: creating KD-Tree\n");
    auto m = X_train.m, n = X_train.n;
    if (m == 0 || n == 0) {
        printf("ERROR: invalid training set\n");
        return false;
    }
    feature_dim = m;
    std::vector<Point> Points;
    for (auto i = 0; i < m; ++i) {
        Points.emplace_back(std::make_pair(X_train.data[i], y_train.data[i][0]));
    }
    root = createKdTree(Points.begin(), Points.end(), 0);
    if (root) {
        printf("INFO: KD-Tree created.\n");
        return true;
    } else {
        printf("ERROR: KD-Tree create failed.\n");
        return false;
    }
}

template <typename DataType, typename LabelType>
LabelType KNN<DataType, LabelType>::predict(const Vec<DataType> &X) {
    if (type == KnnType::SIMPLE_KNN) {
        return predict_simple(X);
    } else {
        return predict_kdtree(X);
    }
}

template <typename DataType, typename LabelType>
LabelType KNN<DataType, LabelType>::predict_simple(const Vec<DataType> &X) {
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
LabelType KNN<DataType, LabelType>::predict_kdtree(const Vec<DataType> &X) {
    if (!root) {
        printf("ERROR: KD-Tree doesn't exist, please creat KD-Tree first\n");
        return 0;
    }

    // clear previous nearest info.
    nearest = nullptr;
    kNearestNodes.clear();

    findNearest(root, X);
    if (!nearest) {
        printf("ERROR: find nearst failed\n");
        return 0;
    }
    std::unordered_set<LabelType> label_set;
    std::multiset<LabelType> label_all;
    for (const auto &n : kNearestNodes) {
        label_set.emplace(n->label);
        label_all.emplace(n->label);
    }
    std::size_t maxCount = 0;
    LabelType predictedLabel = 0;
    for (auto label : label_set) {
        auto count = label_all.count(label);
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

// Ref: https://github.com/junjiedong/KDTree
// KD-Tree is actually a BST(Binary Search Tree), but it's order relation is compared between each
// node's value on current split axis (index)
template <typename DataType, typename LabelType>
std::shared_ptr<typename KNN<DataType, LabelType>::KdNode> KNN<DataType, LabelType>::createKdTree(
    typename std::vector<Point>::iterator start, typename std::vector<Point>::iterator end,
    uint32_t depth) {
    if (start >= end) return nullptr;
    auto axis = depth % feature_dim;
    auto cmp = [axis](const Point &p1, const Point &p2) { return p1.first[axis] < p2.first[axis]; };
    std::size_t len = end - start;
    auto mid = start + len / 2;
    std::nth_element(start, mid, end, cmp);
    // move to make left_val < mid_val, right_val >= mid_val
    mid = std::partition(start, mid,
                         [=](const auto &p) { return p.first[axis] != mid->first[axis]; });
    auto node =
        std::make_shared<typename KNN<DataType, LabelType>::KdNode>(mid->first, mid->second, depth);
    node->left = createKdTree(start, mid, depth + 1);
    node->right = createKdTree(mid + 1, end, depth + 1);
    return node;
}

template <typename DataType, typename LabelType>
void KNN<DataType, LabelType>::findNearest(
    std::shared_ptr<typename KNN<DataType, LabelType>::KdNode> currentNode, Vec<DataType> X) {
    if (!currentNode) return;
    auto axis = currentNode->level % feature_dim;
    // find the leaf node which may be the nearest one to X (check the vertical distance from the
    // split axis).
    if (currentNode->left || currentNode->right) {
        if (X[axis] < currentNode->data[axis] && currentNode->left) {
            findNearest(currentNode->left, X);
        } else if (currentNode->right) {
            findNearest(currentNode->right, X);
        }
    }

    // save k nearest nodes. update the nearest_dist to the lp distance of two points
    auto dist = Lp(X, currentNode->data, p);
    if (!nearest || dist < nearest->nearest_dist) {
        nearest = currentNode;
        nearest->nearest_dist = dist;
        if (kNearestNodes.size() < k) {
            kNearestNodes.insert(nearest);
        } else {
            auto last = kNearestNodes.end();
            --last;
            if ((*last)->nearest_dist > dist) {
                kNearestNodes.erase(last);
                kNearestNodes.insert(nearest);
            }
        }
    }

    // the vertical distance from split axis < lp distance means that a circle with X as point and
    // lp distance as radius may intersect with other nodes' split axes. check them.
    if (std::abs(X[axis] - currentNode->data[axis]) < nearest->nearest_dist) {
        if (currentNode->left && X[axis] >= currentNode->data[axis]) {
            findNearest(currentNode->left, X);
        } else if (currentNode->right && X[axis] < currentNode->data[axis]) {
            findNearest(currentNode->right, X);
        }
    }
}

}  // namespace stat

#endif  // __KNN_H__