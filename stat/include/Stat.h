#ifndef __STAT_H__
#define __STAT_H__

#include "KNN.h"
#include "Model.h"
#include "NaiveBayes.h"
#include "Perceptron.h"
#include "Types.h"
#include "Utils.h"

#include <memory>

namespace stat {

enum ModelType : uint32_t {
    MODEL_UNKNOWN,
    MODEL_PERCEPTRON,           // Perceptron model
    MODEL_KNN,                  // k-Nearest Neighbor model
    MODEL_NAIVE_BAYES,          // Naive Bayes model
    MODEL_DECISION_TREE,        // Decision tree model
    MODEL_LOGISTIC_REGRESSION,  // Logistic regression model
    MODEL_SVM,                  // Support Vector Machine model
    MODEL_ADA_BOOST,            // AdaBoost model
    MODEL_EM,                   // Expectation-Maximization
    MODEL_HMM,                  // Hidden Markov Model
    MODEL_CRF,                  // Condition Random Field
    MODEL_END,
};

template <typename DataType, typename LabelType>
std::unique_ptr<Model<DataType, LabelType>> CreateModel(ModelType type = ModelType::MODEL_UNKNOWN,
                                                        ModelParam param = {{}}) {
    std::unique_ptr<Model<DataType, LabelType>> model = nullptr;
    switch (type) {
        case MODEL_PERCEPTRON: {
            printf("INFO: creating Perceptron model\n");
            model = std::make_unique<Perceptron<DataType, LabelType>>(param);
            break;
        }
        case MODEL_KNN: {
            printf("INFO: creating kNN model\n");
            model = std::make_unique<KNN<DataType, LabelType>>(param);
            break;
        }
        case MODEL_NAIVE_BAYES: {
            printf("INFO: creating Naive Bayes model\n");
            model = std::make_unique<NaiveBayes<DataType, LabelType>>(param);
            break;
        }
        case MODEL_DECISION_TREE:
        case MODEL_LOGISTIC_REGRESSION:
        case MODEL_SVM:
        case MODEL_ADA_BOOST:
        case MODEL_EM:
        case MODEL_HMM:
        case MODEL_CRF:
        default: printf("ERROR: unknown/unsupported model type.\n");
    }
    return model;
}

}  // namespace stat

#endif  // __STAT_H__