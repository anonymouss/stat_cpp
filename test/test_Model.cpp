#include "Stat.h"

#include <cstdio>
#include <thread>

#define TEST_IRIS   // comment out to disable test on iris dataset
#define TEST_MNIST  // comment out to disable test on mnist dataset

#define TestName "Model"
#define ENTER printf("\n=== Run test " TestName " ===\n\n");
#define EXIT printf("\n=== Exit test " TestName " ===\n\n");

#define CHARS(n, ch) printf("%s\n\n", std::string(n, ch).c_str());

/**
 * Wrap is a help class to generate a value by a given type. then lambda expression can get such
 * type by decltype(). This looks cumbersome, but I don't find a good way to pass type info to
 * lambda like pass type to templatized function.
 */
template <typename T>
struct Wrap {
    static constexpr T value = static_cast<T>(1.0);
};

template <typename T>
constexpr T Wrap_v = Wrap<T>::value;

int main() {
    ENTER;

#ifdef TEST_IRIS
    {  // iris
        printf("*** Test on iris dataset ***\n\n");

        auto train = stat::iris::loadTrainSet<double>();
        auto trainX = std::get<0>(train);
        auto trainY = std::get<1>(train);

        auto test = stat::iris::loadTestSet<double>();
        auto testX = std::get<0>(test);
        auto testY = std::get<1>(test);

        // NOTE: There is a trap. We already know tuple here can be simplified to auto [trainX,
        // trainY] = ... from C++17 (aka. structured binding). But we can't use this here, the
        // lambda here will reject it. Details explained here:
        // https://stackoverflow.com/questions/46114214/lambda-implicit-capture-fails-with-variable-declared-from-structured-binding

        auto TEST_MODEL = [&](stat::ModelType type, auto DataType, auto LabelType,
                              stat::ModelParam param) {
            CHARS(50, '=');
            auto model = stat::CreateModel<decltype(DataType), decltype(LabelType)>(type, param);
            if (model) {
                model->train(trainX, trainY);
                model->validate(testX, testY);
            } else {
                printf("ERROR: create model failed\n");
            }
            CHARS(50, '=');
        };

        // test perceptron
        TEST_MODEL(stat::ModelType::MODEL_PERCEPTRON, Wrap_v<double>, Wrap_v<double>,
                   {{"model_type", "original"}, {"model_show", "true"}});  // original form
        TEST_MODEL(stat::ModelType::MODEL_PERCEPTRON, Wrap_v<double>, Wrap_v<double>,
                   {{"model_type", "dual"}, {"model_show", "true"}});  // dual form

        // test k-NN
        TEST_MODEL(stat::ModelType::MODEL_KNN, Wrap_v<double>, Wrap_v<double>,
                   {{"k", "5"}, {"model_type", "knn"}});  // simple knn
        TEST_MODEL(stat::ModelType::MODEL_KNN, Wrap_v<double>, Wrap_v<double>,
                   {{"k", "5"}, {"model_type", "kdtree"}});  // kdtree

        // test naive bayes
        TEST_MODEL(stat::ModelType::MODEL_NAIVE_BAYES, Wrap_v<double>, Wrap_v<double>,
                   {{"model_show", "true"}});  // simple knn
    }
#endif  // TEST_IRIS

#ifdef TEST_MNIST
    {  // mnist
        printf("*** Test on mnist dataset ***\n\n");

        auto train = stat::mnist::loadTrainSet<double>();
        auto trainX = std::get<0>(train);
        auto trainY = std::get<1>(train);

        auto test = stat::mnist::loadTestSet<double>();
        auto testX = std::get<0>(test);
        auto testY = std::get<1>(test);

        auto TEST_MODEL = [&](stat::ModelType type, auto DataType, auto LabelType,
                              stat::ModelParam param) {
            CHARS(50, '=');
            auto model = stat::CreateModel<decltype(DataType), decltype(LabelType)>(type, param);
            if (model) {
                model->train(trainX, trainY);
                model->validate(testX, testY);
            } else {
                printf("ERROR: create model failed\n");
            }
            CHARS(50, '=');
        };

        // test k-NN
        // DISABLED: mnist dataset has 60000 train samples and 10000 test samples, each sample has
        // 784 dim features. this simple kNN impl cost too much time to valid all samples
        // one sample result:
        // accuracy: 0.968700
        // 4h 45m 31s 470ms elapsed
        // TEST_MODEL(stat::ModelType::MODEL_KNN, Wrap_v<double>, Wrap_v<double>, {{"k", "5"}});

        // TEST_MODEL(stat::ModelType::MODEL_KNN, Wrap_v<double>, Wrap_v<double>,
        //            {{"model_type", "kdtree"}});

        // test naive bayes
        // without binaryzation - accuracy is only 0.6097
        TEST_MODEL(stat::ModelType::MODEL_NAIVE_BAYES, Wrap_v<double>, Wrap_v<double>,
                   {{"model_show", "true"}, {"model_type", "gaussian"}});
        {
            // binaryzation - accuracy is only 0.7103
            double threshold = 127.0;
            uint32_t m_train = trainX.m, m_test = testX.m, n = trainX.n;
            stat::Mat<uint32_t> binTrainData, binTestData;
            for (auto i = 0; i < m_train; ++i) {
                stat::Vec<uint32_t> v;
                for (auto j = 0; j < n; ++j) {
                    uint32_t e = trainX.data[i][j] > threshold ? 1 : 0;
                    v.emplace_back(e);
                }
                binTrainData.emplace_back(v);
                v.clear();
            }
            for (auto i = 0; i < m_test; ++i) {
                stat::Vec<uint32_t> v;
                for (auto j = 0; j < n; ++j) {
                    uint32_t e = testX.data[i][j] > threshold ? 1 : 0;
                    v.emplace_back(e);
                }
                binTestData.emplace_back(v);
                v.clear();
            }
            stat::Data<uint32_t> X_train_new{binTrainData, m_train, n};
            stat::Data<uint32_t> X_test_new{binTestData, m_test, n};

            auto model = stat::CreateModel<uint32_t, double>(stat::ModelType::MODEL_NAIVE_BAYES,
                                                             {{"model_show", "false"}});
            if (model) {
                model->train(X_train_new, trainY);
                model->validate(X_test_new, testY);
            }
        }
    }
#endif  // TEST_MNIST

    EXIT;
}