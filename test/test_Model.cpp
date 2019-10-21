#include "Stat.h"

#include <cstdio>
#include <thread>

#define TestName "Model"
#define ENTER printf("\n=== Run test " TestName " ===\n\n");
#define EXIT printf("\n=== Exit test " TestName " ===\n\n");

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
            auto model = stat::CreateModel<decltype(DataType), decltype(LabelType)>(type, param);
            if (model) {
                model->train(trainX, trainY);
                model->validate(testX, testY);
            } else {
                printf("ERROR: create model failed\n");
            }
        };

        // test perceptron
        TEST_MODEL(stat::ModelType::MODEL_PERCEPTRON, Wrap_v<double>, Wrap_v<double>,
                   {{"model_type", "original"}});  // original form
        TEST_MODEL(stat::ModelType::MODEL_PERCEPTRON, Wrap_v<double>, Wrap_v<double>,
                   {{"model_type", "dual"}});  // dual form

        // test k-NN
        TEST_MODEL(stat::ModelType::MODEL_KNN, Wrap_v<double>, Wrap_v<double>, {{"k", "5"}});
    }

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
            auto model = stat::CreateModel<decltype(DataType), decltype(LabelType)>(type, param);
            if (model) {
                model->train(trainX, trainY);
                model->validate(testX, testY);
            } else {
                printf("ERROR: create model failed\n");
            }
        };

        // test k-NN
        // DISABLED: mnist dataset has 60000 train samples and 10000 test samples, each sample has
        // 784 dim features. this simple kNN impl cost too much time to valid all samples
        // TEST_MODEL(stat::ModelType::MODEL_KNN, Wrap_v<double>, Wrap_v<double>, {{"k", "5"}});
    }

    EXIT;
}