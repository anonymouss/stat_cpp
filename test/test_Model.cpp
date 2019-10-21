#include <cstdio>

#include "Stat.h"

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
        auto train = stat::iris::loadTrainSet<double>();
        auto trainX = std::get<0>(train);
        auto trainY = std::get<1>(train);

        auto test = stat::iris::loadTestSet<double>();
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

        TEST_MODEL(stat::ModelType::MODEL_PERCEPTRON, Wrap_v<double>, Wrap_v<double>,
                   {{"model_type", "original"}});
        TEST_MODEL(stat::ModelType::MODEL_PERCEPTRON, Wrap_v<double>, Wrap_v<double>,
                   {{"model_type", "dual"}});
    }

    EXIT;
}