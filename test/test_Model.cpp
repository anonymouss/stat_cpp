#include <cstdio>

#include "Stat.h"

#define TestName "Model"
#define ENTER printf("\n=== Run test " TestName " ===\n\n");
#define EXIT printf("\n=== Exit test " TestName " ===\n\n");

#define TEST_MODEL(type, DataType, LabelType)                   \
{                                                               \
    auto model = stat::CreateModel<DataType, LabelType>(type);  \
    model->train(trainX, trainY);                               \
    model->validate(testX, testY);                              \
}

int main() {
    ENTER;

    { // iris
        auto train = stat::iris::loadTrainSet<double>();
        auto trainX = std::get<0>(train);
        auto trainY = std::get<1>(train);

        auto test = stat::iris::loadTestSet<double>();
        auto testX = std::get<0>(test);
        auto testY = std::get<1>(test);

        TEST_MODEL(stat::ModelType::MODEL_PERCEPTRON, double, double);

    }

    EXIT;
}