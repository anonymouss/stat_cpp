#include <cstdio>

#include "Types.h"
#include "Utils.h"

#define TestName "LoadData"
#define ENTER printf("\n=== Run test " TestName " ===\n\n");
#define EXIT printf("\n=== Exit test " TestName " ===\n\n");

int main() {
    ENTER;

    auto info = [](stat::Data<float> data, const char *msg = "none") {
        printf("INFO: %s size: MxN (%ux%u)\n", msg, data.m, data.n);
    };

    // MNIST
    {
        printf("MNIST\n");
        // if use C++17, this can be simplified to `auto [trainX, trainY] = stat::mnist::loadTrainSet();`
        auto train = stat::mnist::loadTrainSet();
        auto trainX = std::get<0>(train);
        auto trainY = std::get<1>(train);
        info(trainX, "trainX");
        info(trainY, "trainY");

        auto test = stat::mnist::loadTestSet();
        auto testX = std::get<0>(test);
        auto testY = std::get<1>(test);
        info(testX, "testX");
        info(testY, "testY");

        printf("INFO: image size WxH (%ux%u)\n", stat::mnist::ImageWidth, stat::mnist::ImageHeight);
    }

    // IRIS
    {
        printf("IRIS\n");
        auto train = stat::iris::loadTrainSet();
        auto trainX = std::get<0>(train);
        auto trainY = std::get<1>(train);
        info(trainX, "trainX");
        info(trainY, "trainY");

        auto test = stat::iris::loadTestSet();
        auto testX = std::get<0>(test);
        auto testY = std::get<1>(test);
        info(testX, "testX");
        info(testY, "testY");

        auto disp = [](stat::Data<float> data, const char *msg = "none") {
            printf("==> %s\n", msg);
            for (auto i = 0; i < data.m; ++i) {
                for (auto j = 0; j < data.n; ++j) {
                    printf("%f, ", data.data[i][j]);
                }
                printf("\n");
            }
        };

        disp(trainX, "train X");
        disp(trainY, "train y");
        disp(testX, "test X");
        disp(testY, "test y");
    } 

    EXIT;
}