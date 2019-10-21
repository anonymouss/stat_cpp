#ifndef __UTILS_H__
#define __UTILS_H__

#include "Types.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

#include <arpa/inet.h>

namespace stat {
namespace mnist {

// mnist dataset utils

constexpr const char *kMnistTrainImages = "data/mnist/train-images.idx3-ubyte";
constexpr const char *kMnistTrainLables = "data/mnist/train-labels.idx1-ubyte";
constexpr const char *kMnistTestImages = "data/mnist/t10k-images.idx3-ubyte";
constexpr const char *kMnistTestLables = "data/mnist/t10k-labels.idx1-ubyte";

/**
constexpr uint32_t kMnistTrainSetSize = 60000;
constexpr uint32_t kMnistTestSetSize = 10000;
constexpr uint32_t kMnistImageWidth = 28;
constexpr uint32_t kMnistImageHeight = 28;
constexpr uint32_t kMnistImageSize = kMnistImageWidth * kMnistImageHeight;
*/

uint32_t ImageWidth = 0;
uint32_t ImageHeight = 0;

// http://yann.lecun.com/exdb/mnist/
constexpr uint32_t kMagicImage = 0x00000803;
constexpr uint32_t kMagicLabel = 0x00000801;

template <typename DataType = float>
Data<DataType> loadData(const char *filename) {
    std::fstream fin(filename, std::fstream::binary | std::fstream::in);
    if (fin.is_open()) {
        uint32_t magic_number = 0, item_num = 0, image_rows = 1, image_cols = 1;
        Mat<DataType> data;
        fin.read((char *)&magic_number, sizeof magic_number);
        fin.read((char *)&item_num, sizeof item_num);
        magic_number = ::ntohl(magic_number);
        item_num = ::ntohl(item_num);
        bool isImage = magic_number == kMagicImage;
        if (isImage) {
            fin.read((char *)&image_rows, sizeof image_rows);
            fin.read((char *)&image_cols, sizeof image_cols);
            image_rows = ::ntohl(image_rows);
            image_cols = ::ntohl(image_cols);
            ImageWidth = image_rows;
            ImageHeight = image_cols;
        }
        auto vector_size = image_rows * image_cols;

        for (auto it = 0; it < item_num; ++it) {
            Vec<DataType> v;
            for (auto i = 0; i < vector_size; ++i) {
                uint8_t e = 0;
                fin.read((char *)&e, sizeof e);
                v.emplace_back(static_cast<DataType>(e));
            }
            data.emplace_back(std::move(v));
            v.clear();
        }
        fin.close();
        // printf("INFO: load (%s) successfully\n", filename);
        return {data, item_num, vector_size};
    } else {
        printf("ERROR: failed to load data from (%s)\n", filename);
        return {{}, 0, 0};
    }
}

template <typename DataType = float>
std::tuple<Data<DataType>, Data<DataType>> loadTrainSet() {
    auto train_image = loadData<DataType>(kMnistTrainImages);
    auto train_label = loadData<DataType>(kMnistTrainLables);
    // if use C++17, just return {train_image, train_label}
    return std::make_tuple(train_image, train_label);
}

template <typename DataType = float>
std::tuple<Data<DataType>, Data<DataType>> loadTestSet() {
    auto test_image = loadData<DataType>(kMnistTestImages);
    auto test_label = loadData<DataType>(kMnistTestLables);
    return std::make_tuple(test_image, test_label);
}

}  // namespace mnist

namespace iris {
constexpr const char *kIrisTrainX = "data/iris/X_train";
constexpr const char *kIrisTrainY = "data/iris/y_train";
constexpr const char *kIrisTestX = "data/iris/X_test";
constexpr const char *kIrisTestY = "data/iris/y_test";

template <typename DataType = float>
Data<DataType> loadData(const char *filename) {
    std::fstream fin(filename, std::fstream::binary | std::fstream::in);
    if (fin.is_open()) {
        Mat<DataType> data;
        uint32_t rows = 0, cols = 0;
        std::string line;
        while (std::getline(fin, line)) {
            Vec<DataType> v;
            ++rows;
            cols = 0;
            std::stringstream ss(line);
            std::string s;
            while (std::getline(ss, s, ' ')) {
                ++cols;
                auto e = static_cast<DataType>(std::stof(s));
                v.emplace_back(e);
            }
            data.emplace_back(std::move(v));
            v.clear();
        }
        return {data, rows, cols};
    } else {
        printf("ERROR: failed to load data from (%s)\n", filename);
        return {{}, 0, 0};
    }
}

template <typename DataType = float>
std::tuple<Data<DataType>, Data<DataType>> loadTrainSet() {
    auto train_data = loadData<DataType>(kIrisTrainX);
    auto train_label = loadData<DataType>(kIrisTrainY);
    return std::make_tuple(train_data, train_label);
}

template <typename DataType = float>
std::tuple<Data<DataType>, Data<DataType>> loadTestSet() {
    auto test_data = loadData<DataType>(kIrisTestX);
    auto test_label = loadData<DataType>(kIrisTestY);
    return std::make_tuple(test_data, test_label);
}
}  // namespace iris
}  // namespace stat

#endif  // __UTILS_H__