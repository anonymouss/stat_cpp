#ifndef __MODEL_H__
#define __MODEL_H__

#include "Types.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace stat {

using ModelParam = std::unordered_map<std::string, std::string>;

/**
 * Base model class
 */
template <typename DataType, typename LabelType>
class Model {
public:
    virtual ~Model() = default;

    virtual bool train(const Data<DataType> &X_train, const Data<LabelType> &y_train) = 0;

    virtual LabelType predict(const Vec<DataType> &X) = 0;

    virtual double validate(const Data<DataType> &X_test, const Data<LabelType> &y_test) = 0;

    virtual void describe() const = 0;
};

}  // namespace stat

#endif  // __MODEL_H__