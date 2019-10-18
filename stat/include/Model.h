#ifndef __MODEL_H__
#define __MODEL_H__

#include "Types.h"

#include <cstdint>

namespace stat {

/**
 * Base model class
 */
template <typename DataType, typename LabelType>
class Model {
public:
    // TODO:
    // struct ModelParam {};
    virtual ~Model() = default;

    virtual bool train(const Data<DataType> &X_train, const Data<LabelType> &y_train) = 0;

    virtual LabelType predict(const Vec<DataType> &X) = 0;

    virtual double validate(const Data<DataType> &X_test, const Data<LabelType> &y_test) = 0;

    virtual void describe() const = 0;
};

}  // namespace stat

#endif  // __MODEL_H__