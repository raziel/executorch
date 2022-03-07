#pragma once
#include <Tensor.h>
#include <Scalar.h>
#include <error_message.h>
#include <ArrayRef.h>
#include <Tag.h>

namespace torch {
namespace executor {


// Aggregate typing system similar to IValue only slimmed down with less functionality,
// no dependencies on atomic, and fewer supported types to better suit embedded systems
struct EValue {
    union Payload {
        // Scalar will be implicitly supported through the following 3 types
        int64_t as_int;
        double as_double;
        bool as_bool;

        // Raw pointer instead of intrusive_ptr to avoid atomic dependency
        Tensor* as_tensor;
        utils::ArrayRef<int64_t>* as_int_list;
        utils::ArrayRef<double>* as_double_list;
        utils::ArrayRef<bool>* as_bool_list;
        utils::ArrayRef<Tensor>* as_tensor_list;

        // TODO
        // c10::optional equivalent
        // ArraryRef<Scalar>* as_scalar_list;

        Payload() {}
        ~Payload() {}
    };
    Payload payload;
    Tag tag;

    EValue() : tag(Tag::None) {}

    EValue(int64_t i) : tag(Tag::Int) {
        payload.as_int = i;
    }

    bool isInt() const {
        return tag == Tag::Int;
    }

    int64_t toInt() const {
        if (!isInt()) {
            error_with_message("EValue is not an int.");
        }
        return payload.as_int;
    }

    EValue(double d) : tag(Tag::Double) {
        payload.as_double = d;
    }

    bool isDouble() const {
        return tag == Tag::Double;
    }

    double toDouble() const {
        if (!isDouble()) {
            error_with_message("EValue is not a Double.");
        }
        return payload.as_double;
    }

    EValue(bool b) : tag(Tag::Bool) {
        payload.as_bool = b;
    }

    bool isBool() const {
        return tag == Tag::Bool;
    }

    bool toBool() const {
        if (!isBool()) {
            error_with_message("EValue is not a Bool.");
        }
        return payload.as_bool;
    }

    // TODO Support Scalar. Need implict constructors from scalar before we can support in EValue
    // The plan will be to convert out of scalar upon EValue creation and let the implicit constuctor
    // handle toScalar().
    // EValue(Scalar s) {
    //     <set tag and correct union value based on dtype of s>
    // }

    // bool isScalar() const {
    //     return tag == Tag::Bool || tag == Tag::Double || tag == Tag::Int;
    // }

    // Scalar toScalar() const {
    //     if (isDouble())
    //         return toDouble();
    //     else if (isInt())
    //         return toInt();
    //     else if (isBool())
    //         return toBool();
    //     else
    //         error_with_message("EValue is not a Scalar.");
    // }

    EValue(Tensor* t) : tag(Tag::Tensor) {
        payload.as_tensor = t;
    }

    bool isTensor() const {
        return tag == Tag::Tensor;
    }

    Tensor* toTensor() const {
        if (!isTensor()) {
            error_with_message("EValue is not a Tensor.");
        }
        return payload.as_tensor;
    }

    EValue(utils::ArrayRef<int64_t>* i) : tag(Tag::ListInt) {
        payload.as_int_list = i;
    }

    bool isIntList() const {
        return tag == Tag::ListInt;
    }

    utils::ArrayRef<int64_t>* toIntList() const {
        if (!isIntList()) {
            error_with_message("EValue is not an Int List.");
        }
        return payload.as_int_list;
    }

    EValue(utils::ArrayRef<bool>* b) : tag(Tag::ListBool) {
        payload.as_bool_list = b;
    }

    bool isBoolList() const {
        return tag == Tag::ListBool;
    }

    utils::ArrayRef<bool>* toBoolList() const {
        if (!isBoolList()) {
            error_with_message("EValue is not a Bool List.");
        }
        return payload.as_bool_list;
    }

    EValue(utils::ArrayRef<double>* d) : tag(Tag::ListDouble) {
        payload.as_double_list = d;
    }

     bool isDoubleList() const {
        return tag == Tag::ListDouble;
    }

    utils::ArrayRef<double>* toDoubleList() const {
        if (!isDoubleList()) {
            error_with_message("EValue is not a Double List.");
        }
        return payload.as_double_list;
    }

    EValue(utils::ArrayRef<Tensor>* t) : tag(Tag::ListTensor) {
        payload.as_tensor_list = t;
    }

    bool isTensorList() const {
        return tag == Tag::ListTensor;
    }

    utils::ArrayRef<Tensor>* toTensorList() const {
        if (!isIntList()) {
            error_with_message("EValue is not a Tensor List.");
        }
        return payload.as_tensor_list;
    }

    // EValue(utils::ArrayRef<Scalar>* s) : tag(Tag::ListScalar) {
    //     payload.as_scalar_list = s;
    // }

    // bool isScalarList() const {
    //     return tag == Tag::ListScalar;
    // }

    // utils::ArrayRef<Scalar>* toScalarList() const {
    //     if (!isScalarList()) {
    //         error_with_message("EValue is not a Scalar List.");
    //     }
    //     return payload.as_scalar_list;
    // }
};

} // namespace executor
} // namespace torch
