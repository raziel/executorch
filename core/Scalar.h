#pragma once

#include <error_message.h>
#include <Tag.h>

namespace torch {
namespace executor {

// WIP CLASS
class Scalar {
  public:
    Scalar() : Scalar(int64_t(0)) {}
    Scalar(int val) : tag(Tag::Int) {
        v.as_int = val;
    }
    Scalar(int64_t val) : tag(Tag::Int) {
        v.as_int = val;
    }
    Scalar(bool val) : tag(Tag::Bool) {
        v.as_bool = val;
    }
    Scalar(double val) : tag(Tag::Double) {
        v.as_double = val;
    }

    bool isInt() const {
        return tag == Tag::Int;
    }

    int64_t toInt() const {
        if (!isInt()) {
            error_with_message("Scalar is not an int.");
        }
        return v.as_int;
    }

    bool isDouble() const {
        return tag == Tag::Double;
    }

    double toDouble() const {
        if (!isDouble()) {
            error_with_message("Scalar is not a Double.");
        }
        return v.as_double;
    }

    bool isBool() const {
        return tag == Tag::Bool;
    }

    bool toBool() const {
        if (!isBool()) {
            error_with_message("Scalar is not a Bool.");
        }
        return v.as_bool;
    }
  private:
    Tag tag;
    union v_t {
        double as_double;
        int64_t as_int;
        bool as_bool;
        v_t() {} // default constructor
    } v;
};

}
}
