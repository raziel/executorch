#pragma once

namespace torch {
namespace executor {

// TODO: move to error handler
void error_with_message(char* message) {
  // A hacky error function before we have a good convention,
  // better without exception.
  printf("%s\n", message);
  exit(EXIT_FAILURE);
}

struct IntList {
  int size;
  int data[];
};

struct DoubleList {
  int size;
  double data[];
};

struct BoolList {
  int size;
  bool data[];
};

#define TORCH_FORALL_TAGS(_) \
  _(None)                    \
  _(Tensor)                  \
  _(Double)                  \
  _(Int)                     \
  _(Bool)                    \
  _(ListDouble)                  \
  _(ListInt)                     \
  _(ListBool)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
  TORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

// Value is used to unify input/output types in a kernel
// It's possible that an op argument is any of the types.
// in native_functions.yaml, types are Tensor, Scalar, int[], float[], bool,
// tag + union of payload
// Use union of prim or pointers to simplify the structure and storage
struct Value {
  Tag tag;
  union Payload {
    int64_t as_int;
    double as_double;
    bool as_bool;
    // Raw pointers for now instead of intrusive_ptr, because some embedded
    // systems may not support atomic ref count.
    Tensor* as_tensor;
    IntList* as_int_list;
    DoubleList* as_double_list;
    BoolList* as_bool_list;
  };
  Payload payload;

  Value() : tag(Tag::None) {}

  Value(int64_t i) : tag(Tag::Int) {
    payload.as_int = i;
  }

  bool isInt() const {
    return tag == Tag::Int;
  }

  int toInt() const {
    if (!isInt()) {
      error_with_message("Value is not an int.");
    }
    return payload.as_int;
  }

  Value(Tensor* t) : tag(Tag::Tensor) {
    payload.as_tensor = t;
  }

  bool isTensor() const {
    return tag == Tag::Tensor;
  }

  Tensor* toTensor() const {
    if (!isTensor()) {
      error_with_message("Value is not a tensor.");
    }
    return payload.as_tensor;
  }
};

struct ValueList {
  int size;
  Value values[];
};

} // namespace executor
} // namespace torch

