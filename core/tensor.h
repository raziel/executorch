#pragma once
#include <cstdint>

namespace c10 {
struct alignas(1) qint8 {
  using underlying = int8_t;
  int8_t val_;
  qint8() = default;
  explicit qint8(int8_t val) : val_(val) {}
};

}

namespace torch {
namespace executor {
// copied from pytorch/c10/core/ScalarType.h.
// TODO: reuse if possible
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(bool, Bool) /* 11 */

// TODO: include necessary types from c10 (e.g. quantized)
//  _(at::Half, Half) /* 5 */                              \
//  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \
//  _(c10::complex<float>, ComplexFloat) /* 9 */           \
//  _(c10::complex<double>, ComplexDouble) /* 10 */        \
//  _(c10::qint8, QInt8) /* 12 */                          \
//  _(c10::quint8, QUInt8) /* 13 */                        \
//  _(c10::qint32, QInt32) /* 14 */                        \
//  _(at::BFloat16, BFloat16) /* 15 */                     \
//  _(c10::quint4x2, QUInt4x2) /* 16 */                    \
//  _(c10::quint2x4, QUInt2x4) /* 17 */

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
#undef DEFINE_ENUM
      Undefined,
  NumOptions
};

constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

static constexpr uint8_t scalarTypeItemSizes[NumScalarTypes] = {
#define SCALAR_TYPE_SIZE(T, name) sizeof(T),
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_SIZE)
#undef SCALAR_TYPE_SIZE
        0, // Undefined
};


// Tensor should be simple.
// The requirement of tensor is to be sable in operators. It means
// 1. It cannot be the serialization tensor where the methods are limited.
//    However, the data ptr should be mapped from serialization, to avoid
//    unnecessary memory copy.
// 2. It should have all APIs used in operator kernels.
// TODO: APIs common to at::Tensor
struct Tensor {
  ScalarType type;
  void* data = nullptr;
  int dim = 0;
  int nbytes = 0;
  int* sizes = nullptr;
  int* strides = nullptr;
  Tensor() {}
  Tensor(ScalarType type, int dim, int* sizes, void* data=nullptr, int* strides = nullptr)
  : dim(dim), sizes(sizes), type(type), data(data), strides(strides)
  {
    if (!data) {
      nbytes = 0;
      return;
    }
    nbytes = 1;
    for (int i = 0; i < dim; ++i) {
      nbytes *= sizes[i];
    }
    nbytes *= scalarTypeItemSizes[static_cast<uint16_t>(type)];
  }
  // TODO: Quantizer
};

} // namespace executor
} // namespace torch

