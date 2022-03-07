#pragma once
#include <cstdint>
#include <ArrayRef.h>

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
// TODO: safety checks
class Tensor {

  public:

    void* data = nullptr;

    Tensor() {}
    Tensor(ScalarType type, int dim, int* sizes, void* data=nullptr, int* strides = nullptr)
    : dim_(dim), sizes_(sizes, dim), type_(type), data(data), strides_(strides)
    {
      if (!data) {
        return;
      }
      numel_ = compute_numel();
      nbytes_ = numel_ * scalarTypeItemSizes[static_cast<uint16_t>(type_)];
    }

    size_t nbytes() const {
      return nbytes_;
    }

    int size(int dim) const {
      return sizes_[dim];
    }

    int dim() const {
      return dim_;
    }

    int numel() const {
      return numel_;
    }

    const ScalarType& dtype() const {
      return type_;
    }

    // Return the size of one element of the tensor
    int element_size() const{
      return scalarTypeItemSizes[static_cast<int>(type_)];
    }

    utils::ArrayRef<int> sizes() {
      return sizes_;
    }
    const utils::ArrayRef<int> sizes() const {
      return sizes_;
    }

  private:

    ScalarType type_;
    int dim_ = 0;
    int nbytes_ = 0;
    utils::ArrayRef<int> sizes_;
    int* strides_ = nullptr;
    int numel_ = 0;

    /**
    * Compute the number of elements based on the sizes of a tensor.
    */
    int compute_numel() const {
      int n = 1;
      for (auto s : sizes()) {
        n *= s;
      }
      return n;
    }
};

} // namespace executor
} // namespace torch
