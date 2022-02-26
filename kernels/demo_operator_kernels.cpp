#include <core/operator_registry.h>
#include <string>

namespace torch {
namespace executor {

// kernel for demonstration purpose only

// Kernel implementation provided by user.
// The schema is added by user to PyTorch native function DSL in a yaml file,
// defined in
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md
void add_kernel(Tensor &a, Tensor &b, Tensor &c) {
  auto data_a = static_cast<int *>(a.data);
  auto data_b = static_cast<int *>(b.data);
  auto data_c = static_cast<int *>(c.data);
  int n_elements = 1;
  for (int i = 0; i < a.dim; ++i) {
    n_elements *= a.sizes[i];
  }
  for (int i = 0; i < n_elements; ++i) {
    data_c[i] = data_a[i] + data_b[i];
  }
}

// Code-generated glue unbox wrapper
// TODO: provide functions.yaml and the code-gen implementation
void add_op(Value *args) {
  Tensor *a = args[0].toTensor();
  Tensor *b = args[1].toTensor();
  Tensor *c = args[2].toTensor();
  add_kernel(*a, *b, *c);
}

void mul_kernel(Tensor &a, Tensor &b, Tensor &c) {
  auto data_a = static_cast<int *>(a.data);
  auto data_b = static_cast<int *>(b.data);
  auto data_c = static_cast<int *>(c.data);
  int n_elements = 1;
  for (int i = 0; i < a.dim; ++i) {
    n_elements *= a.sizes[i];
  }
  for (int i = 0; i < n_elements; ++i) {
    data_c[i] = data_a[i] * data_b[i];
  }
}

void mul_op(Value *args) {
  Tensor *a = args[0].toTensor();
  Tensor *b = args[1].toTensor();
  Tensor *c = args[2].toTensor();
  mul_kernel(*a, *b, *c);
}

static const std::vector<op_fn_register> op_reg{
    op_fn_register("demo::add", add_op), op_fn_register("demo::mul", mul_op)};

} // namespace executor
} // namespace torch
