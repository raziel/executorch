#include <core/operator_registry.h>
#include <string>

namespace torch {
namespace executor {

// kernel for demonstration purpose only
void add_kernel(Value* args) {
  Tensor* a = args[0].toTensor();
  Tensor* b = args[1].toTensor();
  Tensor* c = args[2].toTensor();
  auto data_a = static_cast<int*>(a->data);
  auto data_b = static_cast<int*>(b->data);
  auto data_c = static_cast<int*>(c->data);
  int n_elements = 1;
  for (int i = 0; i < a->dim; ++i) {
    n_elements *= a->sizes[i];
  }
  for (int i = 0; i < n_elements; ++i) {
    data_c[i] = data_a[i] + data_b[i];
  }
}

void mul_kernel(Value* args) {
  Tensor* a = args[0].toTensor();
  Tensor* b = args[1].toTensor();
  Tensor* c = args[2].toTensor();
  auto data_a = static_cast<int*>(a->data);
  auto data_b = static_cast<int*>(b->data);
  auto data_c = static_cast<int*>(c->data);
  int n_elements = 1;
  for (int i = 0; i < a->dim; ++i) {
    n_elements *= a->sizes[i];
  }
  for (int i = 0; i < n_elements; ++i) {
    data_c[i] = data_a[i] * data_b[i];
  }
}

static const std::vector<op_fn_register> op_reg{
    op_fn_register("demo::add", add_kernel),
    op_fn_register("demo::mul", add_kernel)
};

} // namespace executor
} // namespace torch

