#include "executor.h"

namespace torch {
namespace executor {

Executor::Executor(const executorch::Program* program)
    : program_(program), plan_(program) {}

int Executor::init_execution_plan(int index) {
  auto serialization_plan = program_->execution_plan()->GetMutableObject(index);
  return plan_.init(serialization_plan);
}

int ExecutionPlan::init(executorch::ExecutionPlan* s_plan) {
  serialization_plan_ = s_plan;
  nvalue_ = s_plan->values()->size();
  values_ = new Value[nvalue_];
  for (int i = 0; i < nvalue_; ++i) {
    auto serialization_value = s_plan->values()->Get(i);
    switch (serialization_value->val_type()) {
    case executorch::ValueUnion::Int: {
      values_[i].tag = Tag::Int;
      values_[i].payload.as_int = serialization_value->val_as_Int()->int_val();
    } break;
    case executorch::ValueUnion::Tensor: {
      values_[i].tag = Tag::Tensor;
      auto s_tensor = serialization_value->val_as_Tensor();
      // TODO: use placement new
      Tensor *t = new Tensor(
          static_cast<ScalarType>(s_tensor->scalar_type()),
          s_tensor->sizes()->size(),
          const_cast<int *>(
              s_tensor->sizes()->data()));
      if (s_tensor->buffer_index() > 0) { // 0 is reserved for RW data
        auto buffer =
            program_->buffers()->GetMutableObject(s_tensor->buffer_index());
        t->data = static_cast<void *>(buffer->mutable_data()->data());
      }
      else { // TODO: init RW memory pools and do pointer mapping
        t->data = new uint8_t[t->nbytes];
      }
      values_[i].payload.as_tensor = t;
    } break;
    default: // TODO: support all types
      error_with_message("type not supported");
    }
  }
  
}

} // namespace executor
} // namespace torch
