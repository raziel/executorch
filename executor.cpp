#include "executor.h"
#include <core/instruction.h>
#include <vector> // used for Tensor List initialization to make it simple, can be replaced when memory is more planned out

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

  // Load values
  n_value_ = serialization_plan_->values()->size();
  values_ = new EValue[n_value_];
  for (int i = 0; i < n_value_; ++i) {
    auto serialization_value = serialization_plan_->values()->Get(i);
    switch (serialization_value->val_type()) {
    case executorch::ValueUnion::Int: {
      values_[i].tag = Tag::Int;
      values_[i].payload.as_int = serialization_value->val_as_Int()->int_val();
    } break;
    case executorch::ValueUnion::Double: {
      values_[i].tag = Tag::Double;
      values_[i].payload.as_double = serialization_value->val_as_Double()->double_val();
    } break;
    case executorch::ValueUnion::Bool: {
      values_[i].tag = Tag::Bool;
      values_[i].payload.as_bool = serialization_value->val_as_Bool()->bool_val();
    } break;
    case executorch::ValueUnion::IntList: {
      values_[i].tag = Tag::ListInt;
      const auto items = serialization_value->val_as_IntList()->items();
      values_[i].payload.as_int_list = new ArrayRef<int64_t>(items->data(), items->size());
    } break;
    case executorch::ValueUnion::BoolList: {
      values_[i].tag = Tag::ListBool;
      const auto items = serialization_value->val_as_BoolList()->items();
      // Flatbuffer uses uint_8 to store bool types, so a quick cast to from uint8_t* to bool* is needed
      // TODO: is this safe? I think the standard is 0 is false and non 0 is true, so the simple cast is fine I think.
      values_[i].payload.as_bool_list = new ArrayRef<bool>((bool*)items->data(), items->size());
    } break;
    case executorch::ValueUnion::DoubleList: {
      values_[i].tag = Tag::ListDouble;
      const auto items = serialization_value->val_as_DoubleList()->items();
      values_[i].payload.as_double_list = new ArrayRef<double>(items->data(), items->size());
    } break;
    case executorch::ValueUnion::Tensor: {
      values_[i].tag = Tag::Tensor;
      auto s_tensor = serialization_value->val_as_Tensor();
      // TODO: use placement new
      Tensor *t = new Tensor(
          static_cast<ScalarType>(s_tensor->scalar_type()),
          s_tensor->sizes()->size(),
          const_cast<int *>(
              s_tensor->sizes()->data()),
          nullptr,
          nullptr,
          s_tensor->storage_offset());
      if (s_tensor->buffer_index() > 0) { // 0 is reserved for RW data
        auto buffer =
            program_->buffers()->GetMutableObject(s_tensor->buffer_index());
        t->data = static_cast<void *>(buffer->mutable_data()->data());
      }
      else { // TODO: init RW memory pools and do pointer mapping
        t->data = new uint8_t[t->nbytes()];
      }
      values_[i].payload.as_tensor = t;
    } break;
    case executorch::ValueUnion::TensorList: {
      values_[i].tag = Tag::ListTensor;

      // get list of serialization tensors and allocate storage for executor tensors
      auto s_tensor_list = serialization_value->val_as_TensorList()->items();
      // TODO replace std::vector with better allocation system once we have it set up
      std::vector<Tensor> executor_tensors(s_tensor_list->size());

      // for each serialization (executorch) tensor create an executor tensor and place it in the array
      for (auto s_tensor = s_tensor_list->begin(); s_tensor != s_tensor_list->end(); s_tensor_list++)  {
        executor_tensors.emplace_back(static_cast<ScalarType>(s_tensor->scalar_type()),
          s_tensor->sizes()->size(),
          const_cast<int *>(
              s_tensor->sizes()->data()),
          nullptr,
          nullptr,
          s_tensor->storage_offset()
        );

        if (s_tensor->buffer_index() > 0) { // 0 is reserved for RW data
          auto buffer =
              program_->buffers()->GetMutableObject(s_tensor->buffer_index());
          executor_tensors[i].data = static_cast<void *>(buffer->mutable_data()->data());
        }
        else { // TODO: init RW memory pools and do pointer mapping
          executor_tensors[i].data = new uint8_t[executor_tensors[i].nbytes()];
        };

        i++;
      }

      values_[i].payload.as_tensor_list = new ArrayRef<Tensor>(executor_tensors.data(), s_tensor_list->size());;
    } break;
    case executorch::ValueUnion::String: {
      // TODO requires String view;
      error_with_message("String not implemented");
    } break;
    default: // TODO: support all types
      error_with_message("type not supported");
    }
  }

  // Resolve operators
  n_operator = serialization_plan_->operators()->size();
  operators_ = new OpFunction[n_operator];
  for (int i = 0; i < n_operator; ++i) {
    std::string op_name(serialization_plan_->operators()->Get(i)->name()->str());
    operators_[i] = getOpsFn(op_name);
  }

  // Load chains
  auto chains = serialization_plan_->chains();
  n_chains_ = chains->size();
  chains_ = new Chain[n_chains_];
  for (int i = 0; i < n_chains_; ++i) {
    // load kernels
    auto kernels = chains->Get(i)->kernels();
    Chain* r_chain = &chains_[i]; // runtime chain
    r_chain->s_chain_ = chains->Get(i);
    r_chain->n_kernels_ = kernels->size();
    r_chain->kernels_ = new Kernel[r_chain->n_kernels_];
    for (int j = 0; j < r_chain->n_kernels_; ++j) {
      auto kernel = kernels->Get(j); // serialization kernel
      Kernel* r_kernel = &r_chain->kernels_[j];
      r_kernel->op_index_ = kernel->op_index();
      auto args = kernel->args();
      r_kernel->n_args_ = args->size();
      r_kernel->args_ = new EValue[r_kernel->n_args_];
      for (int k = 0; k < r_kernel->n_args_; ++k) {
        r_kernel->args_[k] = values_[args->Get(k)];
      }
    }
  }

  return 0;
}

int ExecutionPlan::execute() const {
  // V0: execute chains sequentially.
  // TODO: execute them in patterns based on (possible) control flow, delegate or async.
  // chain loo;
  for (int i = 0; i < n_chains_; ++i) {
    Chain* chain = &chains_[i];
    // instruction loop
    for (int j = 0; j < chain->s_chain_->instructions()->size(); ++j) {
      auto instruction = chain->s_chain_->instructions()->Get(j);
      switch (instruction->op()) {
      case CALL_KERNEL: {
        Kernel* kernel = &chain->kernels_[instruction->x()];
        operators_[kernel->op_index_](kernel->args_);
      } break;
      default:
        error_with_message("Instruction is not supported.");
      }
    }
  }
  return 0;
}
} // namespace executor
} // namespace torch
