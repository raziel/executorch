#include <schema/schema_generated.h>
#include <core/tensor.h>
#include <core/value.h>
#include <core/operator_registry.h>

// namespace "executorch" is reserved for serialization
namespace torch {
namespace executor {

// Trying to avoid std libs to reduce size, dependency and increase portability.
// A number of (num, pointer*) structures.
// TODO: consider using liteweighted header-only structures, like span
// martinmoene/span-lite requires C++98 or later
// tcbrindle/span requires C++11 or later

// Kernal and Chain in executor (runtime) namespace. The purpose of a runtime structure
// instead of re-using flatbuffer data, is to aggregate args, which can be done
// one-time at loading stage.
struct Kernel {
  int n_args_;
  Value* args_;
  // Index to the op functor table
  int op_index_;
};

struct Chain {
  int n_kernels_;
  Kernel* kernels_;
};

// ExecutionPlan in executor (runtime) namespace.
// Differences from executorch::ExecutionPlan in serialization:
// It holds values with APIs that are compatible operator unboxing.
// The data pointers of the values should be mapped to serialization buffer.
// It holds function pointers of kernels, instead of operator names.
// TODO: use static memory planning to create all executor related data
struct ExecutionPlan {
  explicit ExecutionPlan(const executorch::Program* program) : program_(program) {}
  int init(executorch::ExecutionPlan* s_plan);
  const executorch::Program* program_;
  executorch::ExecutionPlan* serialization_plan_;

  int n_value_;
  Value* values_;

  int n_operator;
  OpFunction* operators_;

  int n_chains_;
  Chain* chains_;
};

class Executor {

 public:

  // Executes a PyTorch executor program.
  explicit Executor(const executorch::Program* program);

  int init_execution_plan(int index);

  const ExecutionPlan& executionPlan() {return plan_;}

  ~Executor() {}

 private:

  const executorch::Program* program_;
  ExecutionPlan plan_;
};

} // namespace executor
} // namespace torch
