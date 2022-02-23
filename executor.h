#include <schema/schema_generated.h>
#include <core/tensor.h>
#include <core/value.h>

// namespace "executorch" is reserved for serialization
namespace torch {
namespace executor {

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
