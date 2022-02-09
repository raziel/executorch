#include "flatbuffers/flatbuffers.h"
#include "schema/schema_generated.h"

namespace executorch {

class Executor {

 public:

  // Executes a PyTorch executor program.
  explicit Executor(const Program* program);

  ~Executor();

 private:

  const Program* program_;
};

}  // namespace executorch