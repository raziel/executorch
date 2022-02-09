#include "executor.h"

namespace executorch {

Executor::Executor(const Program* program)
    : program_(program) {}

}  // namespace executorch