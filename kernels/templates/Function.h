#pragma once

// ${generated_comment}
#include <Tensor.h>
#include <Scalar.h>
#include <macros.h>

${static_dispatch_ops_headers}
namespace at = torch::executor;

namespace torch {
namespace executor {

${function_definitions}

} // namespace executor
} // namespace torch
