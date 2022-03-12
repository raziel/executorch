#pragma once
// ${generated_comment}

// NB: The implementing C++ file is RegisterDispatchKey.cpp
#include <tensor.h>
#include <macros.h>
// The only #includes we need are for custom classes that have defaults in the C++ API
namespace at = torch::executor;


namespace torch {
namespace executor {

namespace ${dispatch_namespace} {

${dispatch_namespaced_declarations}

} // namespace ${dispatch_namespace}
} // namespace executor
} // namespace torch
