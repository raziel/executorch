// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif


// ${generated_comment}

$ops_headers

namespace at = torch::executor;

namespace torch {
namespace executor {


${dispatch_anonymous_definitions}


namespace ${dispatch_namespace} {

${dispatch_namespaced_definitions}

} // namespace ${dispatch_namespace}

} // namespace executor
} // namespace torch
