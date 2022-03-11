#pragma once

namespace torch {
namespace executor {

// TODO: move to error handler
[[ noreturn ]] void error_with_message(const char* message);

} // namespace executor
} // namespace torch
