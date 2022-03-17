#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

namespace torch {
namespace executor {

// TODO: move to error handler
void error_with_message(const char* message) {
  // A hacky error function before we have a good convention,
  // better without exception.
  printf("%s\n", message);
#if defined(__linux__) || defined(__APPLE__)
  throw std::runtime_error(message);
#else
  // Exception handling is disabled on embedded targets so we use this workaround
  // for now until error handling is improved.
  exit(1);
#endif
}
} // namespace executor
} // namespace torch
