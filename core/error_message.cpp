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
  throw std::runtime_error(message);
  // exit(1); // this line doesnt actually cause tests to fail so switched to the above for now.
}
} // namespace executor
} // namespace torch
