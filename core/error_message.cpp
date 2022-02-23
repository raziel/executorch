#pragma once
#include <stdio.h>
#include <stdlib.h>

namespace torch {
namespace executor {

// TODO: move to error handler
void error_with_message(char* message) {
  // A hacky error function before we have a good convention,
  // better without exception.
  printf("%s\n", message);
  exit(1);
}
} // namespace executor
} // namespace torch

