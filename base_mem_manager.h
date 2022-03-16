#pragma once
#include <core/Tensor.h>

namespace torch {
namespace executor {

// The base memory manager for the executor.
// In a typical embedded system, there can be one or more memory pools
// to store data. This manager has the sizes and base addresses of those
// memory pools.

// In the case of static memory planning, where all the data allocation
// (memory pool id and offset) can be determined offline, this manager simply
// privides the base addresses, and is only used in initialization stage.
class BaseMemManager {
public:
  BaseMemManager(int n_mems, int* sizes, uint8_t** base_addresses)
  : n_mems_(n_mems),
  sizes_(sizes),
  base_addresses_(base_addresses) {}

  // The manager holds n_mems_ memory pools.
  int n_mems_;

  // Sizes of the pools
  int* sizes_;

  // The array of base addresses for each memory pool
  uint8_t** base_addresses_;

  virtual ~BaseMemManager() {}
};
} // namespace executor
} // namespace torch
