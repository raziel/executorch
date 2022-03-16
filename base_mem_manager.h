#pragma once
#include <core/tensor.h>

namespace torch {
namespace executor {

// The base memory manager for the executor. 
// In a typical embedded system, there can be one or more memory pools
// to store 
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

  //
  uint8_t** base_addresses_;
};
} // namespace executor
} // namespace torch
