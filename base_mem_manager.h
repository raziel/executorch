#pragma once
#include <core/tensor.h>

namespace torch {
namespace executor {

// The base memory manager for the executor. 
// In a typical embedded system, there can be one or more memory pools
// to store 
class BaseMemManager {

  BaseMemManager(int n_mems, int* sizes, void** base_addresses)
  : n_mems_(n_mems),
  sizes_(sizes),
  base_addresses_(base_addresses) {}
  
  // The manager holds n_mems_ memory pools. 
  int n_mems_;

  // Sizes of the pools
  int* sizes_;

  //
  void** base_addresses_;
};
} // namespace executor
} // namespace torch
