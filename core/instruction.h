#pragma once

#include <cstdint>

namespace torch {
namespace executor {
// instruction look like:
// op_code X, N
// op_code: 8-bit int
// X: 32-bit int
// N: 8-bit int for padding (reserved for possible future extension)

// meaning of X, N depend on the op:
// K - index of the kernel table
// P - jump offset relative to beginning of current instruction


#define FORALL_OPCODES(_)                                                      \
  _(CALL_KERNEL, "K") /* invoke kernel K */                                           \
  _(JF, "P") /* pop the top of the stack, if false, branch to P */

enum OpCode : uint8_t {
#define DEFINE_OP(op, _) op,
  FORALL_OPCODES(DEFINE_OP)
#undef DEFINE_OP
};

struct Instruction {
  OpCode op;
  int32_t X;
  uint8_t N;
  Instruction(OpCode op, int32_t X, uint8_t N)
      : op(op), X(X), N(N)  {}
};
} // namespace jit
} // namespace torch
