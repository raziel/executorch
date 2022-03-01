#pragma once

#include <Evalue.h>
#include <functional>
#include <vector>

namespace torch {
namespace executor {

using OpFunction = std::function<void(EValue*)>;

void registerOpsFunction(
    const std::string& name,
    const OpFunction& fn);

bool hasOpsFn(const std::string& name);

OpFunction& getOpsFn(const std::string& name);

class op_fn_register {
 public:
  op_fn_register(
      const std::string& name,
      const OpFunction& fn) {
    registerOpsFunction(name, fn);
  }
};

} // namespace executor
} // namespace torch
