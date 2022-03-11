#pragma once

#include <Evalue.h>
#include <functional>
#include <vector>
#include <string>

namespace torch {
namespace executor {

using OpFunction = std::function<void(EValue*)>;
struct Operator {
 private:
 OpFunction op_;
 std::string name_;

 public:
  Operator(std::string name, OpFunction func): op_(func), name_(name) {}
  const std::string name() const {
    return name_;
  }
  const OpFunction op() const {
    return op_;
  }
};

void registerOpsFunction(
    const std::string& name,
    const OpFunction& fn);

void registerOpsFunction(Operator& o);

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

struct RegisterOperators {
  RegisterOperators() = default;

  explicit RegisterOperators(std::vector<Operator>& operators) {
    for (Operator& o : operators) {
      registerOpsFunction(o);
    }
  }
};
} // namespace executor
} // namespace torch
