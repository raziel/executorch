#include <operator_registry.h>
#include <unordered_map>
#include <string>

namespace torch {
namespace executor {

std::unordered_map<std::string, OpFunction>& OpsFnTable() {
  static std::unordered_map<std::string, OpFunction>
      ops_table;
  return ops_table;
}

void registerOpsFunction(
    const std::string& name,
    const OpFunction& fn) {
  OpsFnTable()[name] = fn;
}

void registerOpsFunction(Operator& o) {
  OpsFnTable()[o.name()] = o.op();
}

bool hasOpsFn(const std::string& name) {
  return OpsFnTable().count(name);
}

OpFunction& getOpsFn(const std::string& name) {
  if (!hasOpsFn(name)) {
    error_with_message("operator is not found.");
  }
  return OpsFnTable()[name];
}

} // namespace executor
} // namespace torch
