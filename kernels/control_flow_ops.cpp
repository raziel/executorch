#include <core/operator_registry.h>
#include <executor.h>
#include <string>

namespace torch {
namespace executor {

bool runCondAndGetResult(const Chain& cond_chain) {
  // Run the condition chain
  cond_chain.run();

  // Get the result (bool, can be extended to int for case)
  return cond_chain.outputs[0]->toBool();
}

void if_kernel(Context *context, int cond_chain_id, int then_chain_id,
               int else_chain_id) {
  ExecutionPlan *plan = context->getExecutionPlan();

  // Get the condition chain
  const Chain &cond_chain = plan->chains_[cond_chain_id];

  if (runCondAndGetResult(cond_chain)) {
    // Run the Then chain
    plan->chains_[then_chain_id]->run();
  } else {
    // Run the Else chain
    plan->chains_[else_chain_id]->run();
  }
}

void if_op(Value *args) {
  Context *context = args[0].toContext();
  int cond_chain_id = args[1].toInt();
  int then_chain_id = args[2].toInt();
  int else_chain_id = args[3].toInt();

  if_kernel(context, cond_chain_id, then_chain_id, else_chain_id);
}

void while_kernel(Context *context, int cond_chain_id, int body_chain_id) {
  ExecutionPlan *plan = context->getExecutionPlan();

  // Get the condition chain
  const Chain &cond_chain = plan->chains_[cond_chain_id];

  while (runCondAndGetResult(cond_chain)) {
    // Run the Body chain
    plan->chains_[body_chain_id]->run();
  }
}

void while_op(Value *args) {
  Context *context = args[0].toContext();
  int cond_chain_id = args[1].toInt();
  int body_chain_id = args[2].toInt();

  while_kernel(context, cond_chain_id, body_chain_id);
}

static const std::vector<op_fn_register> cntrl_op_reg{
    op_fn_register("cntrl::if", if_op), op_fn_register("cntrl::while", while_op)};

} // namespace executor
} // namespace torch
