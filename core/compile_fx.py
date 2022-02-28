import torch
from typing import List

import torch
from typing import List

class Kernel():
    def __init__(self, op_index: int, args: List[int]):
        self.op_index = op_index
        self.args = args

    def __repr__(self):
        return "Kernel with op_index {} and args {}".format(self.op_index, self.args)

class Chain():
    def __init__(self, kernels: List[Kernel], inputs: List[int], outputs: List[int]):
        self.kernels = kernels
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return "Kernels: {}\n".format(self.kernels.__repr__()) + "Inputs: {}\n".format(self.inputs) + "Outputs: {}".format(self.outputs)

class Operator():
    def __init__(self, name, overload):
        self.name = name
        self.overload = overload

    def __repr__(self):
        return "op: {}.{}".format(self.name, self.overload)

class ExecutionPlan():
    def __init__(self, inputs, outputs, chains, operators):
        self.inputs = inputs
        self.outputs = outputs
        self.chains = chains
        self.operators = operators

# TODO: will need to deduplicate op table and values
def compile_to_exec_plan(graph):
    ops = []
    arguments = []
    kernels = []
    node_to_idx = {}
    def get_index(node):
        if node not in node_to_idx:
            node_to_idx[node] = get_index.i
            get_index.i += 1
        return node_to_idx[node]
    get_index.i = 0
    for ix, node in enumerate(graph.nodes):
        args_for_kernel = []
        if node.op == "call_function":
            # TODO Hacky way to extract op name
            op_name = node.target.__name__
            # TODO This will be replaced with actual value
            # when we can access op per overload
            op_overload = ""
            operator = Operator(name=op_name, overload=op_overload)
            ops.append(operator)

            for arg in node.args:
                arguments.append(arg)
                args_for_kernel.append(get_index(arg))
                print("Arg {} with index {}".format(arg, get_index.i))

            for kwarg in node.kwargs:
                arguments.append(kawarg)
                kwarg_ix = len(arguments)
                args_for_kernel.append(kwarg_ix)
                print("Kwarg {} with index {}".format(kwarg, get_index.i))

            kernel = Kernel(op_index=len(ops), args=args_for_kernel)
            kernels.append(kernel)
        else:
            # There shouldn't be any other type of FX op
            continue

    # TODO this is probably not true.
    chain = Chain(kernels=kernels, inputs=[v for _, v in node_to_idx.items()], outputs=[v for _, v in node_to_idx.items()])
    return chain, ops, arguments

if __name__ == "__main__":

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.add(x, torch.Tensor([5.0])), torch.add(x, torch.Tensor([6.0]))

    module = MyModule()

    from torch.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

    print(symbolic_traced.graph)
    chain, ops, arguments = compile_to_exec_plan(symbolic_traced.graph)
    print(chain)
