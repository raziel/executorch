import torch
from typing import List

import torch
from typing import List

class Kernel():
    def __init__(self, op_index: int, args: List[int], outputs: List[int]):
        self.op_index = op_index
        self.args = args
        self.outputs = outputs

    def __repr__(self):
        return "Kernel with op_index {} and args {} and outputs {}".format(self.op_index, self.args, self.outputs)

class Chain():
    def __init__(self, kernels: List[Kernel]):
        self.kernels = kernels

    def __repr__(self):
        return self.kernels.__repr__()

class Operator():
    def __init__(self, name, overload):
        self.name = name
        self.overload = overload

    def __repr__(self):
        return "op: {}.{}".format(self.name, self.overload)

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
                print('arg', arg)
                arguments.append(arg)
                args_for_kernel.append(get_index(arg))

            for kwarg in node.kwargs:
                print('kwargs', kwargs)
                arguments.append(kawarg)
                kwarg_ix = len(arguments)
                args_for_kernel.append(kwarg_ix)

            kernel = Kernel(op_index=len(ops), args=args_for_kernel, outputs=get_index(node))
            kernels.append(kernel)
        else:
            # There shouldn't be any other type of FX op
            continue

    chain = Chain(kernels=kernels)
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
    print(compile_to_exec_plan(symbolic_traced.graph))
