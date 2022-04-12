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

def compile_to_exec_plan(graph):
    ops = []
    kernels = []
    node_to_idx = {}
    outputs = []
    nodes_in_chain = set()
    def get_index(node):
        if node not in node_to_idx:
            node_to_idx[node] = get_index.i
            get_index.i += 1
        return node_to_idx[node]
    get_index.i = 0

    for ix, node in enumerate(graph.nodes):
        args_for_kernel = []
        if node.op == "output":
            return_args = node.args[0]
            if not isinstance(return_args, List):
                outputs.append((return_args, get_index(return_args)))
            else:
                for arg in return_args:
                    outputs.append((arg, get_index(arg)))

        if node.op == "call_function":
            # TODO Hacky way to extract op name
            op_name = node.target.__name__
            # TODO This will be replaced with actual value
            # when we can access op per overload
            op_overload = ""
            operator = Operator(name=op_name, overload=op_overload)
            ops.append(operator)

            for arg in node.args:
                args_for_kernel.append(get_index(arg))

            for kwarg in node.kwargs:
                kwarg_ix = len(arguments)
                args_for_kernel.append(kwarg_ix)

            kernel = Kernel(op_index=len(ops), args=args_for_kernel)
            kernels.append(kernel)
            nodes_in_chain.add(node)
        # TODO Since we are developing on actual FX graph, we should
        # not error out in cases we don't support yet
        else:
            # There shouldn't be any other type of FX op
            continue

    arguments_node = [k for k, v in node_to_idx.items()]
    arguments_index = [v for k, v in node_to_idx.items()]

    # To make lookup easier
    output_nodes = set()
    for output_node, output_idx in outputs:
        output_nodes.add(output_node)

    # Chain inputs are:
    #   1. not output
    #   2. not a node in chain
    chain_inputs = []
    for node in node_to_idx:
        if (node in nodes_in_chain) or (node in output_nodes):
            continue
        chain_inputs.append(node_to_idx[node])

    chain = Chain(kernels=kernels, inputs=chain_inputs, outputs=[v for k, v in outputs])
    return chain, ops, arguments_node

if __name__ == "__main__":

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            a = torch.add(x, torch.Tensor([5.0]))
            b = torch.add(torch.Tensor([2.0]), torch.Tensor([3.0]))
            c = torch.add(x + b, torch.Tensor([6.0]))
            return c

    module = MyModule()

    from torch.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

    print(symbolic_traced.graph)
    chain, ops, arguments = compile_to_exec_plan(symbolic_traced.graph)
    print(chain) # chain of Kernels
    print(ops) # op look up table
    print(arguments) # value look up table
