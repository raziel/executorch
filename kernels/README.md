## Context
The `functions.yaml` file follows the same DSL schema as `native_functions.yaml` in PyTorch main repo. The differences 
lie on the boxed variable design (IValue vs Value) and tensor design. In the developing phase, since both kernels and 
Tensor design are fast changing, codegen unboxing will be used to connect boxed and unboxed kernels, due to its flexibility.
## Support features
* Conversion from `Value` to C++ argument types:
  * int
  * Tensor