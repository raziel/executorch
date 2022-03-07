from codegen.model import (FunctionSchema,
                           DispatchKey)


def schema_kernel_name(func: FunctionSchema, dispatch_key: DispatchKey) -> str:
    assert func.is_out_fn(), "ufunc.kernel_name should only be invoked on out schemas"
    return f"ufunc_{func.name.name}_{dispatch_key}"
