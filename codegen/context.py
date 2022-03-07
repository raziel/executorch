from codegen.utils import S, T, context
from codegen.model import (NativeFunction, NativeFunctionsGroup, BackendIndex, DispatchKey)
import codegen.local as local

import functools
from typing import TypeVar, Union, Iterator, Callable, Dict
import contextlib

# Helper functions for defining generators on things in the model

F = TypeVar(
    'F',
    NativeFunction,
    NativeFunctionsGroup,
    Union[NativeFunction, NativeFunctionsGroup],
)


@contextlib.contextmanager
def native_function_manager(g: Union[NativeFunctionsGroup, NativeFunction]) -> Iterator[None]:
    if isinstance(g, NativeFunctionsGroup):
        # By default, we associate all errors with structured native functions
        # with the out variant.  In some cases, it might be better to have
        # a more specific place to hang things; if so, use
        # native_function_manager again on the inside
        f = g.out
    else:
        f = g
    with context(lambda: f'in native_functions.yaml line {f.loc}:\n  {f.func}'):
        with local.parametrize(use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors):
            yield


def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        with native_function_manager(f):
            return func(slf, f)
    return wrapper

# Convenience decorator for functions that explicitly take in a BackendIndex,
# instead of indirectly taking one in as a closure
def with_native_function_and_index(func: Callable[[F, BackendIndex], T]) -> Callable[[F, BackendIndex], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        with native_function_manager(f):
            return func(f, backend_index)
    return wrapper