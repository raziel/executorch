import argparse
import os
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Sequence, Optional, Union, Set, Literal, OrderedDict

import yaml
from collections import namedtuple, defaultdict

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
from codegen.api import meta, structured, unboxing
from codegen.api.types import DispatcherSignature, CppSignature, CppSignatureGroup
from codegen.api.unboxing import convert_arguments
from codegen.context import method_with_native_function, native_function_manager
from codegen.model import NativeFunction, DispatchKey, OperatorName, BackendMetadata, Location, BackendIndex, \
    is_cuda_dispatch_key, BaseOperatorName, Tag, NativeFunctionsGroup, Variant, SchemaKind, FunctionSchema, \
    is_generic_dispatch_key
from codegen.selective_build.selector import SelectiveBuilder
from codegen.api.translate import translate
from codegen.utils import context, YamlLoader, make_file_manager, FileManager, mapMaybe, assert_never, Target, concatMap
from codegen import dest


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}

ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])

# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file
class LineLoader(YamlLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        # Add 1 so line numbering starts at 1
        mapping['__line__'] = node.start_mark.line + 1
        return mapping


def cpp_string(s: str) -> str:
    """Convert a python string into a c++ string literal """
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\a', '\\a')
    s = s.replace('\b', '\\b')
    s = s.replace('\f', '\\f')
    s = s.replace('\n', '\\n')
    s = s.replace('\v', '\\v')
    s = s.replace('\t', '\\t')
    return f'"{s}"'


def static_dispatch_keys(backend: Optional[BackendIndex]) -> List[DispatchKey]:
    if backend is None:
        return []
    else:
        return [
            backend.dispatch_key,
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeExplicitAutograd
        ]


def get_static_dispatch_backend(f: NativeFunction, backend_index: BackendIndex) -> Optional[DispatchKey]:
    if (f.structured_delegate is not None or backend_index.has_kernel(f)):
        # TODO: for ops with structured_delegate it should check the dispatch table of
        # the out variant instead. For now, these structured ops all have CPU/CUDA kernels
        # so we always dispatch to the `backend`, but this could be wrong when we
        # migrate math/default_backend ops to use structured delegate.
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    return None


def static_dispatch_ops_header(
        f: NativeFunction,
        backend_index: Optional[BackendIndex]) -> Optional[str]:
    if backend_index is None or f.manual_kernel_registration:
        return None

    dispatch_key = get_static_dispatch_backend(f, backend_index)
    return (f'#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>'
            if dispatch_key is not None else None)


def static_dispatch_extra_headers(backend: Optional[BackendIndex], skip_tensor_include: bool = False) -> List[str]:
    if skip_tensor_include:
        # See Note [Avoiding Include Cycles In Static Dispatch]
        maybe_inl = '_inl'
    else:
        maybe_inl = ''
    return [f'#include <ATen/{dispatch_key}Functions{maybe_inl}.h>'
            for dispatch_key in static_dispatch_keys(backend)]



def static_dispatch(
        f: NativeFunction, cpp_sig: CppSignature,
        *, method: bool, backend_index: Optional[BackendIndex]
) -> Optional[str]:
    if backend_index is None or f.manual_kernel_registration:
        return None
    target_sig = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False).signature
    name = target_sig.name()
    exprs = translate(cpp_sig.arguments(), target_sig.arguments(), method=method)
    exprs_str = ', '.join(a.expr for a in exprs)

    dispatch_key = get_static_dispatch_backend(f, backend_index)
    if dispatch_key is not None:
        return f'return at::{dispatch_key.lower()}::{name}({exprs_str});'

    return f'TORCH_CHECK(false, "Static dispatch does not support {name} for {backend_index.dispatch_key}.");'


# Generates Operators.h and Operators.cpp.
# These provide macros that, given an operator and overload name, allow users
# to access an "un-overloaded" function version of the operator. This
# is useful for extension writers who want to (1) want to decltype the operator
# and (2) don't want to worry about method-only operators.
@dataclass(frozen=True)
class ComputeOperators:
    target: Union[
        Literal[Target.DECLARATION],
        Literal[Target.DEFINITION]
    ]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()
        call_method_name = 'call'
        redispatch_method_name = 'redispatch'

        if self.target is Target.DECLARATION:
            # Note [The ATen Operators API]
            # The ATen Operators API lives in the at::_ops namespace, and contains compile-time
            # metadata about each operator + entry points into the Dispatcher.
            # The C++ function, method, and redispatch API's are all implemented as wrappers
            # into various bits of the structs defined here.
            #
            # Important characteristics about the Operators API:
            # (1) It follows the Dispatcher API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Overload names are disambiguated.
            #     This is helpful for pytorch extenders who would like to decltype() an aten operator,
            #     that has overloads, e.g. decltype(at::_ops::mul_Tensor::call)
            # (3) No argument defaulting is allowed.
            #     This is more of an implementation detail to avoid #include cycles,
            #     since TensorBody.h (which defines the Tensor class) needs to include this file.
            # (4) manual_cpp_bindings and faithful names are not included in the API.
            #     This applies to stuff like __dispatch__is_complex(), and add_outf().
            #     These aren't "real aten ops", they're just additional functions provided by the C++ API.
            #     They're implemented as wrappers in Functions.h that call into the actual operators
            #     defined here, i.e. at::_ops::is_complex::call() and at::_ops::add_out::call().
            #     This means that ATEN_OP(is_complex) will not fastpath, and will go through the dispatcher.
            return f"""
struct TORCH_API {name} {{
  using schema = {sig.type()};
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})
  static {sig.defn(name=call_method_name, is_redispatching_fn=False)};
  static {sig.defn(name=redispatch_method_name, is_redispatching_fn=True)};
}};"""
        elif self.target is Target.DEFINITION:
            defns = f"""
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name, "aten::{f.func.name.name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, overload_name, "{f.func.name.overload_name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str, {cpp_string(str(f.func))})

// aten::{f.func}
static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow({name}::name, {name}::overload_name)
      .typed<{name}::schema>();
}}
"""

            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ', '.join(['dispatchKeySet'] + [a.name for a in sig.arguments()])
                    dispatcher_call = 'redispatch'
                    method_name = f'{name}::{redispatch_method_name}'
                else:
                    dispatcher_exprs_str = ', '.join([a.name for a in sig.arguments()])
                    dispatcher_call = 'call'
                    method_name = f'{name}::{call_method_name}'

                defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    static auto op = create_{name}_typed_handle();
    return op.{dispatcher_call}({dispatcher_exprs_str});
}}
"""
            return defns
        else:
            assert_never(self.target)



# Generates Functions.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    static_dispatch_backend_index: Optional[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.function not in f.variants:
            return None

        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)

        def generate_defn(faithful: bool) -> str:
            if faithful:
                sig = sig_group.faithful_signature
                assert sig is not None
            else:
                sig = sig_group.signature

            # See Note [The ATen Operators API]
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ', '.join([e.expr for e in exprs])

            static_dispatch_block = static_dispatch(f, sig, method=False, backend_index=self.static_dispatch_backend_index)
            if static_dispatch_block is None:
                return f"""
// aten::{f.func}
TORCH_API inline {sig.decl()} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""
            else:
                return f"""
// aten::{f.func}
TORCH_API inline {sig.decl()} {{
    {static_dispatch_block}
}}
"""
        result = generate_defn(False)
        if sig_group.faithful_signature is not None:
            result += generate_defn(True)

        return result



# Generates TensorBody.h. This file provides the object-oriented (method-based)
# public C++ API, and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Union[
        Literal[Target.DECLARATION],
        Literal[Target.DEFINITION]
    ]
    static_dispatch_backend_index: Optional[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None

        sig_group = CppSignatureGroup.from_native_function(f, method=True, fallback_binding=f.manual_cpp_binding)

        if self.target is Target.DECLARATION:
            result = f"{sig_group.signature.decl()} const;\n"
            if sig_group.faithful_signature is not None:
                result += f"{sig_group.faithful_signature.decl()} const;\n"
            return result

        if self.target is not Target.DEFINITION:
            assert_never(self.target)

        def generate_defn(faithful: bool) -> str:
            if faithful:
                sig = sig_group.faithful_signature
                assert sig is not None
            else:
                sig = sig_group.signature

            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ', '.join([e.expr for e in exprs])

            static_dispatch_block = static_dispatch(f, sig, method=True, backend_index=self.static_dispatch_backend_index)
            if static_dispatch_block is None:
                return f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""
            else:
                return f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    {static_dispatch_block}
}}
"""

        result = generate_defn(faithful=False)
        if sig_group.faithful_signature is not None:
            result += generate_defn(faithful=True)

        return result


# Generates UnboxingFunctions.h & UnboxingFunctions.cpp.
@dataclass(frozen=True)
class ComputeUnboxingFunctions:
    target: Union[Literal[Target.DECLARATION], Literal[Target.DEFINITION]]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:

        if self.target is Target.DECLARATION:
            # Note [The ATen Codegen Unboxing API]
            # Similar to the ATen Operators API, ATen Codegen Unboxing API lives in the at::unboxing namespace, and
            # will be used by codegen unboxing wrappers (CodegenUnboxingWrappers.cpp).
            # The Wrappers will be registered into torch::jit::OperatorRegistry using RegisterOperators API.
            #
            # Important characteristics about the Codegen Unboxing API:
            # (1) It follows the OperatorRegistry API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Under the hood it calls C++ API.
            return f"""
// aten::{f.func}
TORCH_API void {f.func.name.unambiguous_name()}(Stack & stack);
"""
        else:
            sig_group = CppSignatureGroup.from_native_function(
                f, method=(Variant.method in f.variants)
            )
            sig = sig_group.most_faithful_signature()
            # parse arguments into C++ code
            binding_list, code_list = convert_arguments(f)

            # for each C++ argument, generate the conversion code
            code_connector = "\n\t"
            arg_connector = ", "
            # function call and push back to stack
            prefix = "self_base." if sig.method else "at::"
            translated_args = translate(binding_list, sig.arguments(), method=sig.method)
            args_str = f"{arg_connector.join(e.expr for e in translated_args)}"
            if len(f.func.returns) == 0:
                ret_str = ""
                push_str = ""
            else:
                ret_str = "auto result_ = "
                push_str = """
    pack(stack, std::move(result_));
                """
            return f"""
// aten::{f.func}
TORCH_API void {f.func.name.unambiguous_name()}(Stack & stack) {{
    {code_connector.join(code_list)}

    drop(stack, {len(binding_list)});

    {ret_str}{prefix}{sig.name()}({args_str});
    {push_str}
}}
"""


# Generates RegisterCodegenUnboxedKernels.cpp.
@dataclass(frozen=True)
class ComputeCodegenUnboxedKernels:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        # We unconditionally generate function wrappers,
        sig_group = CppSignatureGroup.from_native_function(
            f, method=(Variant.method in f.variants)
        )

        sig = sig_group.most_faithful_signature()

        # escape double quote in schema, get rid of extra double quotes
        schema = cpp_string(str(sig.func))[1:-1]

        return f"""
OperatorGenerator(
    TORCH_SELECTIVE_SCHEMA("aten::{schema}"),
    [](Stack & stack) {{
        RECORD_FUNCTION("{sig.name()}", std::vector<c10::IValue>());
        at::unboxing::{unboxing.name(f)}(stack);
    }},
    aliasAnalysisFromSchema()
),
"""


def gen_unboxing(
        *,
        native_functions: Sequence[NativeFunction],
        cpu_fm: FileManager,
) -> None:
    def key_func(fn: Union[NativeFunction, NativeFunctionsGroup]) -> str:
        return fn.root_name

    cpu_fm.write_sharded(
        "UnboxingFunctions.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "definitions": [ComputeUnboxingFunctions(Target.DEFINITION)(fn)]
        },
        num_shards=5,
        sharded_keys={"definitions"},
    )
    cpu_fm.write(
        "UnboxingFunctions.h",
        lambda: {
            "declarations": list(
                mapMaybe(ComputeUnboxingFunctions(Target.DECLARATION), native_functions)
            ),
        },
    )
    cpu_fm.write_sharded(
        "RegisterCodegenUnboxedKernels.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {"unboxed_ops": [ComputeCodegenUnboxedKernels()(fn)]},
        num_shards=5,
        sharded_keys={"unboxed_ops"},
    )


# Generates MetaFunctions.h
def compute_meta_function_declaration(g: NativeFunctionsGroup) -> Optional[str]:
    if not g.structured:
        return None
    with native_function_manager(g.out):
        name = meta.name(g)
        args = structured.meta_arguments(g)
        args_str = ', '.join(a.decl() for a in args)
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = "at::impl::MetaBase"
        meta_return = "void"
        precomputed = g.out.precomputed if g.structured else None

        if precomputed:
            # Generate the template declaration with one bool parameter for each
            # precomputed element. Each parameter is true if the corresponding (in
            # terms of position) precomputed element has been set.
            precomputed_values = [*precomputed.replace.values(), precomputed.add]
            precomputed_elements = [elem for replace_list in precomputed_values for elem in replace_list]
            precomputed_template_parameters = [elem.name.upper() for elem in precomputed_elements]
            precomputed_template_params_str = ", ".join(f"bool {param} = false" for param in precomputed_template_parameters)
            precompute_template_decl = f"template <{precomputed_template_params_str}>"

            # Generate a string containing declarations of all precomputed elements.
            precomputed_elements_with_cpp_types = [
                structured.argument_type(elem, binds=elem.name)
                for elem in precomputed_elements
            ]

            precomputed_elements_decl = ";\n".join(
                f"{elem.cpp_type(strip_ref=True)} {elem.name}" for elem in precomputed_elements_with_cpp_types
            )

            # Generate "setter" methods for each precomputed element. Each method will return
            # a new instance of precompute_out with the template parameter that corresponds to
            # the member set by the method to true (to indicate that it has been set).
            setter_methods = []
            for i, elem in enumerate(precomputed_elements):
                # Generate the signature. The return type will be the same
                # as the type of `this` but with the template parameter
                # corresponding to the element set by this method set to true.
                # The assert generated below will ensure that this template
                # parameter is false on the type of `this`.
                return_ty_templates = ", ".join(
                    precomputed_template_parameters[:i] + ["true"] + precomputed_template_parameters[i + 1:]
                )
                return_ty = f"precompute_out<{return_ty_templates}>"
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(strip_ref=True)
                signature = f"{return_ty} set_{elem.name}({elem_cpp_ty} value)"

                # Generate an assert which checks that the
                # template parameter corresponding to the precomputed
                # element that is set by this method is false on the
                # class corresponding to the object that `this` points to.
                # This ensures that each element can be set only once.
                assert_msg = f"\"{precomputed_elements[i].name} already set\""
                assert_stmt = f"static_assert({precomputed_template_parameters[i]} == false, {assert_msg});"

                # Generate the new object construction block. All state
                # except the element that this method sets is copied from the
                # object that `this` points to. The value for the element that
                # the method sets is taken from a method parameter.
                construction_stmts = []
                construction_stmts.append(f"{return_ty} ret;")

                for j, elem in enumerate(precomputed_elements):
                    if i == j:
                        construction_stmts.append(f"ret.{elem.name} = value;")
                    else:
                        construction_stmts.append(f"ret.{elem.name} = this->{elem.name};")

                construction_stmts.append("return ret;")
                construction_block = "\n".join(construction_stmts)

                setter_methods.append(f"""
                    {signature} {{
                        {assert_stmt}
                        {construction_block}
                    }}
                """)
            setter_methods_decl = "\n".join(setter_methods)

            # Meta should return an instance of the struct containing the precomputed elements.
            meta_return_template_params = ", ".join(["true"] * len(precomputed_template_parameters))
            # This typedef (actually a using statement) is needed so that TORCH_META_FUNC can reuse the return
            # type (which has a variable number of template parameters).
            meta_return_typedef = f"using meta_return_ty = precompute_out <{meta_return_template_params}>;"
            meta_return = "meta_return_ty"
            precomputed_decl = f"""
                {precompute_template_decl}
                struct TORCH_API precompute_out {{
                    {setter_methods_decl}
                    {precomputed_elements_decl};
            }};"""
        else:
            meta_return_typedef = ""
            precomputed_decl = ""

        return f"""\
struct TORCH_API structured_{name} : public {parent_class} {{
    {precomputed_decl}
    {meta_return_typedef}
    {meta_return} meta({args_str});
}};
"""

def parse_native_yaml_struct(es: object, path: str = "<stdin>") -> ParsedYaml:
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    for e in es:
        assert isinstance(e.get('__line__'), int), e
        loc = Location(path, e['__line__'])
        funcs = e.get('func')
        with context(lambda: f'in {loc}:\n  {funcs}'):
            func, m = NativeFunction.from_yaml(e, loc)
            rs.append(func)
            BackendIndex.grow_index(bs, m)
    error_check_native_functions(rs)
    # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
    indices: Dict[DispatchKey, BackendIndex] = defaultdict(lambda: BackendIndex(
        dispatch_key=DispatchKey.Undefined,
        use_out_as_primary=True,
        external=False,
        device_guard=False,
        index={}))
    for k, v in bs.items():
        # All structured in-tree operators are implemented in terms of their out operator.
        indices[k] = BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            # Only cuda-like devices in tree require device guards
            device_guard=is_cuda_dispatch_key(k),
            index=v)
    return ParsedYaml(rs, indices)


def parse_native_yaml(path: str) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        with open(path, 'r') as f:
            es = yaml.load(f, Loader=LineLoader)
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(es, path=path)

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# Some assertions are already performed during parsing, but those are only within a single NativeFunction.
# Assertions here are meant to be performed across NativeFunctions.
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: Dict[OperatorName, NativeFunction] = {}
    base_func_map: Dict[BaseOperatorName, List[NativeFunction]] = defaultdict(list)
    for f in funcs:
        func_map[f.func.name] = f
        base_func_map[f.func.name.name].append(f)
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map[f.structured_delegate]
            assert delegate_func.structured, \
                f"{f.func.name} is marked as a structured_delegate pointing to " \
                f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. " \
                f"Consider adding 'structured=True' to the delegated operator"
        if f.tag is not None and f.tag is Tag.inplace_view:
            base_name = f.func.name.name
            overload_name = f.func.name.overload_name
            assert base_name.inplace, \
                f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming " \
                "convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            out_of_place_base_name = BaseOperatorName(base_name.base, False, base_name.dunder_method)
            assert len(base_func_map[out_of_place_base_name]) > 0, \
                f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding " \
                f"out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "


def gen_per_operator_headers(
        *,
        native_functions: Sequence[NativeFunction],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        static_dispatch_idx: Optional[BackendIndex],
        selector: SelectiveBuilder,
        backend_indices: Dict[DispatchKey, BackendIndex],
        cpu_fm: FileManager,
        cuda_fm: FileManager,
        ops_fm: FileManager,
        functions_keys: Set[DispatchKey],
        dispatch_keys: Sequence[DispatchKey],
        rocm: bool,
) -> None:
    # For CMake builds, split operator declarations into separate headers in
    # the ATen/ops folder to split up header dependencies
    functions_by_root_name: Dict[str, List[NativeFunction]] = defaultdict(lambda: [])
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)

    grouped_functions_by_root_name: Dict[str, List[Union[NativeFunction, NativeFunctionsGroup]]] = defaultdict(lambda: [])
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)

    for name, functions in functions_by_root_name.items():

        grouped_functions = grouped_functions_by_root_name.get(name, [])
        structured_functions = [fn for fn in grouped_functions
                                if isinstance(fn, NativeFunctionsGroup) and fn.structured]
        is_structured = len(structured_functions) > 0

        ops_fm.write_with_template(
            f'{name}_native.h', 'NativeFunction.h', lambda: {
                'extra_includes': (f'#include <ATen/ops/{name}_meta.h>'
                                   if is_structured else []),
                'native_function_declarations': list(concatMap(
                    # Convert to a set first to remove duplicate kernel names.
                    # Backends are allowed to repeat kernel names; only generate the declaration once!
                    lambda f: list(OrderedDict.fromkeys(concatMap(
                        lambda backend_idx:
                        dest.compute_native_function_declaration(f, backend_idx),
                        backend_indices.values()))),
                    grouped_functions)),
            })

    for category, suffix in [
        ('Functions', ''),
        ('NativeFunctions', '_native'),
    ]:
        cpu_fm.write(f'{category}.h', lambda: {
            'static_dispatch_extra_headers': [],
            f'{category}_includes': [
                f'#include <ATen/ops/{name}{suffix}.h>'
                for name in sorted(functions_by_root_name.keys())
            ],
            f'{category}_declarations': [],
        })

    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue

        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []

        for name, functions in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(concatMap(
                dest.RegisterDispatchKey(
                    backend_indices[dispatch_key],
                    Target.NAMESPACED_DECLARATION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_functions
            ))

            if len(declarations) == 0:
                continue

            dispatch_names.append(name)
            ops_fm.write_with_template(
                f'{name}_{dispatch_namespace}_dispatch.h',
                'DispatchKeyFunction.h', lambda: {
                    'dispatch_namespace': dispatch_namespace,
                    'dispatch_namespaced_declarations': declarations,
                })



def gen_source_files(
        *,
        native_functions: Sequence[NativeFunction],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        structured_native_functions: Sequence[NativeFunctionsGroup],
        selector: SelectiveBuilder,
        backend_indices: Dict[DispatchKey, BackendIndex],
        core_fm: FileManager,
        cpu_fm: FileManager,
        cpu_vec_fm: FileManager,
        cuda_fm: FileManager,
        dispatch_keys: Sequence[DispatchKey],
        functions_keys: Set[DispatchKey],
        rocm: bool,
        force_schema_registration: bool,
        per_operator_headers: bool,
        skip_dispatcher_op_registration: bool,
) -> None:
    extra_cuda_headers = '''\
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>'''
    if rocm:
        extra_cuda_headers = '''\
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>'''

    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm

        if per_operator_headers:
            def operator_headers() -> List[str]:
                headers = []
                for g in grouped_native_functions:
                    is_registered = False
                    if backend_index.has_kernel(g):
                        is_registered = True
                    # The above has_kernel test on a group will only test for
                    # the existence of out dispatch, because that's how
                    # structured kernels work. But sometimes functions can be
                    # grouped but not be structured, and then you need to check
                    # each individual piece, as they may have manual dispatch
                    # entries.
                    elif isinstance(g, NativeFunctionsGroup) and any(backend_index.has_kernel(fn) for fn in g.functions()):
                        is_registered = True
                    # TODO: this condition is a bit questionable
                    elif g.structured and dispatch_key in (DispatchKey.Meta, DispatchKey.CompositeExplicitAutograd):
                        is_registered = True
                    if not is_registered:
                        continue

                    headers.append(f"#include <ATen/ops/{g.root_name}_native.h>")
                    if dispatch_key == DispatchKey.CompositeExplicitAutograd:
                        headers.append(f"#include <ATen/ops/{g.root_name}.h>")
                    if dispatch_key in functions_keys:
                        headers.append(
                            f"#include <ATen/ops/{g.root_name}_{dispatch_namespace}_dispatch.h>")

                return sorted(set(headers))
        else:
            def operator_headers() -> List[str]:
                headers = ["#include <ATen/NativeFunctions.h>"]
                if dispatch_key == DispatchKey.CompositeExplicitAutograd:
                    headers.append("#include <ATen/Functions.h>")
                if dispatch_key in functions_keys:
                    headers.append(f"#include <ATen/{dispatch_key!s}Functions.h>")
                return headers

        backend_index = backend_indices[dispatch_key]
        dispatch_namespace = str(dispatch_key).lower()
        fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
            'extra_cuda_headers': extra_cuda_headers if is_cuda_dispatch_key(dispatch_key) else '',
            'external_backend_headers': '',
            'dispatch_headers': dest.gen_registration_headers(backend_index, per_operator_headers, rocm),
            'ops_headers': operator_headers(),
            'DispatchKey': dispatch_key,
            'dispatch_namespace': dispatch_key.lower(),
            'dispatch_helpers': dest.gen_registration_helpers(backend_index),
            'dispatch_namespaced_definitions': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.NAMESPACED_DEFINITION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
            'dispatch_anonymous_definitions': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.ANONYMOUS_DEFINITION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
            'dispatch_registrations': [] if skip_dispatcher_op_registration else list(concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.REGISTRATION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
        })

    core_fm.write('TensorMethods.cpp', lambda: {})


def get_custom_build_selector(
        provided_op_registration_allowlist: Optional[List[str]],
        op_selection_yaml_path: Optional[str]) -> SelectiveBuilder:
    assert not (
            provided_op_registration_allowlist is not None and
            op_selection_yaml_path is not None), (
            "Both provided_op_registration_allowlist and " +
            "op_selection_yaml_path can NOT be provided at the " +
            "same time.")

    op_registration_allowlist: Optional[Set[str]] = None
    if provided_op_registration_allowlist is not None:
        op_registration_allowlist = set(provided_op_registration_allowlist)

    if op_registration_allowlist is not None:
        selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
            op_registration_allowlist,
            True,
            False,
        )
    elif op_selection_yaml_path is not None:
        selector = SelectiveBuilder.from_yaml_path(op_selection_yaml_path)
    else:
        selector = SelectiveBuilder.get_nop_selector()

    return selector


def pre_group_native_functions(
        native_functions: Sequence[NativeFunction]) -> Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]]:
    pre_grouped_native_functions: Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        assert f.func.kind() not in d
        d[f.func.kind()] = f
    return pre_grouped_native_functions


def get_grouped_native_functions(
        native_functions: Sequence[NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    def flatten_pre_group(d: Dict[SchemaKind, NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            return list(d.values())
        else:
            return [r]

    # TODO: how come ValuesView isn't a Sequence lol
    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    return list(concatMap(flatten_pre_group, list(pre_grouped_native_functions.values())))


def gen_headers(
        *,
        native_functions: Sequence[NativeFunction],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        structured_native_functions: Sequence[NativeFunctionsGroup],
        static_dispatch_idx: Optional[BackendIndex],
        selector: SelectiveBuilder,
        backend_indices: Dict[DispatchKey, BackendIndex],
        core_fm: FileManager,
        cpu_fm: FileManager,
        cuda_fm: FileManager,
        ops_fm: FileManager,
        dispatch_keys: Sequence[DispatchKey],
        functions_keys: Set[DispatchKey],
        rocm: bool,
        per_operator_headers: bool,
) -> None:
    if per_operator_headers:
        gen_per_operator_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )
    else:
        raise "Not supported"

    def static_dispatch_method_headers() -> List[str]:
        return list(mapMaybe(
            lambda fn: static_dispatch_ops_header(fn, backend_index=static_dispatch_idx),
            [fn for fn in native_functions if Variant.method in fn.variants]))


    core_fm.write('TensorBody.h', lambda: {
        'static_dispatch_ops_headers': (
            static_dispatch_method_headers() if per_operator_headers
            else static_dispatch_extra_headers(static_dispatch_idx, skip_tensor_include=True)),
        'tensor_method_declarations': list(mapMaybe(ComputeTensorMethod(
            target=Target.DECLARATION, static_dispatch_backend_index=static_dispatch_idx), native_functions)),
        'tensor_method_definitions': list(mapMaybe(ComputeTensorMethod(
            target=Target.DEFINITION, static_dispatch_backend_index=static_dispatch_idx), native_functions)),
    })


    def gen_aten_interned_strings() -> Dict[str, str]:
        attrs = set()  # All function argument names
        names = set()  # All ATen function names
        for func in native_functions:
            names.add(str(func.func.name.name))
            # Some operators don't have a functional variant but we still create a
            # symbol without the underscore
            names.add(func.func.name.name.base)

            for arg in func.func.schema_order_arguments():
                attrs.add(arg.name)

        # These are keywords in C++, so aren't valid symbol names
        # https://en.cppreference.com/w/cpp/language/operator_alternative
        names -= set(['and', 'and_eq', 'bitand', 'bitor', 'compl', 'not',
                      'not_eq', 'or', 'or_eq', 'xor', 'xor_eq'])

        return {
            'aten_symbols': ' \\\n'.join([
                f"_(aten, {name})" for name in sorted(names)
            ]),
            'attr_symbols': ' \\\n'.join([
                f"_(attr, {name})" for name in sorted(attrs)
            ]),
        }

    core_fm.write('aten_interned_strings.h', gen_aten_interned_strings)

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate operator source files')
    parser.add_argument(
        '-s',
        '--source-path',
        help='path to source directory for kernels',
        default='kernels')
    parser.add_argument(
        '-d', '--install_dir', help='output directory',
        default='build/aten/src/ATen')
    parser.add_argument(
        '-o',
        '--output-dependencies',
        help='output a list of dependencies into the given file and exit')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='run without writing any files (still updates outputs)')
    parser.add_argument(
        '--static_dispatch_backend',
        help='generate static dispatch code for the specific backend (if set)')
    parser.add_argument(
        '--backend_whitelist',
        nargs='*',
        help='filter dispatch backend by the whitelist (if set), '
             'e.g.: CPU CUDA QuantizedCPU ...')
    parser.add_argument(
        '--op_registration_whitelist',
        nargs='*',
        help='filter op registrations by the whitelist (if set); '
             'each item is `namespace`::`operator name` without overload name; '
             'e.g.: aten::empty aten::conv2d ...')
    parser.add_argument(
        '--op_selection_yaml_path',
        help='Provide a path to the operator selection (for custom build) YAML '
             'that contains the information about the set of selected operators '
             'and their categories (training, ...). Each operator is either a '
             'full operator name with overload or just a bare operator name. '
             'The operator names also contain the namespace prefix (e.g. aten::)')
    parser.add_argument(
        '--skip_dispatcher_op_registration',
        action='store_true',
        help='Avoid registering operators into the dispatcher.')
    parser.add_argument(
        '--rocm',
        action='store_true',
        help='reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly')
    parser.add_argument(
        '--per-operator-headers', action='store_true',
        help='generate separate headers per operator in ATen/ops')
    parser.add_argument(
        '--force_schema_registration',
        action='store_true',
        help='force it to generate schema-only registrations for all ops, including'
             'those that are not listed on --op_registration_whitelist')

    options = parser.parse_args()
    native_yaml_path = os.path.join(options.source_path, 'functions.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices

    ops_install_dir = f'{options.install_dir}/ops'
    pathlib.Path(ops_install_dir).mkdir(parents=True, exist_ok=True)
    core_install_dir = f'{options.install_dir}/core'
    pathlib.Path(core_install_dir).mkdir(parents=True, exist_ok=True)

    core_fm = make_file_manager(options=options, install_dir=core_install_dir)
    cpu_fm = make_file_manager(options=options)
    cpu_vec_fm = make_file_manager(options=options)
    cuda_fm = make_file_manager(options=options)
    ops_fm = make_file_manager(options=options, install_dir=ops_install_dir)
    grouped_native_functions = get_grouped_native_functions(native_functions)
    structured_native_functions = [g for g in grouped_native_functions
                                   if isinstance(g, NativeFunctionsGroup)]
    static_dispatch_idx: Optional[BackendIndex] = None
    if options.static_dispatch_backend:
        static_dispatch_idx = backend_indices[DispatchKey.parse(options.static_dispatch_backend)]
    from codegen.model import dispatch_keys
    if options.backend_whitelist:
        dispatch_keys = [k for k in dispatch_keys if is_generic_dispatch_key(k) or str(k) in options.backend_whitelist]
    functions_keys = {
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.Meta,
    }

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    gen_headers(
        native_functions=native_functions,
        grouped_native_functions=grouped_native_functions,
        structured_native_functions=structured_native_functions,
        static_dispatch_idx=static_dispatch_idx,
        selector=selector,
        backend_indices=backend_indices,
        core_fm=core_fm,
        cpu_fm=cpu_fm,
        cuda_fm=cuda_fm,
        ops_fm=ops_fm,
        dispatch_keys=dispatch_keys,
        functions_keys=functions_keys,
        rocm=options.rocm,
        per_operator_headers=options.per_operator_headers,
    )
    gen_source_files(
        native_functions=native_functions,
        grouped_native_functions=grouped_native_functions,
        structured_native_functions=structured_native_functions,
        selector=selector,
        backend_indices=backend_indices,
        core_fm=core_fm,
        cpu_fm=cpu_fm,
        cpu_vec_fm=cpu_vec_fm,
        cuda_fm=cuda_fm,
        dispatch_keys=dispatch_keys,
        functions_keys=functions_keys,
        rocm=options.rocm,
        force_schema_registration=options.force_schema_registration,
        per_operator_headers=options.per_operator_headers,
        skip_dispatcher_op_registration=options.skip_dispatcher_op_registration,
    )
    gen_unboxing(native_functions=native_functions, cpu_fm=cpu_fm)


if __name__ == '__main__':
    main()
