set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE sources_templates "${CMAKE_CURRENT_LIST_DIR}/../kernels/templates/*.cpp")
file(GLOB_RECURSE headers_templates "${CMAKE_CURRENT_LIST_DIR}/../kernels/templates/*.h")
file(GLOB_RECURSE unboxing_templates "${CMAKE_CURRENT_LIST_DIR}/../kernels/templates/*")

file(GLOB_RECURSE all_codegen_scripts "${CMAKE_CURRENT_LIST_DIR}/../codegen/*.py")

if(NOT PYTHON_EXECUTABLE)
    execute_process(
            COMMAND "which" "python3" RESULT_VARIABLE _exitcode OUTPUT_VARIABLE _py_exe)
    if(${_exitcode} EQUAL 0)
        if(NOT MSVC)
            string(STRIP ${_py_exe} PYTHON_EXECUTABLE)
        endif()
        message(STATUS "Setting Python to ${PYTHON_EXECUTABLE}")
    endif()
endif()

list(APPEND CUSTOM_BUILD_FLAGS --skip_dispatcher_op_registration)
set(GEN_COMMAND
        "${PYTHON_EXECUTABLE}" -m codegen.gen
        --static_dispatch_backend=CPU
        --skip_dispatcher_op_registration
        --per-operator-headers
        --backend_whitelist=CPU
        --install_dir=${CMAKE_BINARY_DIR}/generated
        )

foreach(gen_type "headers" "sources")
    # The codegen outputs may change dynamically as PyTorch is
    # developed, but add_custom_command only supports dynamic inputs.
    #
    # We work around this by generating a .cmake file which is
    # included below to set the list of output files. If that file
    # ever changes then cmake will be re-run automatically because it
    # was included and so we get fully dynamic outputs.

    set("GEN_COMMAND_${gen_type}"
            ${GEN_COMMAND}
            --generate ${gen_type}
            --output-dependencies ${CMAKE_BINARY_DIR}/generated/generated_${gen_type}.cmake
            )

    # Dry run to bootstrap the output variables
    execute_process(
            COMMAND ${GEN_COMMAND_${gen_type}} --dry-run
            RESULT_VARIABLE RETURN_VALUE
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )

    if(NOT RETURN_VALUE EQUAL 0)
        message(FATAL_ERROR "Failed to get generated_${gen_type} list")
    endif()

    include("${CMAKE_BINARY_DIR}/generated/generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/generated/ops_generated_${gen_type}.cmake")

    message(STATUS "generated_${gen_type}:  ${generated_${gen_type}}")
    message(STATUS "core_generated_${gen_type}:  ${core_generated_${gen_type}}")
    message(STATUS "ops_generated_${gen_type}:  ${ops_generated_${gen_type}}")
    message(STATUS "${gen_type}_templates: ${${gen_type}_templates}")

endforeach()
message(STATUS "${CMAKE_CURRENT_LIST_DIR}/core")

add_custom_command(
        COMMENT echo -e "Generating headers and sources"
        OUTPUT
        ${generated_sources}
        ${generated_headers}
        ${core_generated_headers}
        ${core_generated_sources}
        ${ops_generated_headers}
        ${ops_generated_sources}
        COMMAND ${GEN_COMMAND}
        DEPENDS ${all_codegen_scripts} ${headers_templates} ${sources_templates}
        ${CMAKE_CURRENT_LIST_DIR}/../kernels/functions.yaml
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
)

add_library(GEN_TARGET STATIC
        ${generated_headers} ${ops_generated_headers}
        ${generated_sources} ${ops_generated_sources})

target_include_directories(GEN_TARGET PUBLIC ${CMAKE_BINARY_DIR}/generated ${CMAKE_CURRENT_LIST_DIR}/../core)
