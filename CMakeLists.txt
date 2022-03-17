include(ExternalProject)

cmake_minimum_required(VERSION 3.14)
#set(CMAKE_VERBOSE_MAKEFILE 1)

set(EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

if("${TARGET_PLATFORM}" STREQUAL "ethos")
    set(TARGET_CPU "cortex-m55" CACHE STRING "Target CPU")
    set(CMAKE_TOOLCHAIN_FILE
        ${EXECUTORCH_SOURCE_DIR}/toolchains/arm-none-eabi-gcc.cmake)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# To stay compatible with as many cross-compilation toolchains as possible,
# the executor runtime must build with C++11, and without any compiler-specific
# extensions.
# TODO: We could move this restriction to the executor instead of applying it
# globally.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Executorch project
project(Executorch
    DESCRIPTION "Simple and portable executor of PyTorch programs"
    VERSION 1.0.1)

include(codegen/Codegen.cmake)

# Check submodule
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/flatbuffers/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/googletest/CMakeLists.txt")
  message(FATAL_ERROR "The submodules were not downloaded! Please run `git submodule update --init --recursive`.")
endif()

# Build flatbuffers
# Because of OSX: https://github.com/google/flatbuffers/pull/6990/files
set(FLATBUFFERS_OSX_BUILD_UNIVERSAL OFF CACHE BOOL "for OSX build")
add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)


target_link_libraries(
  flatbuffers
)

set(FLATC_BUILD_DIR ${EXECUTORCH_SOURCE_DIR}/flatbuffers/build)

# To build the flatc target we add flatbuffers as an external project and
# then use that target later in generating the schema. This is to ensure
# that while cross-compiling we still build flatc with the host toolchain
# and not the target toolchain.
ExternalProject_Add(
  flatc_build
  SOURCE_DIR ${EXECUTORCH_SOURCE_DIR}/flatbuffers
  BINARY_DIR ${FLATC_BUILD_DIR}
  CMAKE_ARGS
  INSTALL_COMMAND ""
)

# Generate schema.

add_custom_command(
  OUTPUT schema_generated.h
  COMMAND ${FLATC_BUILD_DIR}/flatc --cpp --gen-mutable --scoped-enums "${EXECUTORCH_SOURCE_DIR}/schema/schema.fbs"
  DEPENDS "${EXECUTORCH_SOURCE_DIR}/schema/schema.fbs"
  WORKING_DIRECTORY "${EXECUTORCH_SOURCE_DIR}/schema"
)

add_custom_target(schema_header ALL
  DEPENDS flatc_build schema_generated.h
)

# Create library
include_directories("flatbuffers/include")
add_library(executorch SHARED
    executor.cpp
    ${EXECUTORCH_SOURCE_DIR}/core/error_message.cpp
    ${EXECUTORCH_SOURCE_DIR}/core/operator_registry.cpp
    ${EXECUTORCH_SOURCE_DIR}/kernels/demo_operator_kernels.cpp
)

if(APPLE)
  set(link_flag_1 "-Wl,-all_load")
else()
  set(link_flag_1 "-Wl,--whole-archive")
  set(link_flag_2 "-Wl,--no-whole-archive")
endif()

target_link_libraries(executorch PUBLIC
        ${link_flag_1}
        GEN_TARGET
        ${link_flag_2}
)

set_target_properties(executorch PROPERTIES
    OUTPUT_NAME executorch
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)
configure_file(executorch.pc.in executorch.pc @ONLY)

# Include directories.
set(EXECUTORCH_INCLUDE_DIRS
  "${EXECUTORCH_SOURCE_DIR}"
  "${EXECUTORCH_SOURCE_DIR}/schema"
  "${EXECUTORCH_SOURCE_DIR}/core"
)
target_include_directories(executorch
  PUBLIC
    ${EXECUTORCH_INCLUDE_DIRS}
)

install(TARGETS executorch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/executorch.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

add_subdirectory(${EXECUTORCH_SOURCE_DIR}/test ${CMAKE_BINARY_DIR}/test)

add_subdirectory(${EXECUTORCH_SOURCE_DIR}/googletest)
