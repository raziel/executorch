cmake_minimum_required(VERSION 3.14)
# set(CMAKE_VERBOSE_MAKEFILE 1)
#include(codegen/Codegen.cmake)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Executorch project
project(Executorch
    DESCRIPTION "Simple and portable executor of PyTorch programs"
    VERSION 1.0.1)

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

# Generate schema.

add_custom_command(
  OUTPUT schema_generated.h
  COMMAND flatc --cpp --gen-mutable --scoped-enums "${EXECUTORCH_SOURCE_DIR}/schema/schema.fbs"
  DEPENDS "${EXECUTORCH_SOURCE_DIR}/schema/schema.fbs"
  WORKING_DIRECTORY "${EXECUTORCH_SOURCE_DIR}/schema"
)
add_custom_target(schema_header ALL
  DEPENDS schema_generated.h
)

# Create library
include_directories("flatbuffers/include")
add_library(executorch SHARED
    executor.cpp
    ${EXECUTORCH_SOURCE_DIR}/core/error_message.cpp
    ${EXECUTORCH_SOURCE_DIR}/core/operator_registry.cpp
    ${EXECUTORCH_SOURCE_DIR}/kernels/demo_operator_kernels.cpp
)

add_dependencies(executorch schema_header)

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
