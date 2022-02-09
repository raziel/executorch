cmake_minimum_required(VERSION 3.14)
# set(CMAKE_VERBOSE_MAKEFILE 1)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Because of OSX: https://github.com/google/flatbuffers/pull/6990/files
set(FLATBUFFERS_OSX_BUILD_UNIVERSAL OFF CACHE BOOL "for OSX build")
add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)


target_link_libraries(
  flatbuffers
  )

add_custom_command(
  OUTPUT schema_generated.h
  COMMAND flatc --cpp "${EXECUTORCH_SOURCE_DIR}/schema/schema.fbs"
  DEPENDS "${EXECUTORCH_SOURCE_DIR}/schema/schema.fbs"
  WORKING_DIRECTORY "${EXECUTORCH_SOURCE_DIR}/schema"
  )
add_custom_target(cpp_binding ALL
  DEPENDS schema_generated.h
  )