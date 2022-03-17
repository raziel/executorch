cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Wall -Werror")

if(USE_ASAN)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address")
  string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fsanitize=address")
endif()

project(pytree
    DESCRIPTION "Lib to flatten/unflatten nested structures to list of leafs"
    VERSION 0.0.1)
add_library(pytree SHARED
    pytree.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pytree.cpp
)

target_include_directories(pytree
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)

install(TARGETS pytree
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test ${CMAKE_BINARY_DIR}/pytree/test)

