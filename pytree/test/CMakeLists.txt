set(PYTREE_TEST_ROOT ${CMAKE_CURRENT_LIST_DIR})

set(PYTREE_TEST_SRCS
  ${PYTREE_TEST_ROOT}/test_pytree.cpp
)

add_executable(test_pytree
  ${PYTREE_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_pytree PRIVATE USE_GTEST)

set(PYTREE_TEST_DEPENDENCIES gtest)

target_link_libraries(test_pytree PRIVATE gtest pytree)
