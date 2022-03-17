set(EXECUTOR_TEST_ROOT ${PROJECT_SOURCE_DIR}/test)

# Build the cpp gtest binary containing the cpp-only tests.
set(EXECUTOR_TEST_SRCS
  ${EXECUTOR_TEST_ROOT}/test_executor.cpp
)

add_executable(test_executor
  ${EXECUTOR_TEST_SRCS}
)

target_link_libraries(
  test_executor PRIVATE flatbuffers)


# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_executor PRIVATE USE_GTEST)

set(EXECUTOR_TEST_DEPENDENCIES gtest executorch pytree)

target_link_libraries(test_executor PRIVATE ${EXECUTOR_TEST_DEPENDENCIES})
#target_include_directories(test_executor PRIVATE ${ATen_CPU_INCLUDE})
if(NOT MSVC)
  target_compile_options(test_executor PRIVATE -Wno-unused-variable)
endif()

if(LINUX)
  #Update to target_link_options when CMake version can be upgraded
  target_link_libraries(test_executor PRIVATE "-Wl,--no-as-needed")
endif()

if(INSTALL_TEST)
  install(TARGETS test_jit DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_jit> DESTINATION bin OPTIONAL)
  endif()
endif()
