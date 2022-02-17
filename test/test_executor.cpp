#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



// Tests go in torch::jit
namespace torch {
namespace executor {
TEST(ExecutorTest, Simple) {
  printf("Simple test goes here.\n");
}
} // namespace executor
} // namespace torch
