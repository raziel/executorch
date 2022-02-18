#include <gtest/gtest.h>
#include <core/tensor.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



// Tests go in torch::jit
namespace torch {
namespace executor {
TEST(ExecutorTest, Simple) {
  Tensor a;
  a.type = ScalarType::Int;
  a.dim = 2;
  a.sizes = new int[a.dim]{2, 2};
  int a_data[4]{1, 2, 3, 4};
  a.data = a_data;
  a.nbytes = 2 * 2 * sizeof(int);

  auto data_p = static_cast<int*>(a.data);
  for (int i = 0; i < a.sizes[0]; ++i) {
    for (int j = 0; j < a.sizes[1]; ++j) {
      printf("a[%d, %d] = %d\n", i, j, data_p[2 * i + j]);
    }
  }
}
} // namespace executor
} // namespace torch
