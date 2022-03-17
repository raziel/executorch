#include <gtest/gtest.h>
#include <pytree.h>
#include <string>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

namespace torch {
namespace executor {
namespace pytree {

using Leaf = int;

TEST(PyTreeTest, List) {
  Inflator inf("L2#1#1($,$)");
  Leaf items[2] = {11, 12};
  auto c = inf.unflatten(items);
  ASSERT_TRUE(c.isList());
  ASSERT_EQ(c.size(), 2);

  const auto &child0 = c[0];
  const auto &child1 = c[1];

  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child0, 11);
  ASSERT_EQ(child1, 12);
}

TEST(PyTreeTest, Tuple) {
  Inflator inf("T1#1($)");
  Leaf items[1] = {11};
  auto c = inf.unflatten(items);
  ASSERT_TRUE(c.isList());
  ASSERT_EQ(c.size(), 1);

  const auto &child0 = c[0];

  ASSERT_TRUE(child0.isLeaf());
  ASSERT_EQ(child0, 11);
}

TEST(PyTreeTest, Dict) {
  Inflator inf("D2#1#1(\"key0\":$,\"key1\":$)");
  Leaf items[2] = {11, 12};
  auto c = inf.unflatten(items);
  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 2);

  auto key0 = c.key(0);
  auto key1 = c.key(1);

  ASSERT_EQ(strcmp(key0, "key0"), 0);
  ASSERT_EQ(strcmp(key1, "key1"), 0);

  const auto &child0 = c.value(0);
  const auto &child1 = c.value(1);
  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child0, 11);
  ASSERT_EQ(child1, 12);

  const auto &ckey0 = c.at("key0");
  ASSERT_EQ(child0, ckey0);

  ASSERT_EQ(c.at("key0"), 11);
  ASSERT_EQ(c.at("key1"), 12);
}

TEST(PyTreeTest, Leaf) {
  Inflator inf("$");
  Leaf items[2] = {11};
  auto c = inf.unflatten(items);
  ASSERT_TRUE(c.isLeaf());
  ASSERT_EQ(c, 11);
}

TEST(PyTreeTest, DictWithList) {
  Inflator inf("D2#2#1(\"key0\":L2#1#1($,$),\"key1\":$)");
  Leaf items[3] = {11, 12, 13};
  auto c = inf.unflatten(items);
  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 2);

  auto key0 = c.key(0);
  auto key1 = c.key(1);

  ASSERT_EQ(strcmp(key0, "key0"), 0);
  ASSERT_EQ(strcmp(key1, "key1"), 0);

  const auto &child1 = c.value(1);
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child1, 13);

  const auto &list = c.value(0);
  ASSERT_TRUE(list.isList());
  ASSERT_EQ(list.size(), 2);

  const auto &list_child0 = list[0];
  const auto &list_child1 = list[1];

  ASSERT_TRUE(list_child0.isLeaf());
  ASSERT_TRUE(list_child1.isLeaf());

  ASSERT_EQ(list_child0, 11);
  ASSERT_EQ(list_child1, 12);
}

TEST(PyTreeTest, DictKeysStrInt) {
  Inflator inf("D4#1#1#1#1(\"key0\":$,1:$,23:$,123:$)");
  Leaf items[4] = {11, 12, 13, 14};
  auto c = inf.unflatten(items);
  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 4);

  auto key0 = c.key(0);
  auto key1 = c.key(1);

  ASSERT_EQ(strcmp(key0, "key0"), 0);
  ASSERT_EQ(key1, 1);

  const auto &child0 = c.value(0);
  const auto &child1 = c.value(1);
  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());
  ASSERT_EQ(child0, 11);
  ASSERT_EQ(child1, 12);

  const auto &ckey0 = c.at("key0");
  ASSERT_EQ(child0, ckey0);

  ASSERT_EQ(c.at(1), 12);
  ASSERT_EQ(c.at(23), 13);
  ASSERT_EQ(c.at(123), 14);
}

} // namespace pytree
} // namespace executor
} // namespace torch
