#include <pytree.h>

#include <cstring>
#include <ctype.h>

namespace torch {
namespace executor {
namespace pytree {

struct Stack final {
  struct Item {
    size_t idx;
    size_t last_sep_idx = 0;
  };

  constexpr static const size_t SIZE = 4;
  Item data[SIZE];
  size_t size = 0;

  void push(size_t idx) {
    assert(size < SIZE);
    Item &item = data[size++];
    item.idx = idx;
    item.last_sep_idx = idx;
  }

  Item &pop() {
    assert(size > 0);
    return data[--size];
  }

  Item &top() {
    assert(size > 0);
    return data[size - 1];
  }
};

Inflator::Inflator(Spec spec) : spec_(spec) {
  Stack stack;
  size_t i = 0;
  size_t containers_num = 0;
  size_t size = strlen(spec);
  spec_data_.idxs = new size_t[size];
  while (i < size) {
    const auto c = spec[i];
    switch (c) {
    case 'D':
    case 'L':
    case '$': {
      containers_num++;
      break;
    }
    case '(': {
      stack.push(i);
      break;
    }
    case ')': {
      auto &item = stack.top();
      size_t last_sep_idx = item.last_sep_idx;
      spec_data_.idxs[last_sep_idx] = i;
      stack.pop();
      break;
    }
    case '"': {
      size_t idx = i;
      i++;
      while (spec[i] != '"') {
        i++;
      }
      spec_data_.idxs[idx] = i;
      spec_data_.idxs[i] = idx;
      break;
    }
    case ',': {
      auto &item = stack.top();
      size_t last_sep_idx = item.last_sep_idx;
      spec_data_.idxs[last_sep_idx] = i;
      item.last_sep_idx = i;
      break;
    }
    }
    i++;
  }
  spec_data_.containers_num = containers_num;
}

size_t Inflator::read_number(size_t &read_idx) {
  size_t num = 0;
  while (isdigit(spec_[read_idx])) {
    size_t digit = spec_[read_idx] - '0';
    num = 10 * num + digit;
    read_idx++;
  }
  return num;
}

Inflator::Layout Inflator::read_layout(size_t &read_idx) {
  const size_t child_num = read_number(read_idx);
  Layout layout{child_num, new size_t[child_num]};

  size_t child_idx = 0;
  while (spec_[read_idx] == '#') {
    layout.leaves_num[child_idx++] = read_number(++read_idx);
  }
  return layout;
}

} // namespace pytree
} // namespace executor
} // namespace torch
