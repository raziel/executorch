#include <cassert>
#include <cstring>
#include <ctype.h>

namespace torch {
namespace executor {
namespace pytree {

using Spec = const char *;
// to save size on template<T> - using Value = void* ?

enum class Kind { List, Dict, Leaf, None };

using KeyStr = const char *;
using KeyInt = int;

struct Key {
  enum class Kind { None, Int, Str } kind;

  union KeyValue {
    KeyInt as_int;
    KeyStr as_str;
  } value;

  Key() : kind(Kind::None) {}

  Key(KeyInt key) : kind(Kind::Int) { value.as_int = key; }

  Key(KeyStr key) : kind(Kind::Str) { value.as_str = key; }

  operator KeyInt() const {
    assert(kind == Key::Kind::Int);
    return value.as_int;
  }

  operator KeyStr() const {
    assert(kind == Key::Kind::Str);
    return value.as_str;
  }

  static bool eq(KeyStr l, KeyStr r) { return strcmp(l, r) == 0; }
  static bool eq(KeyInt l, KeyInt r) { return l == r; }
};

template <typename T> struct ContainerHandle;

template <typename T> struct Container final {
  Kind kind;
  size_t size;

  struct List {
    // owning pointers
    ContainerHandle<T> *items;
  } list;

  struct Dict {
    // owning pointers
    Key *keys;
    // owning pointers
    ContainerHandle<T> *values;
  } dict;

  // non-owning pointer to leaf object
  T *leaf;

  Container(Kind kind, size_t size) : kind(kind), size(size) {}
  Container(T *leaf) : kind(Kind::Leaf), size(1u), leaf(leaf) {}
  ~Container() {
    switch (kind) {
    case Kind::List:
      delete[] list.items;
      break;
    case Kind::Dict:
      delete[] dict.keys;
      delete[] dict.values;
      break;
    case Kind::None:
    case Kind::Leaf:
      break;
    }
  }
  Container(const Container &) = delete;
  Container &operator=(const Container) = delete;
  Container(Container &&from) {
    list = from.list;
    dict = from.dict;
    leaf = from.leaf;
    size = from.size;

    from.kind = Kind::None;
    from.size = 0u;
  }

  Container &&operator=(Container &&from) {
    list = from.list;
    dict = from.dict;
    leaf = from.leaf;
    size = from.size;

    from.kind = Kind::None;
    from.size = 0u;
  }
};

template <typename T> struct ContainerHandle {
  Container<T> *handle;

  ContainerHandle() {}
  ContainerHandle(Container<T> *c) : handle(c) {}
  ~ContainerHandle() {
    if (handle) {
      delete handle;
    }
  }

  ContainerHandle(const ContainerHandle &) = delete;
  ContainerHandle &operator=(const ContainerHandle &) = delete;

  ContainerHandle(ContainerHandle &&from) : handle(from.handle) {
    from.handle = nullptr;
  }
  ContainerHandle &operator=(ContainerHandle &&from) {
    handle = from.handle;
    from.handle = nullptr;
    return *this;
  }

  operator T() const {
    assert(handle->kind == Kind::Leaf);
    return *handle->leaf;
  }

  T &leaf() const {
    assert(handle->kind == Kind::Leaf);
    return *handle->leaf;
  }

  ContainerHandle &operator[](size_t idx) const {
    assert(handle->kind == Kind::List);
    assert(idx < handle->size);
    return handle->list.items[idx];
  }

  bool contains(KeyStr key) const {
    assert(isDict());
    for (size_t i = 0; i < handle->size; ++i) {
      if (Container<T>::Dict::key_eq(handle->dict.keys[i], key)) {
        return true;
      }
    }
    return false;
  }

  template <typename U, typename K>
  const ContainerHandle &at(U lookup_key, K kind) const {
    assert(isDict());
    for (size_t i = 0; i < handle->size; ++i) {
      Key &key = handle->dict.keys[i];
      if (key.kind == kind && Key::eq(key, lookup_key)) {
        return handle->dict.values[i];
      }
    }
    assert(false);
  }

  const ContainerHandle &at(KeyInt lookup_key) const {
    return at(lookup_key, Key::Kind::Int);
  }

  const ContainerHandle &at(KeyStr lookup_key) const {
    return at(lookup_key, Key::Kind::Str);
  }

  const Key &key(size_t idx) const {
    assert(isDict());
    return handle->dict.keys[idx];
  }

  const ContainerHandle &value(size_t idx) const {
    assert(isDict());
    return handle->dict.values[idx];
  }

  size_t size() const { return handle->size; }

  bool isDict() const { return handle->kind == Kind::Dict; }

  bool isList() const { return handle->kind == Kind::List; }

  bool isLeaf() const { return handle->kind == Kind::Leaf; }
};

struct Inflator {
  explicit Inflator(Spec spec);
  template <typename T> ContainerHandle<T> unflatten(T *leaves) {
    return unflatten_internal(leaves, 0);
  };

private:
  struct Layout {
    size_t child_num;
    size_t *leaves_num;
  };
  size_t read_number(size_t &read_idx);
  Layout read_layout(size_t &read_idx);

  template <typename T>
  ContainerHandle<T> unflatten_internal(T *leaves, size_t read_idx) {

    switch (spec_[read_idx]) {
    case 'T':
    case 'L': {
      read_idx++;
      auto layout = read_layout(read_idx);
      const auto size = layout.child_num;
      auto c = new Container<T>(Kind::List, size);
      c->list = {new ContainerHandle<T>[size]};

      size_t child_idx = 0;
      size_t leaves_offset = 0;

      while (spec_[read_idx] != ')') {
        auto next_delim_idx = spec_data_.idxs[read_idx];
        read_idx++;
        c->list.items[child_idx] =
            unflatten_internal(leaves + leaves_offset, read_idx);
        read_idx = next_delim_idx;
        leaves_offset += layout.leaves_num[child_idx++];
      }
      return c;
    }

    case 'D': {
      read_idx++;
      auto layout = read_layout(read_idx);
      const auto size = layout.child_num;
      auto c = new Container<T>(Kind::Dict, size);
      c->dict = {new Key[size], new ContainerHandle<T>[size]};

      size_t child_idx = 0;
      size_t leaves_offset = 0;

      while (spec_[read_idx] != ')') {
        auto next_delim_idx = spec_data_.idxs[read_idx];
        read_idx++;
        if (spec_[read_idx] == '"') {
          auto key_delim_idx = spec_data_.idxs[read_idx];
          read_idx++;
          size_t key_len = key_delim_idx - read_idx;

          auto key = new char[key_len + 1];
          memcpy(key, spec_ + read_idx, key_len);
          key[key_len] = '\0';

          c->dict.keys[child_idx] = key;
          read_idx = key_delim_idx + 2;
        } else {
          assert(isdigit(spec_[read_idx]));
          size_t key = read_number(read_idx);
          c->dict.keys[child_idx] = key;
          read_idx += 1;
        }

        c->dict.values[child_idx] =
            unflatten_internal(leaves + leaves_offset, read_idx);
        read_idx = next_delim_idx;
        leaves_offset += layout.leaves_num[child_idx++];
      }
      return c;
    }

    case '$':
      return new Container<T>(leaves);
    }
    assert(false);
  }
  Spec spec_;
  struct SpecData {
    size_t *idxs;
    size_t containers_num;
  } spec_data_;
};

} // namespace pytree
} // namespace executor
} // namespace torch
