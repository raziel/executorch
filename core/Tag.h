#pragma once

#define TORCH_FORALL_TAGS(_) \
  _(None)                    \
  _(Tensor)                  \
  _(String)                  \
  _(Double)                  \
  _(Int)                     \
  _(Bool)                    \
  _(ListBool)                \
  _(ListDouble)              \
  _(ListInt)                 \
  _(ListTensor)              \
  _(ListScalar)              \

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
  TORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};
