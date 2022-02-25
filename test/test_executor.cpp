#include <gtest/gtest.h>
#include <core/tensor.h>
#include <core/value.h>
#include <core/operator_registry.h>
#include <schema_generated.h>
#include <unordered_map>
#include <executor.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



// Tests go in torch::jit
namespace torch {
namespace executor {

struct Serializer {
  flatbuffers::Offset<executorch::Tensor> tensorToFB(
      flatbuffers::FlatBufferBuilder& fbb,
      const Value& ivalue) {
    auto tensor = ivalue.toTensor();
    // If there's no data in that tensor, the buffer_index is 0
    int buffer_index = 0;
    if (tensor->data) {
      auto it = memoized_storage_map_.find(tensor->data);
      if (it != memoized_storage_map_.end()) {
        buffer_index = it->second;
      }
      else {
        memoized_storage_map_[tensor->data] = ++buffer_index_;
        tensor_data_.push_back(tensor);
        buffer_index = buffer_index_;
      }
    }

    std::vector<int> sizes{&tensor->sizes[0], &tensor->sizes[0] + tensor->dim};

    return executorch::CreateTensorDirect(
        fbb,
        buffer_index,
        static_cast<int8_t>(tensor->type),
        0, // hard code storage offset for now. It's tensor's local storage offset, not related to serialization
        &sizes,
        0,
        false, // requires grad
        0);
  }

  flatbuffers::Offset<executorch::Value> valueToFB(flatbuffers::FlatBufferBuilder& fbb, const Value& ivalue) {
    flatbuffers::Offset<void> offset = 0;
    executorch::ValueUnion ivalue_type = executorch::ValueUnion::NONE;
    if (ivalue.tag == Tag::Tensor) {

      ivalue_type = executorch::ValueUnion::Tensor;
      offset = tensorToFB(fbb, ivalue).Union();
    } else {
      // TODO: Support other types.
      error_with_message("Type not supported yet.");
    }
    return CreateValue(fbb, ivalue_type, offset);
  }

  uint32_t storeValueAndGetIndex(
      flatbuffers::FlatBufferBuilder& fbb,
      const Value& ivalue) {
    auto offset = valueToFB(fbb, ivalue);
    uint32_t size = value_offsets_.size();
    value_offsets_.push_back(offset);
    return size;
  }

  void serializeValues(flatbuffers::FlatBufferBuilder& fbb, const std::vector<Value>& values) {
    for (const auto& v : values) {
      storeValueAndGetIndex(fbb, v);
    }

    // store buffers
    // Buffer 0 is fixed for RW values
    auto buffer_offset = executorch::CreateBuffer(
        fbb,
        fbb.CreateVector(
            reinterpret_cast<const uint8_t*>(0),
            0));
    buffer_offsets_.push_back(buffer_offset);
    for (auto td : tensor_data_) {
      fbb.ForceVectorAlignment(
          td->nbytes, sizeof(uint8_t), FLATBUFFERS_MAX_ALIGNMENT);
      auto buffer_offset = executorch::CreateBuffer(
          fbb,
          fbb.CreateVector(
              reinterpret_cast<const uint8_t*>(td->data),
              td->nbytes));
      buffer_offsets_.push_back(buffer_offset);
    }
  }

  flatbuffers::DetachedBuffer serializeModule(flatbuffers::FlatBufferBuilder& fbb) {
    // Prepare for a graph
    // It has two operations, mul and add to finish
    // z = a * x
    // y = z * b

    // values
    // TODO: Add other types like IntList or BoolList
    // Values: a, b, x, y and intermediate z (ax), all tensors
    // Constant tensors a and b have data.
    std::vector<Value> values;
    auto a_sizes = new int[]{2, 2};
    int a_data[4]{1, 2, 3, 4};
    Tensor a(ScalarType::Int, 2, a_sizes, a_data);
    values.emplace_back(&a);

    auto b_sizes = new int[]{2, 2};
    int b_data[4]{5, 6, 7, 8};
    Tensor b(ScalarType::Int, 2, b_sizes, b_data);
    values.emplace_back(&b);

    // Rest of tensors (x, z, y) don't have data
    auto x_sizes = new int[]{2, 2};
    Tensor x(ScalarType::Int, 2, x_sizes);

    auto y_sizes = new int[]{2, 2};
    Tensor y(ScalarType::Int, 2, y_sizes);

    auto z_sizes = new int[]{2, 2};
    Tensor z(ScalarType::Int, 2, z_sizes);

    values.emplace_back(&x);
    values.emplace_back(&y);
    values.emplace_back(&z);

    int debugint = values[0].payload.as_tensor->sizes[0];
    serializeValues(fbb, values);

    // operators
    std::vector<flatbuffers::Offset<executorch::Operator>>
        operator_vector;
    operator_vector.push_back(executorch::CreateOperator(
        fbb,
        fbb.CreateSharedString("demo::mul"),
        fbb.CreateSharedString("")));
    operator_vector.push_back(executorch::CreateOperator(
        fbb,
        fbb.CreateSharedString("demo::add"),
        fbb.CreateSharedString("")));

    // 0: a,
    // 1: b,
    // 2: x,
    // 3: y,
    // 4: z

    // Kernels
    std::vector<flatbuffers::Offset<executorch::Kernel>> kernel_vector;
    std::vector<int> op0_args{0, 2, 4};
    kernel_vector.push_back(executorch::CreateKernelDirect(
        fbb,
        0, /* op index, 0 for mul */
        &op0_args
        ));

    std::vector<int> op1_args{4, 1, 3};
    kernel_vector.push_back(executorch::CreateKernelDirect(
        fbb,
        1, /* op index, 0 for add */
        &op1_args
        ));

    std::vector<flatbuffers::Offset<executorch::Chain>> chain_vector;
    std::vector<int> inputs{2}; // x. Q: would a and b counted inputs?
    std::vector<int> outputs{3}; // y.
    chain_vector.push_back(executorch::CreateChainDirect(
        fbb,
        &inputs,
        &outputs,
        &kernel_vector
        ));


    std::vector<flatbuffers::Offset<executorch::ExecutionPlan>> execution_plan_vector;
    execution_plan_vector.push_back(executorch::CreateExecutionPlanDirect(
        fbb,
        &value_offsets_,
        &inputs,
        &outputs,
        &chain_vector,
        &operator_vector
    ));

    auto program_offset = executorch::CreateProgramDirect(
        fbb,
        1, /* version */
        &execution_plan_vector,
        &buffer_offsets_
    );

    fbb.Finish(program_offset);
    return fbb.Release();
  }

  int buffer_index_ = 0; // 0 is hard-coded for RW data
  std::vector<flatbuffers::Offset<executorch::Value>> value_offsets_;
  std::vector<flatbuffers::Offset<executorch::Buffer>>
      buffer_offsets_;
  std::unordered_map<const void*, uint32_t> memoized_storage_map_;
  std::vector<Tensor*> tensor_data_;
};

TEST(ExecutorTest, Tensor) {
  auto sizes = new int[]{2, 2};
  int data[4]{1, 2, 3, 4};
  Tensor a(ScalarType::Int, 2, sizes, data);

  auto data_p = static_cast<int*>(a.data);
  for (int i = 0; i < a.sizes[0]; ++i) {
    for (int j = 0; j < a.sizes[1]; ++j) {
      printf("a[%d, %d] = %d\n", i, j, data_p[2 * i + j]);
    }
  }
}

TEST(ExecutorTest, Value) {
  auto sizes = new int[]{2, 2};
  int data[4]{1, 2, 3, 4};
  Tensor a(ScalarType::Int, 2, sizes, data);

  Value v(&a);
  ASSERT_TRUE(v.isTensor());
  ASSERT_EQ(v.toTensor()->nbytes, 16);
}

TEST(ExecutorTest, Serialize) {
  flatbuffers::FlatBufferBuilder fbb;
  Serializer serializer;
  auto buff = serializer.serializeModule(fbb);
  auto program = executorch::GetProgram(buff.data());

  ASSERT_EQ(program->execution_plan()->Length(), 1);
  auto operators = program->execution_plan()->Get(0)->operators();
  ASSERT_EQ(operators->Length(), 2);
  ASSERT_EQ(operators->Get(1)->name()->str(), "demo::add");

  auto values = program->execution_plan()->Get(0)->values();
  ASSERT_EQ(values->Length(), 5);
  auto b = values->Get(1)->val_as_Tensor();
  ASSERT_EQ(b->sizes()->Length(), 2);
  ASSERT_EQ(b->sizes()->Get(0), 2);
  ASSERT_EQ(b->sizes()->Get(1), 2);

  auto buffer = program->buffers()->GetMutableObject(b->buffer_index());
  void* ptr = static_cast<void*>(buffer->mutable_data()->data());
  auto d_ptr = static_cast<int*>(ptr);
  ASSERT_EQ(d_ptr[3], 8);
}

TEST(ExecutorTest, Load) {
  flatbuffers::FlatBufferBuilder fbb;
  Serializer serializer;
  auto buff = serializer.serializeModule(fbb);
  auto program = executorch::GetProgram(buff.data());
  Executor executor(program);
  executor.init_execution_plan(0);

  const auto& plan = executor.executionPlan();
  ASSERT_EQ(plan.n_value_, 5);
  Tensor* b = plan.values_[1].toTensor();
  ASSERT_EQ(b->type, ScalarType::Int);
  ASSERT_EQ(b->dim, 2);
  auto d_ptr = static_cast<int*>(b->data);
  ASSERT_EQ(d_ptr[3], 8);

  ASSERT_EQ(plan.n_chains_, 1);
  ASSERT_EQ(plan.chains_[0].n_kernels_, 2);
  ASSERT_EQ(plan.chains_[0].kernels_[0].n_args_, 3);
  ASSERT_EQ(plan.chains_[0].kernels_[0].op_index_, 0);
}

TEST(ExecutorTest, Registry) {
  auto func = getOpsFn("demo::add");
  ASSERT_TRUE(func);

  Value* values = new Value[3];

  auto a_sizes = new int[]{2, 2};
  int a_data[4]{1, 2, 3, 4};
  Tensor a(ScalarType::Int, 2, a_sizes, a_data);
  values[0] = Value(&a);

  auto b_sizes = new int[]{2, 2};
  int b_data[4]{5, 6, 7, 8};
  Tensor b(ScalarType::Int, 2, b_sizes, b_data);
  values[1] = Value(&b);

  auto c_sizes = new int[]{2, 2};
  int c_data[4]{0, 0, 0, 0};
  Tensor c(ScalarType::Int, 2, c_sizes, c_data);
  values[2] = Value(&c);

  func(values);
  auto d_ptr = static_cast<int*>(c.data);
  ASSERT_EQ(d_ptr[3], 12);
}

TEST(ExecutorTest, Execute) {
  flatbuffers::FlatBufferBuilder fbb;
  Serializer serializer;
  auto buff = serializer.serializeModule(fbb);
  auto program = executorch::GetProgram(buff.data());
  Executor executor(program);
  executor.init_execution_plan(0);

  const auto& plan = executor.executionPlan();

  // Prepare for inputs
  int input_index = plan.serialization_plan_->inputs()->Get(0);
  auto input = plan.values_[input_index];
  auto input_t = input.toTensor();
  auto data_input = static_cast<int*>(input_t->data);
  for (int i = 0; i < 4; ++i) {
    data_input[i] = 1;
  }

  plan.execute();

  // Read output
  int output_index = plan.serialization_plan_->outputs()->Get(0);
  auto output = plan.values_[output_index];
  auto output_t = output.toTensor();
  auto data_output = static_cast<int*>(output_t->data);
  ASSERT_EQ(data_output[0], 6);
  ASSERT_EQ(data_output[1], 8);
  ASSERT_EQ(data_output[2], 10);
  ASSERT_EQ(data_output[3], 12);
}

} // namespace executor
} // namespace torch
