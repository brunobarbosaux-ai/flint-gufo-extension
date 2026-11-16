// DFlash2 draft forward smoke test.
//
// The drafter's weights are synthetic: every tensor is a small F32 matrix with
// deterministic values, so the test needs no artifact, no target model and no
// oracle. What it covers is the wiring the block forward owns -- tensor upload
// and role binding, the context ring, block/row bookkeeping, the trace hook the
// parity harness reads, and the selector's interface. Numeric agreement with the
// in-tree executor is the parity step's job, not this test's.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gufo/ext/experimental/ggml_dequant.hpp"
#include "src/gpu/dflash/dflash_forward.hpp"

namespace dflash = gufo::models::flint::gpu::dflash;
namespace core = gufo::core;
namespace quant = gufo::quant;

namespace {

void Check(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

/// A host tensor with deterministic values and a plausible mapped size.
struct HostTensor {
  std::vector<float> values;
  dflash::DflashTensor ref;
};

HostTensor MakeTensor(std::size_t elements) {
  HostTensor tensor;
  tensor.values.resize(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    // Small, distinct, symmetric around zero: enough structure for the stages
    // to produce finite values without saturating.
    tensor.values[i] =
        static_cast<float>(static_cast<int>(i % 23) - 11) * 0.03125F;
  }
  tensor.ref = {.data = tensor.values.data(),
                .type = core::GgmlType::kF32,
                .num_elements = elements,
                .available_bytes = quant::EncodedSizeBytes(core::GgmlType::kF32,
                                                           elements)};
  return tensor;
}

/// One synthetic drafter, small enough to print when it breaks.
struct Fixture {
  dflash::DflashConfig config;
  dflash::DflashWeights weights;
  std::vector<HostTensor> storage;
  std::vector<float> embedding, head;
  float* embedding_device{nullptr};
  float* head_device{nullptr};

  explicit Fixture(std::uint32_t layers = 2) {
    config.hidden_size = 64;
    config.intermediate_size = 96;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.head_dim = 16;
    config.rotary_dim = 16;
    config.rope_theta = 10000.0F;
    config.context_length = 4096;
    config.num_layers = layers;
    config.vocab_size = 128;
    config.target_layer_ids = {0, 1};
    config.block_size = 8;
    config.mask_token_id = 127;
    config.conv_kernel_size = 2;
    config.conv_group_size = 16;
    config.selector_rank = 8;
    config.selector_top_k = 4;
    config.sliding_window = 64;
    config.rotary_from_head_dim = true;
    weights.config = config;

    const std::size_t hidden = config.hidden_size;
    const std::size_t intermediate = config.intermediate_size;
    const std::size_t attention = config.AttentionSize();
    const std::size_t kv = config.KvSize();
    const std::size_t dynamic = config.ConvCoefficients();

    const auto bind = [&](std::size_t elements) -> dflash::DflashTensor {
      storage.push_back(MakeTensor(elements));
      return storage.back().ref;
    };
    weights.fc_projection = bind(hidden * config.target_layer_ids.size() *
                                 hidden);
    weights.fc_norm = bind(hidden);
    weights.output_norm = bind(hidden);
    weights.selector_predecessor = bind(std::size_t{config.vocab_size} *
                                        config.selector_rank);
    weights.selector_successor = bind(std::size_t{config.vocab_size} *
                                      config.selector_rank);
    weights.selector_hidden = bind(hidden * config.selector_rank);
    weights.layers.resize(config.num_layers);
    for (auto& layer : weights.layers) {
      layer.attn_norm = bind(hidden);
      layer.attn_q = bind(attention * hidden);
      layer.attn_k = bind(kv * hidden);
      layer.attn_v = bind(kv * hidden);
      layer.attn_output = bind(hidden * attention);
      layer.attn_q_norm = bind(config.head_dim);
      layer.attn_k_norm = bind(config.head_dim);
      layer.ffn_norm = bind(hidden);
      layer.ffn_gate = bind(intermediate * hidden);
      layer.ffn_up = bind(intermediate * hidden);
      layer.ffn_down = bind(hidden * intermediate);
      layer.attn_conv_base = bind(hidden * config.conv_kernel_size * 2U);
      layer.attn_conv_projection = bind(dynamic * hidden);
      layer.ffn_conv_base = bind(hidden * config.conv_kernel_size * 2U);
      layer.ffn_conv_projection = bind(dynamic * hidden);
    }

    embedding.assign(std::size_t{config.vocab_size} * hidden, 0.0F);
    for (std::size_t i = 0; i < embedding.size(); ++i)
      embedding[i] =
          static_cast<float>(static_cast<int>(i % 29) - 14) * 0.015625F;
    head.assign(std::size_t{config.vocab_size} * hidden, 0.0F);
    for (std::size_t i = 0; i < head.size(); ++i)
      head[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.0078125F;
    // The borrowed weights live in device memory, like the engine's.
    Check(hipMalloc(&embedding_device, embedding.size() * sizeof(float)),
          "embedding allocation");
    Check(hipMalloc(&head_device, head.size() * sizeof(float)),
          "head allocation");
    Check(hipMemcpy(embedding_device, embedding.data(),
                    embedding.size() * sizeof(float), hipMemcpyHostToDevice),
          "embedding upload");
    Check(hipMemcpy(head_device, head.data(), head.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "head upload");
  }

  ~Fixture() {
    (void)hipFree(embedding_device);
    (void)hipFree(head_device);
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  [[nodiscard]] dflash::DflashBorrowed Borrowed() const {
    return {.token_embedding = embedding_device,
            .token_embedding_type = core::GgmlType::kF32,
            .output = head_device,
            .output_type = core::GgmlType::kF32};
  }

  /// Feature rows in the drafter's layout: [row][tap][hidden].
  [[nodiscard]] std::vector<float> Features(std::uint32_t rows) const {
    std::vector<float> features(std::size_t{rows} *
                                config.target_layer_ids.size() *
                                config.hidden_size);
    for (std::size_t i = 0; i < features.size(); ++i)
      features[i] =
          static_cast<float>(static_cast<int>(i % 31) - 15) * 0.03125F;
    return features;
  }
};

std::unique_ptr<dflash::DflashForward> MakeForward(const Fixture& fixture,
                                                   std::string* error) {
  auto forward = dflash::DflashForward::Create(
      fixture.weights, fixture.Borrowed(), 1024, error);
  if (forward == nullptr)
    throw std::runtime_error("Create failed: " + *error);
  return forward;
}

void TestUploadAndConfig() {
  Fixture fixture;
  std::string error;
  auto forward = MakeForward(fixture, &error);
  const auto& config = forward->config();
  if (config.hidden_size != fixture.config.hidden_size ||
      config.vocab_size != fixture.config.vocab_size ||
      config.num_layers != fixture.config.num_layers ||
      config.target_layer_ids != fixture.config.target_layer_ids)
    throw std::runtime_error("config did not survive Create");
  if (forward->HistoryCapacity() != fixture.config.sliding_window)
    throw std::runtime_error("context ring is not the attention window");
  if (forward->ContextLength() != 0)
    throw std::runtime_error("a fresh drafter has context");
  for (std::uint32_t layer = 0; layer < config.num_layers; ++layer) {
    if (forward->InjectedKey(layer) == nullptr ||
        forward->InjectedValue(layer) == nullptr)
      throw std::runtime_error("injected rings are missing");
  }
}

void TestContextAndBlock() {
  Fixture fixture;
  std::string error;
  auto forward = MakeForward(fixture, &error);

  // Two chunks of rows, so the injection loop runs more than once.
  constexpr std::uint32_t kRows = 40;
  const auto features = fixture.Features(kRows);
  if (!forward->AppendContext(features, 0, kRows, &error))
    throw std::runtime_error("AppendContext failed: " + error);
  if (forward->ContextLength() != kRows)
    throw std::runtime_error("context length did not advance");

  std::vector<std::string> stages;
  std::size_t logits = 0;
  const auto trace = [&](std::string_view name, const float*, std::size_t count) {
    stages.emplace_back(name);
    if (name == "logits")
      logits = count;
  };

  dflash::DflashDraft draft;
  if (!forward->Run(/*anchor=*/5, /*position=*/kRows, /*draft_count=*/7, 0.0F,
                    {}, trace, &draft, &error))
    throw std::runtime_error("Run failed: " + error);
  if (draft.tokens.size() != 7)
    throw std::runtime_error("Run returned the wrong proposal count");
  for (const auto token : draft.tokens) {
    if (token >= fixture.config.vocab_size)
      throw std::runtime_error("proposal is outside the vocabulary");
  }
  // The trace the parity harness reads must cover the whole block, in order.
  const std::vector<std::string> required{
      "embedding", "layer.0.attn_norm", "layer.0.attn_conv_in", "layer.0.q",
      "layer.0.k", "layer.0.v", "layer.0.attention", "layer.0.attn_output",
      "layer.0.attn_conv_out", "layer.0.attn_residual", "layer.0.ffn_norm",
      "layer.0.ffn_conv_in", "layer.0.ffn_down", "layer.0.ffn_conv_out",
      "layer.0.output", "layer.1.output", "normalized", "selector_hidden",
      "logits"};
  for (const auto& name : required) {
    bool found = false;
    for (const auto& stage : stages)
      found = found || stage == name;
    if (!found)
      throw std::runtime_error("trace is missing stage " + name);
  }
  if (logits != std::size_t{7} * fixture.config.vocab_size)
    throw std::runtime_error("logits stage has the wrong size");
  for (const auto confidence : draft.confidences) {
    if (!std::isfinite(confidence) || confidence < 0.0F || confidence > 1.0F)
      throw std::runtime_error("greedy confidence is not a probability");
  }
  if (!draft.candidate_ids.empty())
    throw std::runtime_error("greedy drafting produced candidates");

  // A block is a pure function of the state appended so far: nothing in it is
  // committed, so a second identical call must return the same proposals.
  dflash::DflashDraft again;
  if (!forward->Run(5, kRows, 7, 0.0F, {}, {}, &again, &error))
    throw std::runtime_error("second Run failed: " + error);
  if (again.tokens != draft.tokens)
    throw std::runtime_error("a block is not deterministic");
  if (forward->ContextLength() != kRows)
    throw std::runtime_error("Run mutated the injected context");
}

void TestSampledDraft() {
  Fixture fixture;
  std::string error;
  auto forward = MakeForward(fixture, &error);
  constexpr std::uint32_t kRows = 20;
  const auto features = fixture.Features(kRows);
  if (!forward->AppendContext(features, 0, kRows, &error))
    throw std::runtime_error("AppendContext failed: " + error);

  const std::vector<float> uniforms{0.05F, 0.25F, 0.5F, 0.75F, 0.95F};
  dflash::DflashDraft draft;
  if (!forward->Run(3, kRows, 5, 0.8F, uniforms, {}, &draft, &error))
    throw std::runtime_error("sampled Run failed: " + error);
  const std::size_t expected =
      std::size_t{5} * fixture.config.selector_top_k;
  if (draft.tokens.size() != 5 || draft.candidate_ids.size() != expected ||
      draft.candidate_probabilities.size() != expected)
    throw std::runtime_error("sampled draft returned the wrong shape");
  for (std::size_t row = 0; row < 5; ++row) {
    double sum = 0.0;
    for (std::uint32_t c = 0; c < fixture.config.selector_top_k; ++c) {
      const auto probability =
          draft.candidate_probabilities[row * fixture.config.selector_top_k + c];
      if (!std::isfinite(probability) || probability < 0.0F ||
          probability > 1.0F)
        throw std::runtime_error("candidate probability is not a probability");
      sum += probability;
      if (draft.candidate_ids[row * fixture.config.selector_top_k + c] >=
          fixture.config.vocab_size)
        throw std::runtime_error("candidate is outside the vocabulary");
    }
    if (std::abs(sum - 1.0) > 1e-4)
      throw std::runtime_error("candidate probabilities do not sum to one");
  }
}

void TestClampedBlock() {
  Fixture fixture;
  std::string error;
  auto forward = MakeForward(fixture, &error);
  constexpr std::uint32_t kRows = 8;
  const auto features = fixture.Features(kRows);
  if (!forward->AppendContext(features, 0, kRows, &error))
    throw std::runtime_error("AppendContext failed: " + error);
  dflash::DflashDraft draft;
  // One more proposal than a full block can hold is clamped, not rejected.
  if (!forward->Run(1, kRows, fixture.config.block_size, 0.0F, {}, {}, &draft,
                    &error))
    throw std::runtime_error("clamped Run failed: " + error);
  if (draft.tokens.size() != fixture.config.block_size - 1U)
    throw std::runtime_error("block size was not enforced");
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "flint dflash forward tests skipped: no HIP device\n";
    return 0;
  }
  try {
    TestUploadAndConfig();
    TestContextAndBlock();
    TestSampledDraft();
    TestClampedBlock();
  } catch (const std::exception& error) {
    std::cerr << "flint dflash forward tests failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "flint dflash forward tests passed\n";
  return 0;
}
