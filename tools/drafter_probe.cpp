// DFlash2 drafter adoption probe (M1a, checkpoint 1).
//
// Opens the drafter GGUF through the extension's public reader and re-derives
// the whole loader contract: every metadata key the in-tree loader requires,
// the diffusion topology rules, the target-layer convention and the exact
// tensor inventory for the configured block count. It answers the first M1a
// question -- does the extension's own surface see everything a vendored
// loader needs -- without a GPU, a target model or a line of forward code.
//
// Usage: flint_drafter_probe DRAFTER.gguf [--tensors] [--loader]
//   exit 0 when the artifact satisfies the contract, 1 otherwise.
//
// --loader additionally runs the vendored loader on the same file and diffs its
// view against this probe's independent parse: same geometry, same taps, same
// tensor inventory with the same element counts and storage classes. The probe
// deliberately re-derives the contract instead of calling the loader, so the two
// disagreeing is a finding in either one.

#include <gufo/ext/gguf_reader.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <map>
#include <utility>
#include <vector>

#include "src/gpu/dflash/dflash_weights.hpp"

namespace gufo::models::flint {
namespace {

using core::GgufReader;
using core::GgmlType;

/// The vendored drafter loader, for the `--loader` differential check.
namespace dflash = gpu::dflash;

struct Spec {
  // Transformer geometry (dflash.*).
  std::uint32_t hidden{0};
  std::uint32_t intermediate{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t rotary_dim{0};
  /// False when the GGUF names no rotary width and head_dim is the fallback
  /// (llama.cpp's rule; the in-tree loader insists on the metadata, which this
  /// export does not carry).
  bool rotary_explicit{true};
  std::uint32_t context_length{0};
  std::uint32_t blocks{0};
  float rope_theta{0.0F};

  // Diffusion topology.
  std::uint32_t block_size{0};
  std::uint32_t conv_kernel{0};
  std::uint32_t conv_group{0};
  std::uint32_t selector_rank{0};
  std::uint32_t selector_top_k{0};
  std::uint32_t sliding_window{0};
  std::uint32_t mask_token{0};
  bool causal{true};

  // Derived.
  std::uint32_t vocab{0};
  /// Zero-based layer outputs, after the GGUF layer-input convention is undone.
  std::vector<std::uint32_t> taps;
};

/// One expected tensor: the loader's inventory, as a list.
struct Expected {
  std::string name;
  std::uint64_t elements{0};
  bool matrix{false};
};

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
  return false;
}

std::optional<std::uint32_t> Uint32(const GgufReader& reader,
                                    std::string_view key) {
  return reader.GetMetadataUint32(key);
}

/// The loader accepts either an explicit rotary dimension or the degenerate
/// four-section M-RoPE layout, whose first section is half the rotary width.
std::optional<std::uint32_t> RotaryDim(const GgufReader& reader) {
  if (const auto value = reader.GetMetadataUint32("dflash.rope.dimension_count"))
    return value;
  const auto* metadata = reader.FindMetadata("dflash.rope.dimension_sections");
  if (metadata == nullptr)
    return std::nullopt;
  const auto* unsigned_values =
      std::get_if<std::vector<std::uint64_t>>(&metadata->value);
  const auto* signed_values =
      std::get_if<std::vector<std::int64_t>>(&metadata->value);
  std::vector<std::uint64_t> sections;
  if (unsigned_values != nullptr) {
    sections = *unsigned_values;
  } else if (signed_values != nullptr) {
    for (const auto value : *signed_values) {
      if (value < 0)
        return std::nullopt;
      sections.push_back(static_cast<std::uint64_t>(value));
    }
  } else {
    return std::nullopt;
  }
  if (sections.size() != 4 || sections[0] == 0 || sections[1] != 0 ||
      sections[2] != 0 || sections[3] != 0 ||
      sections[0] > std::numeric_limits<std::uint32_t>::max() / 2U) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(sections[0] * 2U);
}

std::optional<std::vector<std::uint64_t>> IntegerArray(const GgufReader& reader,
                                                       std::string_view key) {
  const auto* metadata = reader.FindMetadata(key);
  if (metadata == nullptr)
    return std::nullopt;
  if (const auto* unsigned_values =
          std::get_if<std::vector<std::uint64_t>>(&metadata->value))
    return *unsigned_values;
  if (const auto* signed_values =
          std::get_if<std::vector<std::int64_t>>(&metadata->value)) {
    std::vector<std::uint64_t> converted;
    for (const auto value : *signed_values) {
      if (value < 0)
        return std::nullopt;
      converted.push_back(static_cast<std::uint64_t>(value));
    }
    return converted;
  }
  return std::nullopt;
}

bool ReadSpec(const GgufReader& reader, Spec* spec, std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "dflash")
    return Fail(error_msg, "general.architecture is not 'dflash'");

  const auto hidden = Uint32(reader, "dflash.embedding_length");
  const auto intermediate = Uint32(reader, "dflash.feed_forward_length");
  const auto heads = Uint32(reader, "dflash.attention.head_count");
  const auto kv_heads = Uint32(reader, "dflash.attention.head_count_kv");
  const auto head_dim = Uint32(reader, "dflash.attention.key_length");
  const auto rotary = RotaryDim(reader);
  const auto context = Uint32(reader, "dflash.context_length");
  const auto blocks = Uint32(reader, "dflash.block_count");
  const auto theta = reader.GetMetadataFloat32("dflash.rope.freq_base");
  if (!hidden || !intermediate || !heads || !kv_heads || !head_dim || !context ||
      !blocks || !theta) {
    return Fail(error_msg, "missing transformer geometry metadata");
  }
  spec->hidden = *hidden;
  spec->intermediate = *intermediate;
  spec->heads = *heads;
  spec->kv_heads = *kv_heads;
  spec->head_dim = *head_dim;
  spec->rotary_explicit = rotary.has_value();
  spec->rotary_dim = rotary.value_or(*head_dim);
  spec->context_length = *context;
  spec->blocks = *blocks;
  spec->rope_theta = *theta;
  if (spec->hidden == 0 || spec->intermediate == 0 || spec->blocks == 0 ||
      spec->heads == 0 || spec->kv_heads == 0 || spec->head_dim == 0 ||
      spec->heads % spec->kv_heads != 0 || spec->rotary_dim == 0 ||
      spec->rotary_dim > spec->head_dim || spec->rotary_dim % 2 != 0) {
    return Fail(error_msg, "invalid transformer topology");
  }

  const auto block_size = Uint32(reader, "dflash.block_size");
  const auto conv_kernel = Uint32(reader, "dflash.conv_kernel_size");
  const auto conv_group = Uint32(reader, "dflash.conv_group_size");
  const auto selector_rank = Uint32(reader, "dflash.selector_rank");
  const auto selector_top_k = Uint32(reader, "dflash.selector_top_k");
  const auto sliding_window = Uint32(reader, "dflash.attention.sliding_window");
  const auto mask_token = Uint32(reader, "tokenizer.ggml.mask_token_id");
  const auto causal = reader.GetMetadataBool("dflash.attention.causal");
  if (!block_size || !conv_kernel || !conv_group || !selector_rank ||
      !selector_top_k || !sliding_window || !mask_token || !causal) {
    return Fail(error_msg, "missing diffusion topology metadata");
  }
  spec->block_size = *block_size;
  spec->conv_kernel = *conv_kernel;
  spec->conv_group = *conv_group;
  spec->selector_rank = *selector_rank;
  spec->selector_top_k = *selector_top_k;
  spec->sliding_window = *sliding_window;
  spec->mask_token = *mask_token;
  spec->causal = *causal;
  if (spec->block_size < 2 || spec->conv_kernel == 0 ||
      spec->conv_group == 0 || spec->hidden % spec->conv_group != 0 ||
      spec->selector_rank == 0 || spec->selector_top_k == 0 ||
      spec->selector_top_k > 16 || spec->sliding_window < spec->block_size ||
      spec->causal) {
    return Fail(error_msg, "invalid diffusion topology");
  }

  // The taps are layer *inputs* in the GGUF (llama.cpp's convention); the
  // residual we capture is the layer *output*, one index lower.
  const auto encoded_taps = IntegerArray(reader, "dflash.target_layers");
  if (!encoded_taps || encoded_taps->empty())
    return Fail(error_msg, "missing dflash.target_layers");
  for (const auto encoded : *encoded_taps) {
    if (encoded == 0 ||
        encoded - 1U > std::numeric_limits<std::uint32_t>::max())
      return Fail(error_msg, "target layer index is not a positive input index");
    const auto normalized = static_cast<std::uint32_t>(encoded - 1U);
    if (!spec->taps.empty() && normalized <= spec->taps.back())
      return Fail(error_msg, "target layer indices are not increasing");
    spec->taps.push_back(normalized);
  }

  // The drafter has no vocabulary of its own: it borrows the target's
  // embedding and head, so the selector matrices define the vocabulary.
  if (reader.FindTensor("d2t") != nullptr)
    return Fail(error_msg, "reduced-vocabulary d2t exports are unsupported");
  const auto* predecessor = reader.FindTensor("selector_predecessor.weight");
  if (predecessor == nullptr || predecessor->data == nullptr ||
      predecessor->ElementCount() % spec->selector_rank != 0) {
    return Fail(error_msg, "selector_predecessor.weight is missing or malformed");
  }
  const auto vocab = predecessor->ElementCount() / spec->selector_rank;
  if (vocab == 0 || vocab > std::numeric_limits<std::uint32_t>::max())
    return Fail(error_msg, "selector vocabulary size is invalid");
  spec->vocab = static_cast<std::uint32_t>(vocab);
  return true;
}

bool IsVectorType(GgmlType type) {
  return type == GgmlType::kF32 || type == GgmlType::kBF16;
}

bool IsMatrixType(GgmlType type) {
  // Q5_K is not native to the draft GEMM tier, but the loader accepts it and the
  // upload path host-dequantizes it to BF16, so an artifact that stores it is
  // loadable -- just larger in device memory than Q6_K.
  return IsVectorType(type) || type == GgmlType::kQ8_0 ||
         type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
         type == GgmlType::kQ6_K;
}

std::vector<Expected> Inventory(const Spec& spec) {
  const std::uint64_t hidden = spec.hidden;
  const std::uint64_t attention = std::uint64_t{spec.heads} * spec.head_dim;
  const std::uint64_t kv = std::uint64_t{spec.kv_heads} * spec.head_dim;
  const std::uint64_t groups = hidden / spec.conv_group;
  const std::uint64_t dynamic = 2U * spec.conv_kernel * groups;

  std::vector<Expected> expected{
      {"selector_predecessor.weight", std::uint64_t{spec.vocab} *
                                          spec.selector_rank, true},
      {"selector_successor.weight",
       std::uint64_t{spec.vocab} * spec.selector_rank, true},
      {"selector_hidden.weight", hidden * spec.selector_rank, true},
      {"fc.weight", hidden * spec.taps.size() * hidden, true},
      {"enc.output_norm.weight", hidden, false},
      {"output_norm.weight", hidden, false},
  };
  for (std::uint32_t i = 0; i < spec.blocks; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto add = [&](std::string name, std::uint64_t elements, bool matrix) {
      expected.push_back({prefix + name, elements, matrix});
    };
    add("attn_norm.weight", hidden, false);
    add("attn_q.weight", attention * hidden, true);
    add("attn_k.weight", kv * hidden, true);
    add("attn_v.weight", kv * hidden, true);
    add("attn_output.weight", hidden * attention, true);
    add("attn_q_norm.weight", spec.head_dim, false);
    add("attn_k_norm.weight", spec.head_dim, false);
    add("ffn_norm.weight", hidden, false);
    add("ffn_gate.weight", std::uint64_t{spec.intermediate} * hidden, true);
    add("ffn_up.weight", std::uint64_t{spec.intermediate} * hidden, true);
    add("ffn_down.weight", hidden * spec.intermediate, true);
    add("attn_conv_base", hidden * spec.conv_kernel * 2U, false);
    add("attn_conv_proj.weight", dynamic * hidden, true);
    add("ffn_conv_base", hidden * spec.conv_kernel * 2U, false);
    add("ffn_conv_proj.weight", dynamic * hidden, true);
  }
  return expected;
}

struct Report {
  std::size_t checked{0};
  std::vector<std::string> problems;
};

void CheckInventory(const GgufReader& reader, const Spec& spec, Report* report) {
  const auto expected = Inventory(spec);
  std::vector<bool> seen(expected.size(), false);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& want = expected[i];
    const auto* tensor = reader.FindTensor(want.name);
    if (tensor == nullptr) {
      report->problems.push_back("missing tensor " + want.name);
      continue;
    }
    seen[i] = true;
    ++report->checked;
    if (tensor->ElementCount() != want.elements) {
      report->problems.push_back(
          "tensor " + want.name + " has " +
          std::to_string(tensor->ElementCount()) + " elements, expected " +
          std::to_string(want.elements));
    }
    const bool ok = want.matrix ? IsMatrixType(tensor->type)
                                : IsVectorType(tensor->type);
    if (!ok) {
      report->problems.push_back("tensor " + want.name + " has unsupported type " +
                                 std::string(core::ToString(tensor->type)));
    }
    if (!tensor->sized || tensor->data == nullptr) {
      report->problems.push_back("tensor " + want.name + " is unsized or unmapped");
    }
  }
  // Extra tensors are reported, not rejected: the loader tolerates private
  // copies of the tied embedding and head.
  for (const auto& tensor : reader.GetTensors()) {
    bool claimed = false;
    for (std::size_t i = 0; i < expected.size(); ++i)
      claimed = claimed || (seen[i] && expected[i].name == tensor.name);
    if (!claimed) {
      report->problems.push_back("unexpected tensor " +
                                 std::string(tensor.name) + " (" +
                                 std::string(core::ToString(tensor.type)) + ", " +
                                 std::to_string(tensor.ElementCount()) +
                                 " elements)");
    }
  }
}

void PrintTensorHistogram(const GgufReader& reader) {
  std::size_t vectors = 0;
  std::size_t matrices = 0;
  std::uint64_t bytes = 0;
  for (const auto& tensor : reader.GetTensors()) {
    bytes += tensor.size_bytes;
    if (IsVectorType(tensor.type)) {
      ++vectors;
    } else {
      ++matrices;
    }
  }
  std::printf("| tensors | vectors | matrices | storage MiB |\n"
              "| --: | --: | --: | --: |\n"
              "| %zu | %zu | %zu | %.1f |\n",
              reader.GetTensors().size(), vectors, matrices,
              static_cast<double>(bytes) / (1024.0 * 1024.0));
}

void PrintTensors(const GgufReader& reader) {
  std::printf("\n| tensor | type | elements | bytes |\n| :-- | :-- | --: | --: |\n");
  for (const auto& tensor : reader.GetTensors()) {
    std::printf("| %s | %s | %llu | %zu |\n", std::string(tensor.name).c_str(),
                std::string(core::ToString(tensor.type)).c_str(),
                static_cast<unsigned long long>(tensor.ElementCount()),
                tensor.size_bytes);
  }
}

/// Role name -> tensor, for the diff against this probe's inventory.
std::map<std::string, const dflash::DflashTensor*> Roles(
    const dflash::DflashWeights& weights) {
  std::map<std::string, const dflash::DflashTensor*> roles{
      {"selector_predecessor.weight", &weights.selector_predecessor},
      {"selector_successor.weight", &weights.selector_successor},
      {"selector_hidden.weight", &weights.selector_hidden},
      {"fc.weight", &weights.fc_projection},
      {"enc.output_norm.weight", &weights.fc_norm},
      {"output_norm.weight", &weights.output_norm},
  };
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto& layer = weights.layers[i];
    roles[prefix + "attn_norm.weight"] = &layer.attn_norm;
    roles[prefix + "attn_q.weight"] = &layer.attn_q;
    roles[prefix + "attn_k.weight"] = &layer.attn_k;
    roles[prefix + "attn_v.weight"] = &layer.attn_v;
    roles[prefix + "attn_output.weight"] = &layer.attn_output;
    roles[prefix + "attn_q_norm.weight"] = &layer.attn_q_norm;
    roles[prefix + "attn_k_norm.weight"] = &layer.attn_k_norm;
    roles[prefix + "ffn_norm.weight"] = &layer.ffn_norm;
    roles[prefix + "ffn_gate.weight"] = &layer.ffn_gate;
    roles[prefix + "ffn_up.weight"] = &layer.ffn_up;
    roles[prefix + "ffn_down.weight"] = &layer.ffn_down;
    roles[prefix + "attn_conv_base"] = &layer.attn_conv_base;
    roles[prefix + "attn_conv_proj.weight"] = &layer.attn_conv_projection;
    roles[prefix + "ffn_conv_base"] = &layer.ffn_conv_base;
    roles[prefix + "ffn_conv_proj.weight"] = &layer.ffn_conv_projection;
  }
  return roles;
}

/// Compares the vendored loader's view of the file with the probe's own.
bool CheckLoader(const GgufReader& reader, const Spec& spec, Report* report) {
  std::string error;
  const auto weights = dflash::DflashWeights::LoadFromGguf(reader, &error);
  if (!weights.has_value()) {
    report->problems.push_back("loader rejected the file: " + error);
    return false;
  }
  const auto& config = weights->config;

  const auto compare = [&](auto got, auto want, const char* label) {
    if (got != want)
      report->problems.push_back(std::string("loader ") + label + ": " +
                                 std::to_string(got) + ", probe says " +
                                 std::to_string(want));
  };
  compare(config.hidden_size, spec.hidden, "hidden");
  compare(config.intermediate_size, spec.intermediate, "intermediate");
  compare(config.num_attention_heads, spec.heads, "heads");
  compare(config.num_key_value_heads, spec.kv_heads, "kv heads");
  compare(config.head_dim, spec.head_dim, "head dim");
  compare(config.rotary_dim, spec.rotary_dim, "rotary dim");
  compare(config.context_length, spec.context_length, "context length");
  compare(config.num_layers, spec.blocks, "blocks");
  compare(config.vocab_size, spec.vocab, "vocabulary");
  compare(config.block_size, spec.block_size, "block size");
  compare(config.mask_token_id, spec.mask_token, "mask token");
  compare(config.conv_kernel_size, spec.conv_kernel, "conv kernel");
  compare(config.conv_group_size, spec.conv_group, "conv group");
  compare(config.selector_rank, spec.selector_rank, "selector rank");
  compare(config.selector_top_k, spec.selector_top_k, "selector top-k");
  compare(config.sliding_window, spec.sliding_window, "sliding window");
  compare(config.rotary_from_head_dim, !spec.rotary_explicit,
          "rotary fallback flag");
  if (config.target_layer_ids != spec.taps)
    report->problems.push_back("loader taps differ from the probe's");
  if (static_cast<float>(config.rope_theta) != spec.rope_theta)
    report->problems.push_back("loader rope theta differs");

  // Inventory: every tensor the probe expects must be bound to a role with the
  // same element count and the same storage class, and nothing else may be.
  const auto roles = Roles(*weights);
  const auto expected = Inventory(spec);
  std::size_t bound = 0;
  for (const auto& want : expected) {
    const auto role = roles.find(want.name);
    if (role == roles.end()) {
      report->problems.push_back("loader has no role for " + want.name);
      continue;
    }
    ++bound;
    const auto& tensor = *role->second;
    if (tensor.num_elements != want.elements) {
      report->problems.push_back(
          "loader " + want.name + " has " +
          std::to_string(tensor.num_elements) + " elements, probe expects " +
          std::to_string(want.elements));
    }
    const bool ok = want.matrix ? IsMatrixType(tensor.type)
                                : IsVectorType(tensor.type);
    if (!ok) {
      report->problems.push_back("loader " + want.name +
                                 " has unexpected storage class " +
                                 std::string(core::ToString(tensor.type)));
    }
    if (tensor.empty() || !tensor.FitsAvailableStorage()) {
      report->problems.push_back("loader " + want.name +
                                 " is empty or overruns its mapping");
    }
  }
  if (bound != expected.size())
    report->problems.push_back("loader bound " + std::to_string(bound) +
                               " of " + std::to_string(expected.size()) +
                               " expected tensors");
  const std::size_t role_count = roles.size();
  if (role_count != expected.size())
    report->problems.push_back("loader has " + std::to_string(role_count) +
                               " roles, probe expects " +
                               std::to_string(expected.size()));
  return true;
}

}  // namespace

int RunDrafterProbe(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: flint_drafter_probe DRAFTER.gguf [--tensors] "
                 "[--loader]\n");
    return 2;
  }
  bool list_tensors = false;
  bool use_loader = false;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--tensors") == 0) {
      list_tensors = true;
    } else if (std::strcmp(argv[i], "--loader") == 0) {
      use_loader = true;
    } else {
      std::fprintf(stderr, "flint_drafter_probe: unknown option %s\n",
                   argv[i]);
      return 2;
    }
  }

  std::string error;
  const auto reader = GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "Flint drafter probe: %s: %s\n", argv[1],
                 error.c_str());
    return 1;
  }

  Spec spec;
  if (!ReadSpec(*reader, &spec, &error)) {
    std::fprintf(stderr, "Flint drafter probe: %s: %s\n", argv[1], error.c_str());
    return 1;
  }

  std::printf("# DFlash2 drafter: %s\n\n", argv[1]);
  std::printf("| quantity | value |\n| :-- | --: |\n");
  std::printf("| hidden | %u |\n", spec.hidden);
  std::printf("| intermediate | %u |\n", spec.intermediate);
  std::printf("| heads / kv heads / head dim | %u / %u / %u |\n", spec.heads,
              spec.kv_heads, spec.head_dim);
  std::printf("| rotary dim / theta | %u / %.1f |\n", spec.rotary_dim,
              static_cast<double>(spec.rope_theta));
  std::printf("| rotary dim source | %s |\n",
              spec.rotary_explicit ? "dflash.rope metadata"
                                   : "head_dim fallback (no rotor metadata)");
  std::printf("| context length | %u |\n", spec.context_length);
  std::printf("| blocks | %u |\n", spec.blocks);
  std::printf("| block / mask token | %u / %u |\n", spec.block_size,
              spec.mask_token);
  std::printf("| conv kernel / group | %u / %u |\n", spec.conv_kernel,
              spec.conv_group);
  std::printf("| selector rank / top-k | %u / %u |\n", spec.selector_rank,
              spec.selector_top_k);
  std::printf("| sliding window | %u |\n", spec.sliding_window);
  std::printf("| causal | %s |\n", spec.causal ? "yes" : "no");
  std::printf("| vocabulary (from selector) | %u |\n", spec.vocab);
  std::printf("| taps (GGUF layer inputs -> captured outputs) | ");
  for (std::size_t i = 0; i < spec.taps.size(); ++i)
    std::printf("%s%u->%u", i == 0 ? "" : ", ", spec.taps[i] + 1, spec.taps[i]);
  std::printf(" |\n\n");
  PrintTensorHistogram(*reader);

  Report report;
  CheckInventory(*reader, spec, &report);
  const bool loader_ran = use_loader && CheckLoader(*reader, spec, &report);
  std::printf("\n| check | result |\n| :-- | :-- |\n");
  std::printf("| tensors checked | %zu |\n", report.checked);
  if (use_loader)
    std::printf("| vendored loader | %s |\n",
                loader_ran && report.problems.empty() ? "agrees"
                                                      : "differs or rejected");
  std::printf("| problems | %zu |\n", report.problems.size());
  if (!report.problems.empty()) {
    std::printf("\n");
    for (const auto& problem : report.problems)
      std::printf("- %s\n", problem.c_str());
  }
  if (list_tensors)
    PrintTensors(*reader);

  if (!report.problems.empty())
    return 1;
  std::printf("\nDrafter contract satisfied.\n");
  return 0;
}

}  // namespace gufo::models::flint

int main(int argc, char** argv) {
  return gufo::models::flint::RunDrafterProbe(argc, argv);
}
