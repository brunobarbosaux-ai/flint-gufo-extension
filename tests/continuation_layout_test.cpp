#include "continuation_layout.hpp"

#include <iostream>
#include <stdexcept>

namespace f = gufo::models::flint;
namespace cache = gufo::cache;

namespace {

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

f::Config MakeConfig() {
  f::Config c;
  c.num_layers = 6;
  c.num_kv = 2;
  c.head_dim = 64;
  c.vocab_size = 1000;
  c.sliding_window = 512;
  c.is_swa = {true, true, false, true, true, false};
  return c;
}

void TestGeometry() {
  const f::ContinuationLayout layout(MakeConfig(), 262144);
  const auto& components = layout.components();
  Require(components.size() == 2 * 6 + 1, "one K, one V per layer, one logits");
  Require(layout.row_bytes() == 2 * 64 * 2, "row = num_kv * head_dim * f16");
  for (std::size_t i = 0; i < components.size(); ++i)
    Require(components[i].descriptor.id.value == i + 1, "ids are 1-based");

  // Layers 0,1,3,4 slide; 2 and 5 are global.
  for (std::uint32_t layer = 0; layer < 6; ++layer) {
    const bool window = layer != 2 && layer != 5;
    for (std::size_t part = 0; part < 2; ++part) {
      const auto& c = components[layer * 2 + part];
      Require(c.layer == layer, "layer order");
      Require((part == 0) == (c.part == f::ContinuationLayout::Part::kKey),
              "K before V");
      Require(c.descriptor.kind == (window ? cache::ComponentKind::kWindowRows
                                           : cache::ComponentKind::kAppendRows),
              "kind follows the layer");
      Require(c.descriptor.window_rows == (window ? 512U : 0U),
              "window rows only on window components");
      Require(c.descriptor.row_bytes == layout.row_bytes(), "row bytes");
      Require(c.descriptor.rows_per_chunk == 256, "chunk of 256 rows");
      Require(c.descriptor.state_bytes == 0, "rows carry no private bytes");
    }
  }
  const auto& logits = components.back();
  Require(logits.descriptor.kind == cache::ComponentKind::kPrivateState,
          "logits are private state");
  Require(logits.descriptor.state_bytes == 1000 * sizeof(float),
          "logits are vocab floats");
  Require(logits.descriptor.row_bytes == 0 &&
              logits.descriptor.rows_per_chunk == 0,
          "private state has no rows");
  Require(layout.logits_id().value == components.size(), "logits id");
  Require(layout.at(layout.logits_id()).part ==
              f::ContinuationLayout::Part::kLogits,
          "lookup by id");
}

void TestPositions() {
  const f::ContinuationLayout layout(MakeConfig(), 262144);
  const auto positions = layout.Positions(37010);
  Require(positions.size() == layout.components().size(), "one per component");
  for (std::size_t i = 0; i < positions.size(); ++i) {
    Require(positions[i].id.value == i + 1, "position ids");
    Require(positions[i].valid_rows == 37010, "one frontier for all");
  }
}

void TestRetainedRows() {
  const f::ContinuationLayout layout(MakeConfig(), 262144);
  const auto& window = layout.components()[0];
  const auto& global = layout.components()[4];
  Require(window.descriptor.kind == cache::ComponentKind::kWindowRows &&
              global.descriptor.kind == cache::ComponentKind::kAppendRows,
          "fixture");
  // 10000 - 512 = 9488, whose chunk starts at 9472.
  Require(layout.FirstRetainedRow(window, 10000) == 9472, "window chunk start");
  Require(layout.FirstRetainedRow(global, 10000) == 0, "global keeps row 0");
  // Short of the window nothing is dropped.
  Require(layout.FirstRetainedRow(window, 300) == 0, "short prefix");
  Require(layout.FirstRetainedRow(window, 512) == 0, "exactly the window");
  Require(layout.FirstRetainedRow(window, 513) == 0, "chunk of row 1");
  Require(layout.FirstRetainedRow(window, 768) == 256, "second chunk");
  // The ring (window + prefill tile) must hold the most a restore retains.
  constexpr std::uint32_t kPrefillTile = 4096;
  for (cache::Rows p : {1ULL, 511ULL, 512ULL, 4096ULL, 37010ULL, 262144ULL}) {
    const auto retained = p - layout.FirstRetainedRow(window, p);
    Require(retained <= 512 + f::ContinuationLayout::kRowsPerChunk - 1,
            "a restore retains at most window + chunk - 1 rows");
    Require(retained <= 512 + kPrefillTile, "and the ring holds that");
  }
}

void TestInvalid() {
  bool threw = false;
  try {
    f::Config c = MakeConfig();
    c.vocab_size = 0;
    const f::ContinuationLayout layout(c, 1024);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Require(threw, "a config without a vocabulary is refused");
  const f::ContinuationLayout layout(MakeConfig(), 1024);
  threw = false;
  try {
    (void)layout.at({0});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Require(threw, "id 0 is invalid");
  threw = false;
  try {
    (void)layout.at({999});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Require(threw, "unknown id is invalid");
}

}  // namespace

int main() {
  try {
    TestGeometry();
    TestPositions();
    TestRetainedRows();
    TestInvalid();
  } catch (const std::exception& failure) {
    std::cerr << "FAIL: " << failure.what() << '\n';
    return 1;
  }
  std::cout << "continuation_layout_test OK\n";
  return 0;
}
