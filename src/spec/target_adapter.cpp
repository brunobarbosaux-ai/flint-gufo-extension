#include "src/spec/target_adapter.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <string>
#include <utility>

namespace gufo::models::flint::spec {

namespace {

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
  return false;
}

}  // namespace

std::unique_ptr<TargetAdapter> TargetAdapter::Create(
    gpu::Session& session, std::span<const std::uint32_t> taps,
    std::uint32_t hidden_width, std::string* error_msg) {
  if (taps.empty() || taps.size() > gpu::Session::kMaxHiddenTaps) {
    Fail(error_msg, "target adapter needs 1 to " +
                        std::to_string(gpu::Session::kMaxHiddenTaps) +
                        " capture taps");
    return nullptr;
  }
  if (hidden_width == 0) {
    Fail(error_msg, "target adapter needs a non-zero hidden width");
    return nullptr;
  }
  if (!session.SetHiddenCapture(taps, error_msg))
    return nullptr;
  return std::unique_ptr<TargetAdapter>(
      new TargetAdapter(session, {taps.begin(), taps.end()}, hidden_width));
}

TargetAdapter::TargetAdapter(gpu::Session& session,
                             std::vector<std::uint32_t> taps,
                             std::uint32_t hidden_width)
    : session_(session), taps_(std::move(taps)), hidden_width_(hidden_width) {}

bool TargetAdapter::AppendPrompt(std::span<const std::int32_t> tokens,
                                 std::string* error_msg) {
  if (tokens.empty())
    return Fail(error_msg, "prompt is empty");
  for (std::size_t offset = 0; offset < tokens.size();
       offset += kPrefillChunk) {
    const std::size_t count =
        std::min<std::size_t>(kPrefillChunk, tokens.size() - offset);
    const std::uint32_t before = session_.Position();
    if (!session_.Append(tokens.subspan(offset, count), error_msg))
      return false;
    if (session_.Position() != before + count)
      return Fail(error_msg, "prompt append stopped short of the request");
  }
  return true;
}

bool TargetAdapter::VerifyRows(std::span<const std::int32_t> tokens,
                               std::string* error_msg) {
  if (tokens.empty() || tokens.size() > gpu::Session::kMaxVerifyRows)
    return Fail(error_msg, "verify takes 1 to " +
                               std::to_string(gpu::Session::kMaxVerifyRows) +
                               " tokens");
  const std::uint32_t before = session_.Position();
  if (!session_.AppendAll(tokens, error_msg))
    return false;
  if (session_.Position() != before + tokens.size())
    return Fail(error_msg, "verify pass stopped short of the request");
  return true;
}

std::span<const float> TargetAdapter::RowLogits(std::uint32_t row) const {
  return session_.Logits(row);
}

std::span<const float> TargetAdapter::Logits() const {
  return session_.Logits();
}

bool TargetAdapter::Features(std::uint32_t first, std::uint32_t count,
                             std::vector<float>* out, std::string* error_msg) {
  if (out == nullptr)
    return Fail(error_msg, "features output is null");
  if (static_cast<std::uint64_t>(first) + count > session_.Position())
    return Fail(error_msg, "features requested past the current position");
  const std::size_t row_floats = taps_.size() * hidden_width_;
  out->resize(std::size_t{count} * row_floats);
  for (std::uint32_t offset = 0; offset < count; ++offset) {
    const std::uint32_t row = first + offset;
    for (std::size_t tap = 0; tap < taps_.size(); ++tap) {
      if (!session_.HasHidden(row))
        return Fail(error_msg, "captured residual for row " +
                                   std::to_string(row) +
                                   " left the ring; copy rows out earlier");
      const std::span<const float> captured =
          session_.CapturedHidden(row, static_cast<std::uint32_t>(tap));
      if (captured.size() != hidden_width_)
        return Fail(error_msg, "captured residual has the wrong width");
      std::memcpy(out->data() + std::size_t{offset} * row_floats +
                      tap * hidden_width_,
                  captured.data(), hidden_width_ * sizeof(float));
    }
  }
  return true;
}

void TargetAdapter::Rewind(std::uint32_t position) noexcept {
  session_.Rewind(position);
}

std::uint32_t TargetAdapter::Position() const noexcept {
  return session_.Position();
}

std::uint32_t TargetAdapter::Capacity() const noexcept {
  return session_.Capacity();
}

std::uint32_t TargetAdapter::HiddenWidth() const noexcept {
  return hidden_width_;
}

std::span<const std::uint32_t> TargetAdapter::Taps() const noexcept {
  return taps_;
}

bool TargetAdapter::CheckRowsBitIdentical(std::span<const std::int32_t> tokens,
                                          std::string* error_msg) {
  if (tokens.empty() || tokens.size() > gpu::Session::kMaxVerifyRows)
    return Fail(error_msg, "row check takes 1 to " +
                               std::to_string(gpu::Session::kMaxVerifyRows) +
                               " tokens");
  const std::uint32_t start = session_.Position();
  if (start + tokens.size() > session_.Capacity())
    return Fail(error_msg, "row check does not fit the session");

  // Reference: one token at a time. Logits(row) aliases session storage, so
  // each row is copied out before the next append.
  std::vector<float> reference;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (!session_.Append(tokens.subspan(i, 1), error_msg))
      return false;
    const std::span<const float> row = session_.Logits();
    reference.insert(reference.end(), row.begin(), row.end());
  }
  const std::size_t vocab = reference.size() / tokens.size();

  session_.Rewind(start);
  if (!VerifyRows(tokens, error_msg))
    return false;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const std::span<const float> row =
        session_.Logits(static_cast<std::uint32_t>(i));
    if (row.size() != vocab)
      return Fail(error_msg, "row " + std::to_string(i) +
                                 " has the wrong vocabulary width");
    // Bit comparison: a tolerance would hide exactly the drift this checks.
    if (std::memcmp(row.data(), reference.data() + i * vocab,
                    vocab * sizeof(float)) != 0) {
      std::size_t first_bad = 0;
      while (first_bad < vocab && std::bit_cast<std::uint32_t>(row[first_bad]) ==
                                      std::bit_cast<std::uint32_t>(
                                          reference[i * vocab + first_bad]))
        ++first_bad;
      return Fail(error_msg,
                  "verify row " + std::to_string(i) +
                      " differs from single-token append at logit " +
                      std::to_string(first_bad));
    }
  }
  return true;
}

}  // namespace gufo::models::flint::spec
