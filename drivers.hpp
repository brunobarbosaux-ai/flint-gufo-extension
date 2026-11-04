#ifndef GUFO_MODELS_FLINT_DRIVERS_HPP_
#define GUFO_MODELS_FLINT_DRIVERS_HPP_

#include <chrono>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>
#include <memory>

namespace gufo::models::flint {

int RunFlintBench(const ext::TextBenchOptions& opt,
                  const std::shared_ptr<const core::GgufReader>& reader,
                  std::chrono::steady_clock::time_point load_start);

int RunFlintPrompt(const ext::TextPromptOptions& opt,
                   const core::GgufReader& reader,
                   std::chrono::steady_clock::time_point load_start);

int RunFlintChat(const ext::TextPromptOptions& opt,
                 const core::GgufReader& reader,
                 std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_DRIVERS_HPP_
