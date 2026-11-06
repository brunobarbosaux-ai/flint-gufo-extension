#ifndef GUFO_MODELS_FLINT_RUNNER_HPP_
#define GUFO_MODELS_FLINT_RUNNER_HPP_

#include <gufo/ext/text.hpp>
#include <memory>
#include <optional>
#include <string>

namespace gufo::models::flint {

// Loads the Flint artifact named by options.model_path and returns a text
// runner. Returns nullptr and sets *error on failure; never throws across the
// factory boundary.
std::shared_ptr<ext::TextModelRunner> CreateFlintTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

// Sampling defaults the prompt and chat drivers recommend for Flint, used by
// both TextProvider::sampling_defaults and TextRunnerDescriptor. Always set:
// Flint has no published preset to defer to.
std::optional<sampling::TextSamplingDefaults> FlintTextSamplingDefaults(
    const core::GgufReader&);

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_RUNNER_HPP_
