#ifndef GUFO_MODELS_FLINT_PROBE_HPP_
#define GUFO_MODELS_FLINT_PROBE_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::flint {

// A cheap, deterministic artifact probe. Reports whether the GGUF names the
// Flint architecture. Geometry validation lives in ParseConfig so the provider
// can surface a precise error instead of a silent non-match.
[[nodiscard]] bool ProbeFlintGguf(const core::GgufReader& reader);

// Filesystem form: true when the single *.gguf under `model_root` (or the file
// itself) is a Flint artifact. Never opens a GPU context.
[[nodiscard]] bool ProbeFlint(std::string_view model_root);

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_PROBE_HPP_
