#include "src/gpu/engine.hpp"

#include <string>

namespace gufo::models::flint::gpu {

bool ValidateEngineOptions(const EngineOptions& options,
                           std::string* error_msg) {
  // An odd pass leaves its last row wrong: on main, prefill_tile 4157 first
  // differs at row 4156 and 4159 at row 4158 (the last row of each pass), while
  // 4096, 4158 and 4162 are exact. The mechanism is not pinned, so reject the
  // value instead of rounding it silently.
  if (options.prefill_tile >= 2 && options.prefill_tile % 2 == 0)
    return true;
  if (error_msg != nullptr)
    *error_msg = "Flint: prefill_tile must be even and at least 2 (got " +
                 std::to_string(options.prefill_tile) +
                 "): an odd pass leaves its last row wrong";
  return false;
}

}  // namespace gufo::models::flint::gpu
