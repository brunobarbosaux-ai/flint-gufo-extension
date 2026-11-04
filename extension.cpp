#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"

namespace gufo::models::flint {

#if !defined(ENGINE_ENABLE_HIP)
// A CPU build still discovers the artifact and reports it truthfully. Device
// execution is not registered, so the provider is probe-only.
void RegisterFlintCpu(ext::Registry& registry) {
  registry.AddTextProvider({
      .id = "flint",
      .probe = &ProbeFlint,
      .probe_gguf = &ProbeFlintGguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
  });
}
#endif

}  // namespace gufo::models::flint

void flint_register(gufo::ext::Registry& registry) {
#if !defined(ENGINE_ENABLE_HIP)
  gufo::models::flint::RegisterFlintCpu(registry);
#else
  (void)registry;
#endif
}
