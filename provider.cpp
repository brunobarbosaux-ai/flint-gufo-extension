#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "drivers.hpp"
#include "probe.hpp"
#include "runner.hpp"

// Provider metadata (ProviderInfo) is generated from extension.json and
// registered by the generated registry, so this entry point only adds the text
// provider. The CPU build registers nothing: flint_register already reports the
// artifact through the probe-only provider.
void flint_provider_register(gufo::ext::Registry& registry) {
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "flint",
      .probe = &gufo::models::flint::ProbeFlint,
      .probe_gguf = &gufo::models::flint::ProbeFlintGguf,
      .create = &gufo::models::flint::CreateFlintTextRunner,
      .bench_driver = &gufo::models::flint::RunFlintBench,
      .prompt_driver = &gufo::models::flint::RunFlintPrompt,
      .chat_driver = &gufo::models::flint::RunFlintChat,
  });
#else
  (void)registry;
#endif
}
