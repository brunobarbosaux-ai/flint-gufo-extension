#include <gufo/ext/generated/flint_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "drivers.hpp"
#include "probe.hpp"
#include "runner.hpp"

// Provider entry point, run by the host's generated registry only when the
// provider target is enabled. Metadata comes from extension.json and lives in
// the generated header, so registration adds the generated ProviderInfo first
// and then the typed text callbacks. The probe-only provider for CPU builds is
// registered by flint_register from extension.cpp.
void flint_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_flint::kProvider_flint);
  registry.AddTextProvider({
      .id = "flint",
      .probe = &gufo::models::flint::ProbeFlint,
      .probe_gguf = &gufo::models::flint::ProbeFlintGguf,
      .create = &gufo::models::flint::CreateFlintTextRunner,
      .bench_driver = &gufo::models::flint::RunFlintBench,
      .prompt_driver = &gufo::models::flint::RunFlintPrompt,
      .chat_driver = &gufo::models::flint::RunFlintChat,
      .sampling_defaults = &gufo::models::flint::FlintTextSamplingDefaults,
  });
}
