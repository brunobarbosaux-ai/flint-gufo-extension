# Flint 40B MoE external Gufo extension.
#
# Host-side code (probe, config, chat template, reference) builds without HIP so
# diagnostics work without a GPU context. HIP targets carry the engine, its
# private-toolkit kernels and the vendored GEMM tier.

# ---------------------------------------------------------------------------
# Host tier: probe, config, chat template, CPU reference. No GPU context.
# ---------------------------------------------------------------------------
gufo_extension_target(
  NAME gufo_flint
  SOURCES
    src/runtime/config.cpp
    src/chat_template.cpp
    src/reference.cpp
    src/gpu/dflash/dflash_weights.cpp
    probe.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_flint)
  target_include_directories(gufo_flint PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_flint PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

# ---------------------------------------------------------------------------
# HIP tier: Flint's own kernels + device model + engine + vendored GEMM tier.
# ---------------------------------------------------------------------------
gufo_extension_target(
  NAME gufo_flint_hip
  SOURCES
    src/gpu/device_model.cpp
    src/gpu/engine.cpp
    src/gpu/engine_stub.cpp
    src/spec/target_adapter.cpp
    src/spec/cycle.cpp
    src/spec/draft_policy.cpp
    src/tokenizer_support.cpp
    hip/vendor/rocm/blaslt.cpp
    hip/vendor/qwen/tokenizer.cpp
  LINK_LIBRARIES
    gufo_flint
    gufo_ext_runtime
    ICU::uc
    ICU::i18n
    hip::host
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
)

if(TARGET gufo_flint_hip)
  # Flint's own kernels mirror the reference fork's flint target: -O3, no
  # fast-math.
  gufo_extension_hip_sources(gufo_flint_hip
    SOURCES
      src/gpu/dflash/dflash_forward.cpp
      src/gpu/kernels/kernels.hip.cpp
      # DFlash-2 draft kernels, vendored verbatim from the in-tree Qwen family.
      # Same tier as Flint's own kernels: -O3, no fast-math, so the arithmetic
      # the parity harness compares is the arithmetic the oracle runs.
      src/gpu/dflash/dflash_kernels.hip.cpp
  )
  # The vendored Qwen GEMM tier mirrors the in-tree qwen38_flash_next tiers:
  # its HIP targets compile with fast-math, and the mmq tier adds
  # -Wno-unused-value. Same source, same options, same arithmetic as the
  # reference builds.
  gufo_extension_hip_sources(gufo_flint_hip
    SOURCES
      hip/vendor/rocm/kernels.hip.cpp
      hip/vendor/rocm/mmq/mmid.hip.cpp
      hip/vendor/rocm/mmq/mmvq.hip.cpp
      hip/vendor/rocm/mmq/moe_wave64.hip.cpp
      hip/vendor/rocm/mmq/flint_ggml_stubs.hip.cpp
      hip/vendor/rocm/mmq/flint_mmq.hip.cpp
      hip/vendor/rocm/mmq/quantize.hip.cpp
    OPTIONS
      -ffast-math
      -fno-finite-math-only
      -Wno-unused-value
  )
  # The wave64 W8A8 launcher instantiates the blocked WMMA GEMM, which needs
  # the wave64 subtarget; the in-tree Qwen unit compiles it with fast-math.
  gufo_extension_hip_sources(gufo_flint_hip
    SOURCES
      hip/vendor/rocm/w8a8_wave64.hip
    OPTIONS
      -mwavefrontsize64
      -ffast-math
      -fno-finite-math-only
  )
  if(DEFINED GUFO_EXTENSION_BUILD_ROOT)
    target_sources(gufo_flint_hip PRIVATE
      "${GUFO_EXTENSION_BUILD_ROOT}/hip/vendor/quant/ggml_dequant.cpp")
  else()
    target_sources(gufo_flint_hip PRIVATE hip/vendor/quant/ggml_dequant.cpp)
  endif()
  target_compile_definitions(gufo_flint_hip PRIVATE ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_flint_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
  # The pinned Nix toolchain keeps each HIP library under its own prefix, so
  # the hipCUB/rocPRIM/rocWMMA include paths must come from the host's
  # discovered variables; a system ROCm install shares one prefix and masks
  # the need for them.
  target_include_directories(gufo_flint_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR}
  )
  target_include_directories(gufo_flint_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# ---------------------------------------------------------------------------
# Text tier: runner, CLI drivers, provider registration.
# ---------------------------------------------------------------------------
gufo_extension_target(
  NAME gufo_flint_text
  SOURCES
    continuation_adapter.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_cache
    gufo_ext_runtime
    hip::host
    Threads::Threads
)

if(TARGET gufo_flint_text)
  target_include_directories(gufo_flint_text PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# ---------------------------------------------------------------------------
# Drafter probe: validates a DFlash2 drafter GGUF against the loader contract
# (metadata, topology, tensor inventory) through the public reader. Host tier,
# no GPU context, so it runs wherever the artifact is.
# ---------------------------------------------------------------------------
gufo_extension_target(
  NAME flint_drafter_probe
  SOURCES
    tools/drafter_probe.cpp
  LINK_LIBRARIES
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
)

if(TARGET flint_drafter_probe)
  target_include_directories(flint_drafter_probe PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# ---------------------------------------------------------------------------
# Drafter parity driver (M1b): dumps the trace directory the upstream reference
# harness compares stage by stage. Needs the target model (for the tapped
# residuals and the borrowed embedding/head), so it links the HIP engine.
# ---------------------------------------------------------------------------
gufo_extension_target(
  NAME flint_drafter_parity
  SOURCES
    tools/drafter_parity.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_drafter_parity)
  target_include_directories(flint_drafter_parity PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# ---------------------------------------------------------------------------
# Speculation gates (M3): drives the runner's text contract with FLINT_SPEC
# off / dflash / null and checks greedy identity, null-drafter identity and
# the cycle's tokens per cycle and speedup.
# ---------------------------------------------------------------------------
gufo_extension_target(
  NAME flint_spec_gate
  SOURCES
    tools/spec_gate.cpp
  LINK_LIBRARIES
    gufo_flint_text
    gufo_flint_hip
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_spec_gate)
  target_include_directories(flint_spec_gate PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
# The draft policy (M4): pure host code, no device or weights.
gufo_extension_target(
  NAME flint_draft_policy_test
  SOURCES
    tests/draft_policy_test.cpp
    src/spec/draft_policy.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

if(TARGET flint_draft_policy_test)
  target_include_directories(flint_draft_policy_test PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  add_test(NAME flint.draft_policy COMMAND flint_draft_policy_test)
  set_tests_properties(flint.draft_policy PROPERTIES LABELS "flint;cpu")
endif()

gufo_extension_target(
  NAME flint_config_test
  SOURCES
    tests/config_test.cpp
  LINK_LIBRARIES
    gufo_flint
    gufo_ext_runtime
)

if(TARGET flint_config_test)
  add_test(NAME flint.config COMMAND flint_config_test)
  set_tests_properties(flint.config PROPERTIES LABELS "flint;cpu")
endif()

gufo_extension_target(
  NAME flint_probe_test
  SOURCES
    tests/probe_test.cpp
  LINK_LIBRARIES
    gufo_flint
    gufo_ext_runtime
)

if(TARGET flint_probe_test)
  add_test(NAME flint.probe COMMAND flint_probe_test)
  set_tests_properties(flint.probe PROPERTIES LABELS "flint;cpu")
endif()

gufo_extension_target(
  NAME flint_ops_test
  SOURCES
    tests/ops_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_ops_test)
  add_test(NAME flint.ops COMMAND flint_ops_test)
  set_tests_properties(flint.ops PROPERTIES LABELS "flint;hip;gpu;gfx1151")
endif()

# The vendored DFlash2 draft kernels: every launcher against a host reference on
# shapes small enough to reason about. No weights, no target model, so this is
# the gate the vendoring itself has to pass.
gufo_extension_target(
  NAME flint_dflash_ops_test
  SOURCES
    tests/dflash_ops_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_runtime
    hip::host
)

# The DFlash2 draft forward: synthetic weights, no artifact, no target model.
# Covers upload and role binding, the injected context ring, block bookkeeping,
# the trace hook the parity harness reads, and the selector interface.
gufo_extension_target(
  NAME flint_dflash_forward_test
  SOURCES
    tests/dflash_forward_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
    hip::host
)

# The target adapter (M2): verify rows bit-identical to single-token appends,
# packed features, and rewind, against the real target model. Skips (exit 0)
# when the artifact or a HIP device is absent.
gufo_extension_target(
  NAME flint_target_adapter_test
  SOURCES
    tests/target_adapter_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_target_adapter_test)
  target_include_directories(flint_target_adapter_test PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  add_test(NAME flint.target_adapter COMMAND flint_target_adapter_test)
  set_tests_properties(flint.target_adapter
    PROPERTIES LABELS "flint;hip;gpu;gfx1151")
endif()

if(TARGET flint_dflash_ops_test)
  # The vendored header, like its in-tree source, is gated on the backend
  # macro its real consumer defines.
  target_compile_definitions(flint_dflash_ops_test PRIVATE ENGINE_ENABLE_HIP=1)
  add_test(NAME flint.dflash_ops COMMAND flint_dflash_ops_test)
  set_tests_properties(flint.dflash_ops PROPERTIES LABELS "flint;hip;gpu;gfx1151")
endif()

if(TARGET flint_dflash_forward_test)
  add_test(NAME flint.dflash_forward COMMAND flint_dflash_forward_test)
  set_tests_properties(flint.dflash_forward
    PROPERTIES LABELS "flint;hip;gpu;gfx1151")
endif()

# Source extensions own installation of their additional data and notices.
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/flint)
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/THIRD_PARTY_NOTICES.md"
  DESTINATION share/licenses/gufo/extensions/flint)
