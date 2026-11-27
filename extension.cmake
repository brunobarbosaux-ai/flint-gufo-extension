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
    src/tool_dialect.cpp
    src/tool_dialect_register.cpp
    src/reference.cpp
    src/gpu/engine_options.cpp
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
  # Matrix-core decode attention for the global layers (docs/ATTENTION-WMMA.md):
  # off by default, it is not bit-exact to the scalar kernels. A build-time
  # choice, so one binary has one numerics mode. `gufo extension install` passes
  # no CMake variable, hence the environment variable of the same name.
  # option() applies its default only on a fresh configure, so an environment
  # value is also checked against the cache: a build directory configured once
  # with the variable set must not silently keep that mode after it is unset.
  # The environment sets the initial value on a fresh configure and overrides a
  # cache that disagrees with it; with no environment value the cache stands.
  set(_flint_matrix_preset OFF)
  if(DEFINED CACHE{FLINT_MATRIX_ATTENTION})
    set(_flint_matrix_preset ON)
  endif()
  option(FLINT_MATRIX_ATTENTION
    "Flint: matrix-core decode attention for the global layers" OFF)
  if(DEFINED ENV{FLINT_MATRIX_ATTENTION})
    if("$ENV{FLINT_MATRIX_ATTENTION}" MATCHES "^(1|ON|on|On|TRUE|true|YES|yes)$")
      set(_flint_matrix_env ON)
    else()
      set(_flint_matrix_env OFF)
    endif()
    if(NOT _flint_matrix_preset)
      set(FLINT_MATRIX_ATTENTION ${_flint_matrix_env} CACHE BOOL
        "Flint: matrix-core decode attention for the global layers" FORCE)
    else()
      if(FLINT_MATRIX_ATTENTION)
        set(_flint_matrix_cached ON)
      else()
        set(_flint_matrix_cached OFF)
      endif()
      if(NOT _flint_matrix_env STREQUAL _flint_matrix_cached)
        message(WARNING
          "Flint: FLINT_MATRIX_ATTENTION=$ENV{FLINT_MATRIX_ATTENTION} in the "
          "environment overrides the cached value (${_flint_matrix_cached}); "
          "the matrix-core mode changes numerics and the KV payload layout "
          "(flint-kv-rows-window-v1 vs -matrix). Use one build directory per mode.")
        set(FLINT_MATRIX_ATTENTION ${_flint_matrix_env} CACHE BOOL
          "Flint: matrix-core decode attention for the global layers" FORCE)
      endif()
    endif()
  endif()
  if(FLINT_MATRIX_ATTENTION)
    message(STATUS "Flint: matrix-core decode attention ON")
    target_compile_definitions(gufo_flint_hip PRIVATE FLINT_MATRIX_ATTENTION=1)
  endif()
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
# Speculation gates (M3): drives the runner's text contract with the gate's own speculative options
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
# Decode attention bench: scalar against matrix-core kernels on synthetic KV.
# GPU, no artifact.
# ---------------------------------------------------------------------------
gufo_extension_target(
  NAME flint_attn_bench
  SOURCES
    tools/attn_bench.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_runtime
    hip::host
)

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
  NAME flint_engine_options_test
  SOURCES
    tests/engine_options_test.cpp
  LINK_LIBRARIES
    gufo_flint
    gufo_ext_runtime
)

if(TARGET flint_engine_options_test)
  add_test(NAME flint.engine_options COMMAND flint_engine_options_test)
  set_tests_properties(flint.engine_options PROPERTIES LABELS "flint;cpu")
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
  NAME flint_tool_template_test
  SOURCES
    tests/tool_template_test.cpp
  LINK_LIBRARIES
    gufo_flint
    gufo_ext_runtime
)

if(TARGET flint_tool_template_test)
  add_test(NAME flint.tool_template COMMAND flint_tool_template_test
    "${GUFO_EXTENSION_BUILD_ROOT}/tests/golden/template_cases.json")
  set_tests_properties(flint.tool_template PROPERTIES LABELS "flint;cpu")
endif()

gufo_extension_target(
  NAME flint_tool_scan_test
  SOURCES
    tests/tool_scan_test.cpp
  LINK_LIBRARIES
    gufo_flint
    gufo_ext_runtime
)

if(TARGET flint_tool_scan_test)
  add_test(NAME flint.tool_scan COMMAND flint_tool_scan_test)
  set_tests_properties(flint.tool_scan PROPERTIES LABELS "flint;cpu")
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

# Session row access and snapshot restore against the real target: restore
# through snapshot and through split row copies must match a fresh session, and
# the row-write hook must report each append once. Skips (exit 0) when the
# artifact or a HIP device is absent.
gufo_extension_target(
  NAME flint_engine_snapshot_test
  SOURCES
    tests/engine_snapshot_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_engine_snapshot_test)
  target_include_directories(flint_engine_snapshot_test PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  add_test(NAME flint.engine_snapshot COMMAND flint_engine_snapshot_test)
  set_tests_properties(flint.engine_snapshot
    PROPERTIES LABELS "flint;hip;gpu;gfx1151")
endif()

# Row independence of the routed MMQ expert projections (random weights).
gufo_extension_target(
  NAME flint_moe_split_test
  SOURCES
    tests/moe_split_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_moe_split_test)
  add_test(NAME flint.moe_split COMMAND flint_moe_split_test)
  set_tests_properties(flint.moe_split PROPERTIES LABELS "flint;hip;gpu;gfx1151")
endif()

# Prefill split invariance: one Session::Prefill call vs the same prompt split
# at even rows, comparing logits and every layer's K/V rows. Skips without
# FLINT_TARGET or a HIP device.
gufo_extension_target(
  NAME flint_prefill_split_test
  SOURCES
    tests/prefill_split_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_prefill_split_test)
  target_include_directories(flint_prefill_split_test PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  add_test(NAME flint.prefill_split COMMAND flint_prefill_split_test)
  set_tests_properties(flint.prefill_split
    PROPERTIES LABELS "flint;hip;gpu;gfx1151")
  # prefill_tile 4136 leaves 23 rows of ring slack from rounding alone, fewer
  # than the 30 masked key rows below the window that an aligned key tile can
  # read. Guard for the RingLength slack; it passes with and without the extra
  # rows on gfx1151 (the <= 1 ulp effect did not reproduce), so it does not
  # discriminate. Odd tiles (4157, 4159, 4161, 4223) differ for another reason,
  # with and without the extra rows.
  add_test(NAME flint.prefill_split_tile4136 COMMAND flint_prefill_split_test)
  set_tests_properties(flint.prefill_split_tile4136
    PROPERTIES LABELS "flint;hip;gpu;gfx1151"
               ENVIRONMENT "FLINT_SPLIT_PREFILL_TILE=4136")
endif()

# Snapshot restore across a wrapped sliding-window ring (9000-token prefix, the
# ring wraps twice): same-session and fresh-session restores must continue
# bit-identically. Skips (exit 0) when the artifact or a HIP device is absent.
gufo_extension_target(
  NAME flint_ring_snapshot_test
  SOURCES
    tests/ring_snapshot_test.cpp
  LINK_LIBRARIES
    gufo_flint_hip
    gufo_flint
    gufo_ext_core
    gufo_ext_runtime
    hip::host
)

if(TARGET flint_ring_snapshot_test)
  target_include_directories(flint_ring_snapshot_test PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  add_test(NAME flint.ring_snapshot COMMAND flint_ring_snapshot_test)
  set_tests_properties(flint.ring_snapshot
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
