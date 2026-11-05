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
  gufo_extension_hip_sources(gufo_flint_hip
    SOURCES
      src/gpu/kernels/kernels.hip.cpp
      hip/vendor/rocm/kernels.hip.cpp
      hip/vendor/rocm/mmq/mmid.hip.cpp
      hip/vendor/rocm/mmq/mmvq.hip.cpp
      hip/vendor/rocm/mmq/moe_wave64.hip.cpp
      hip/vendor/rocm/mmq/flint_ggml_stubs.hip.cpp
      hip/vendor/rocm/mmq/flint_mmq.hip.cpp
      hip/vendor/rocm/mmq/quantize.hip.cpp
  )
  # The wave64 W8A8 launcher instantiates the blocked WMMA GEMM, which needs
  # the wave64 subtarget; SLP vectorization spills the Q-quant tiles (mirrors
  # the in-tree per-source flags for the Qwen wave64 units).
  gufo_extension_hip_sources(gufo_flint_hip
    SOURCES
      hip/vendor/rocm/w8a8_wave64.hip
    OPTIONS
      -mwavefrontsize64
      -fno-slp-vectorize
  )
  if(DEFINED GUFO_EXTENSION_BUILD_ROOT)
    target_sources(gufo_flint_hip PRIVATE
      "${GUFO_EXTENSION_BUILD_ROOT}/hip/vendor/quant/ggml_dequant.cpp")
  else()
    target_sources(gufo_flint_hip PRIVATE hip/vendor/quant/ggml_dequant.cpp)
  endif()
  target_compile_definitions(gufo_flint_hip PRIVATE ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_flint_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
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
  target_compile_options(gufo_flint_text PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  target_include_directories(gufo_flint_text PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
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

# Source extensions own installation of their additional data and notices.
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/flint)
install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/THIRD_PARTY_NOTICES.md"
  DESTINATION share/licenses/gufo/extensions/flint)
