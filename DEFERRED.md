# Deferred items

### Base-session VRAM undercount in `Model::SessionBytes` (pre-existing)
- `Session::Init` (`src/gpu/engine.cpp`) allocates per-session scratch pools, step-graph workspaces and allocator slack totalling ~600–666 MiB more than `SessionBytes` accounts for. Measured with `flint_spec_gate --mode memory` (second-session delta): spec-off claim 1457 MiB vs measured 2123 MiB; spec-on claim 1873 MiB vs measured 2443 MiB. Identical in both modes, so it predates DFlash2 and is not an M5 regression.
- The M5 memory gate criterion (hidden ring + drafter accounted in `ResourceClaim`) is met: the spec-on per-session claim grows 436 MiB over spec-off and the measured growth is exactly the 320 MiB hidden ring.
- Action before release: audit `Session::Init` allocations against `SessionBytes` and add the missing terms. Evidence: `/tmp/opencode/m5-smoke/memory2.log`.
