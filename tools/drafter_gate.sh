#!/usr/bin/env bash
# One-command DFlash drafter gate (M1c Phase F).
#
# Tiers:
#   parity  Level 1 (BF16 vs upstream, 20 positions) and Level 2 (Q6_K vs BF16
#           greedy proposals): flint_drafter_parity sweeps + drafter_sweep.py
#   accept  Level 3 (acceptance vs the llama.cpp oracle at p0): oracle at p0 and
#           p0.5, ours free-running and forced-stream for BF16 and Q6_K, compare
#   full    parity, then accept
#
# One GPU model at a time (plan rule 3): the oracle server is stopped before any
# flint_drafter_parity run. drafter_accept.py oracle starts and stops its own
# server, so it runs strictly before the extension tools, never concurrently.
#
# Writes $RESULTS_DIR/summary.md and summary.json. Exit code is non-zero if any
# level that ran failed.
#
# Environment overrides:
#   BUILD_DIR       dir with flint_drafter_parity
#                   (default /home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build)
#   TARGET          target gguf, Q4_K_XL          (default /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf)
#   DRAFT_BF16      BF16 drafter gguf             (default /home/bruno/AI/GGUF/mtp/Flint-DFlash2-BF16.gguf)
#   DRAFT_Q6K       Q6_K drafter gguf             (default /home/bruno/AI/GGUF/mtp/Flint-DFlash2-Q6_K.gguf)
#   ORACLE_BIN_DIR  llama.cpp b11804 dir          (default /home/bruno/AI/llama/llama-b11804-rocm)
#   UPSTREAM_DIR    z-lab/dflash checkout         (default /tmp/opencode/dflash-upstream)
#   HARNESS         reference harness             (default /home/bruno/AI/Gufo-mtp-perf/tools/qwen27b/dflash_reference.py)
#   RESULTS_DIR     where logs/JSON land          (default /tmp/opencode/drafter-gate)
#   SKIP_ORACLE     1 = reuse cached oracle JSONL in RESULTS_DIR (default 0)
#   PYTHON          python interpreter            (default python3)

set -uo pipefail

TIER="${1:-full}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-/home/bruno/AI/Gufo/build/host-test/extensions/build/flint/build}"
TARGET="${TARGET:-/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf}"
DRAFT_BF16="${DRAFT_BF16:-/home/bruno/AI/GGUF/mtp/Flint-DFlash2-BF16.gguf}"
DRAFT_Q6K="${DRAFT_Q6K:-/home/bruno/AI/GGUF/mtp/Flint-DFlash2-Q6_K.gguf}"
ORACLE_BIN_DIR="${ORACLE_BIN_DIR:-/home/bruno/AI/llama/llama-b11804-rocm}"
UPSTREAM_DIR="${UPSTREAM_DIR:-/tmp/opencode/dflash-upstream}"
HARNESS="${HARNESS:-/home/bruno/AI/Gufo-mtp-perf/tools/qwen27b/dflash_reference.py}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/opencode/drafter-gate}"
SKIP_ORACLE="${SKIP_ORACLE:-0}"
PY="${PYTHON:-python3}"

ORACLE_PORT=8098
UPSTREAM_REF=07ebd93db9f472af339b644bb70221ad8428328a
UPSTREAM_SHA256=f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e
PROMPTS="$REPO/tools/fixtures/accept/prompts.jsonl"
POSITIONS="64,128,256,512,1000,1024,1500,2000,2046,2047,2048,2049,2050,2100,2560,3000,3500,4000,4096,4500"
PARITY_BIN="$BUILD_DIR/flint_drafter_parity"

mkdir -p "$RESULTS_DIR"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

kill_port() {
  local port="$1"
  fuser -k "$port/tcp" >/dev/null 2>&1 || true
  sleep 2
}

# Rule 3: no llama-server may hold the GPU while an extension tool runs.
free_gpu() {
  kill_port "$ORACLE_PORT"
  if pgrep -f '[l]lama-server' >/dev/null; then
    log "llama-server still running; killing it"
    pkill -f '[l]lama-server' || true
    sleep 2
    if pgrep -f '[l]lama-server' >/dev/null; then
      log "ERROR: llama-server survived pkill"
      return 1
    fi
  fi
}

# Each level writes level-N.json; summary.{md,json} are built from those files.
write_level() {  # level passed(true|false) detail
  "$PY" - "$RESULTS_DIR/level-$1.json" "$2" "$3" <<'PY'
import json, sys
path, passed, detail = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
with open(path, "w") as handle:
    json.dump({"passed": passed, "detail": detail}, handle)
PY
}

write_summary() {
  "$PY" - "$RESULTS_DIR" "$TIER" <<'PY'
import json, os, sys
results, tier = sys.argv[1], sys.argv[2]
names = {"1": "Level 1 (BF16 vs upstream)",
         "2": "Level 2 (Q6_K vs BF16 greedy)",
         "3": "Level 3 (acceptance vs oracle)"}
levels = {}
for n in ("1", "2", "3"):
    path = os.path.join(results, f"level-{n}.json")
    if os.path.exists(path):
        with open(path) as handle:
            levels[n] = json.load(handle)
overall = bool(levels) and all(v["passed"] for v in levels.values())
with open(os.path.join(results, "summary.json"), "w") as handle:
    json.dump({"levels": levels, "passed": overall}, handle, indent=2)
lines = [f"# Drafter gate, tier `{tier}`", "", f"Results: `{results}`", ""]
for n in ("1", "2", "3"):
    if n in levels:
        verdict = "PASS" if levels[n]["passed"] else "FAIL"
        lines.append(f"- {names[n]}: {verdict}, {levels[n]['detail']}")
lines += ["", f"Overall: {'PASS' if overall else 'FAIL'}"]
text = "\n".join(lines) + "\n"
with open(os.path.join(results, "summary.md"), "w") as handle:
    handle.write(text)
sys.stdout.write(text)
sys.exit(0 if overall else 1)
PY
}

# Records the failed level, prints the summary, and exits non-zero.
abort_tier() {  # level detail
  log "ERROR: $2"
  write_level "$1" false "$2"
  write_summary
  exit 1
}

ensure_upstream() {
  if [[ ! -f "$UPSTREAM_DIR/dflash/model.py" ]]; then
    log "cloning z-lab/dflash into $UPSTREAM_DIR"
    git clone https://github.com/z-lab/dflash "$UPSTREAM_DIR" || return 1
    git -C "$UPSTREAM_DIR" checkout "$UPSTREAM_REF" || return 1
  fi
  local sha
  sha="$(sha256sum "$UPSTREAM_DIR/dflash/model.py" | cut -d' ' -f1)"
  if [[ "$sha" != "$UPSTREAM_SHA256" ]]; then
    log "upstream model.py sha256 $sha != expected $UPSTREAM_SHA256"
    return 1
  fi
  log "upstream verified at $UPSTREAM_REF"
}

# greedy-native.json is a copy of greedy.json: the head route does not change the
# selector's greedy proposals (byte-identical for both quants), and Level 2 reads
# that name.
run_sweep() {  # label draft
  local label="$1" draft="$2" dir="$RESULTS_DIR/sweep-$1"
  free_gpu || return 1
  log "sweep $label -> $dir"
  "$PARITY_BIN" "$TARGET" "$draft" --positions "$POSITIONS" --out "$dir" \
      --temperature 0.8 --head reference > "$RESULTS_DIR/sweep-$label.log" 2>&1 || return 1
  cp "$dir/greedy.json" "$dir/greedy-native.json"
}

tier_parity() {
  ensure_upstream || abort_tier 1 "upstream verification failed (see log above)"
  run_sweep bf16 "$DRAFT_BF16" || abort_tier 1 "sweep bf16 failed (see sweep-bf16.log)"
  run_sweep q6k "$DRAFT_Q6K" || abort_tier 1 "sweep q6k failed (see sweep-q6k.log)"

  free_gpu || abort_tier 1 "could not free the GPU"
  "$PY" "$HERE/drafter_sweep.py" \
      --sweep-bf16 "$RESULTS_DIR/sweep-bf16" --sweep-q6k "$RESULTS_DIR/sweep-q6k" \
      --draft-bf16 "$DRAFT_BF16" --draft-q6k "$DRAFT_Q6K" --target "$TARGET" \
      --upstream "$UPSTREAM_DIR/dflash/model.py" --harness "$HARNESS" \
      --harness-cwd "$(dirname "$HARNESS")" --out "$RESULTS_DIR/sweep-summary" \
      > "$RESULTS_DIR/sweep-summary.log" 2>&1 \
      || abort_tier 1 "drafter_sweep.py failed (see sweep-summary.log)"
  cat "$RESULTS_DIR/sweep-summary.log"

  local lines
  lines="$("$PY" - "$RESULTS_DIR/sweep-summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    data = json.load(handle)
l1, l2 = data["level1"], data["level2"]
total = len(l1["positions"])
print("1", str(l1["passed"]).lower(),
      f"{l1['positions_passed']}/{total} positions pass every stage gate")
print("2", str(l2["passed"]).lower(),
      f"{l2['identical']}/{l2['total']} ({l2['fraction'] * 100:.1f}%) greedy proposals identical")
PY
)" || abort_tier 1 "could not read sweep-summary.json"
  local n passed detail
  while read -r n passed detail; do
    write_level "$n" "$passed" "$detail"
  done <<< "$lines"
}

# C5 verdict A, per quant (fixed thresholds, plan §C5):
#   pooled mean len >= oracle - 0.25; code/json/prose mean len >= oracle - 0.40;
#   pooled acceptance within 0.03 (fractions).
c5_verdict() {
  "$PY" - "$RESULTS_DIR" <<'PY'
import json, sys
results = sys.argv[1]
passed, lens, notes, oracle_len = True, [], [], None
for q, label in (("bf16", "BF16"), ("q6k", "Q6_K")):
    with open(f"{results}/compare-{q}.json") as handle:
        rows = {row["category"]: row for row in json.load(handle)["rows"]}
    missing = [c for c in ("all", "code", "json", "prose") if c not in rows]
    if missing:
        print(f"false missing category rows for {label}: {missing}")
        sys.exit(0)
    pooled = rows["all"]
    oracle_len = pooled["oracle_p0"]["mean_len"]
    ours_len = pooled["ours_stream"]["mean_len"]
    acc_gap = abs(pooled["ours_stream"]["acceptance"] - pooled["oracle_p0"]["acceptance"])
    weak = [c for c in ("code", "json", "prose")
            if rows[c]["ours_stream"]["mean_len"] < rows[c]["oracle_p0"]["mean_len"] - 0.40]
    passed = passed and ours_len >= oracle_len - 0.25 and not weak and acc_gap <= 0.03
    lens.append(f"{label} {ours_len:.3f}")
    if weak:
        notes.append(f"{label} below -0.40 on {'/'.join(weak)}")
    if acc_gap > 0.03:
        notes.append(f"{label} acceptance gap {acc_gap * 100:.1f} pts")
detail = f"pooled mean len {' / '.join(lens)} vs oracle {oracle_len:.3f}"
if notes:
    detail += " (" + "; ".join(notes) + ")"
print(f"{str(passed).lower()} {detail}")
PY
}

tier_accept() {
  free_gpu || abort_tier 3 "could not free the GPU"
  "$PARITY_BIN" "$TARGET" "$DRAFT_BF16" --prompts "$PROMPTS" \
      --dump-prompts "$RESULTS_DIR/prompts-dump.jsonl" > "$RESULTS_DIR/prompts-dump.log" 2>&1 \
      || abort_tier 3 "prompt dump failed (see prompts-dump.log)"

  if [[ "$SKIP_ORACLE" == 1 ]]; then
    local f
    for f in oracle-p0.jsonl oracle-p05.jsonl; do
      [[ -f "$RESULTS_DIR/$f" ]] || abort_tier 3 "SKIP_ORACLE=1 but $RESULTS_DIR/$f is missing"
    done
    log "SKIP_ORACLE=1: reusing cached oracle JSONL"
  else
    free_gpu || abort_tier 3 "could not free the GPU"
    "$PY" "$HERE/drafter_accept.py" oracle --dump "$RESULTS_DIR/prompts-dump.jsonl" \
        --results-dir "$RESULTS_DIR" --server-bin "$ORACLE_BIN_DIR/llama-server" \
        --target "$TARGET" --draft "$DRAFT_BF16" \
        || abort_tier 3 "oracle run failed (see output above)"
  fi

  local q draft
  for q in bf16 q6k; do
    if [[ "$q" == bf16 ]]; then draft="$DRAFT_BF16"; else draft="$DRAFT_Q6K"; fi
    free_gpu || abort_tier 3 "could not free the GPU"
    "$PARITY_BIN" "$TARGET" "$draft" --temperature 0 \
        --prompts-ids "$RESULTS_DIR/prompts-dump.jsonl" \
        --report "$RESULTS_DIR/ours-free-$q.jsonl" > "$RESULTS_DIR/ours-free-$q.log" 2>&1 \
        || abort_tier 3 "free-running run $q failed (see ours-free-$q.log)"
    free_gpu || abort_tier 3 "could not free the GPU"
    "$PARITY_BIN" "$TARGET" "$draft" --temperature 0 \
        --prompts-ids "$RESULTS_DIR/prompts-dump.jsonl" \
        --report "$RESULTS_DIR/ours-stream-$q.jsonl" \
        --stream "$RESULTS_DIR/oracle-p0.jsonl" > "$RESULTS_DIR/ours-stream-$q.log" 2>&1 \
        || abort_tier 3 "forced-stream run $q failed (see ours-stream-$q.log)"
    "$PY" "$HERE/drafter_accept.py" compare --oracle "$RESULTS_DIR/oracle-p0.jsonl" \
        --ours-free "$RESULTS_DIR/ours-free-$q.jsonl" \
        --ours-stream "$RESULTS_DIR/ours-stream-$q.jsonl" \
        --out "$RESULTS_DIR/compare-$q" > "$RESULTS_DIR/compare-$q.log" 2>&1 \
        || abort_tier 3 "compare $q failed (see compare-$q.log)"
  done

  local verdict passed detail
  verdict="$(c5_verdict)"
  passed="${verdict%% *}"
  detail="${verdict#* }"
  [[ "$passed" == true || "$passed" == false ]] || abort_tier 3 "could not compute the C5 verdict"
  write_level 3 "$passed" "$detail"
}

case "$TIER" in
  parity|accept|full) ;;
  *)
    echo "usage: $0 [parity|accept|full]" >&2
    exit 2
    ;;
esac

# The summary covers only the levels this run produced.
rm -f "$RESULTS_DIR"/level-*.json
case "$TIER" in
  parity) tier_parity ;;
  accept) tier_accept ;;
  full)   tier_parity; tier_accept ;;
esac

write_summary
