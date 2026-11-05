#!/usr/bin/env bash
# Flint parity + quality gates against strycast's serve path.
#
# Tiers (see tools/flint/README.md):
#   quick  G8 + depth curve at 2048 and 32768, 1 run
#   stage  G8 + depth curve at 512/2048/8192/32768, 2 runs
#   full   G8, G4 vs the recorded baseline, GX capture, G5, the six-depth
#          Q4 curve, the same curve on the fork binary, and the Q6 curve
#
# The runner starts/stops the model servers itself. One model on the GPU at a
# time: the Q4 strycast server, the fork server, and the Q6 strycast server
# are started sequentially, never together.
#
# Environment overrides:
#   GUFO_BIN        strycast binary           (default build/release/gufo)
#   FORK_BIN        fork binary               (default /home/bruno/AI/gufo-mikealanni-ref/build/release/gufo)
#   MODEL_Q4        Q4 gguf                   (default /home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf)
#   MODEL_Q6        Q6 gguf                   (default /home/bruno/AI/GGUF/flint-UD-Q6_K_XL.gguf)
#   ORACLE_BIN_DIR  llama.cpp b11804 dir      (default /home/bruno/AI/llama/llama-b11804-rocm)
#   RESULTS_DIR     where logs/JSON land      (default /tmp/opencode/flint-gates)
#   PORT            strycast/fork serve port  (default 8016)
#   ORACLE_PORT     llama.cpp oracle port     (default 8098)

set -uo pipefail

TIER="${1:-quick}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

GUFO_BIN="${GUFO_BIN:-$REPO/build/release/gufo}"
FORK_BIN="${FORK_BIN:-/home/bruno/AI/gufo-mikealanni-ref/build/release/gufo}"
MODEL_Q4="${MODEL_Q4:-/home/bruno/AI/GGUF/flint-UD-Q4_K_XL.gguf}"
MODEL_Q6="${MODEL_Q6:-/home/bruno/AI/GGUF/flint-UD-Q6_K_XL.gguf}"
PORT="${PORT:-8016}"
ORACLE_PORT="${ORACLE_PORT:-8098}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/opencode/flint-gates}"
PY="${PYTHON:-python3}"

mkdir -p "$RESULTS_DIR"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

kill_port() {
  local port="$1"
  fuser -k "$port/tcp" >/dev/null 2>&1 || true
  sleep 2
}

wait_health() {
  # Poll with a real request: the fork binds its port before the model loads.
  local base="$1" model="$2" tries="${3:-120}"
  for _ in $(seq 1 "$tries"); do
    if curl -s -m 10 -X POST "$base/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        -o /dev/null -w '%{http_code}' | grep -q 200; then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_serve() {
  local bin="$1" gguf="$2" ctx="$3" name="$4" logf="$5"
  kill_port "$PORT"
  log "starting $name ($bin) ctx=$ctx -> $logf"
  "$bin" serve llm -m "$gguf" -p "$PORT" -c "$ctx" \
    --served-model-name "$name" > "$logf" 2>&1 &
  if ! wait_health "http://127.0.0.1:$PORT" "$name"; then
    log "ERROR: $name did not become healthy (see $logf)"
    return 1
  fi
  log "$name healthy"
}

start_oracle() {
  local mode="${1:-cpu}" logf="$2"
  kill_port "$ORACLE_PORT"
  log "starting oracle ($mode) -> $logf"
  ( cd "$HERE" && VERIFY_SHA=0 MODEL="$MODEL_Q4" PORT="$ORACLE_PORT" \
      ./oracle_server.sh "$mode" > "$logf" 2>&1 & )
  for _ in $(seq 1 120); do
    curl -s -m 5 "http://127.0.0.1:$ORACLE_PORT/health" | grep -q ok && { log "oracle healthy"; return 0; }
    sleep 5
  done
  log "ERROR: oracle did not become healthy"
  return 1
}

stop_all() {
  kill_port "$PORT"
  kill_port "$ORACLE_PORT"
}

depth_curve() {
  local label="$1" depths="$2" runs="$3" max_tok="$4" out="$5"
  log "depth curve [$label] depths=$depths runs=$runs -> $out"
  "$PY" "$HERE/flint_depth_curve.py" \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$label" --depths "$depths" --runs "$runs" \
    --max-tokens "$max_tok" --no-cache-prompt --label "$label" \
    --out "$out" 2>&1 | tee "${out%.jsonl}.log"
}

run_common() {
  local tier="$1"
  local sessions=8
  # G8: concurrency determinism.
  log "G8 (1..$sessions sessions)"
  "$PY" "$HERE/serve_gates.py" g8 --engine serve \
    --base-url "http://127.0.0.1:$PORT/v1" --model flint-q4 \
    --sessions "$sessions" --gate 2>&1 | tee "$RESULTS_DIR/g8-$tier.log"

  if [[ "$tier" == quick ]]; then
    depth_curve flint-q4 "2048,32768" 1 256 "$RESULTS_DIR/depth-q4-quick.jsonl"
  elif [[ "$tier" == stage ]]; then
    depth_curve flint-q4 "512,2048,8192,32768" 2 256 "$RESULTS_DIR/depth-q4-stage.jsonl"
  fi
}

run_full() {
  # G8 + full six-depth Q4 curve on strycast.
  log "=== FULL: strycast Q4 ==="
  start_serve "$GUFO_BIN" "$MODEL_Q4" 131072 flint-q4 "$RESULTS_DIR/serve-q4.log" || return 1
  "$PY" "$HERE/serve_gates.py" g8 --engine serve \
    --base-url "http://127.0.0.1:$PORT/v1" --model flint-q4 --sessions 8 --gate \
    2>&1 | tee "$RESULTS_DIR/g8-full.log"
  depth_curve flint-q4 "512,2048,8192,32768,65280,130816" 3 256 "$RESULTS_DIR/depth-q4.jsonl"

  # G4/G5 need the oracle for tokenization; oracle also serves the G5 arm.
  log "=== FULL: oracle (cpu, tokenizer) ==="
  stop_all
  start_oracle cpu "$RESULTS_DIR/oracle-cpu.log" || return 1
  log "G4 serve arm"
  "$PY" "$HERE/serve_gates.py" g4 --engine serve \
    --base-url "http://127.0.0.1:$PORT/v1" --model flint-q4 \
    --oracle-url "http://127.0.0.1:$ORACLE_PORT" \
    --prompts 8 --prompt-len 512 --out "$RESULTS_DIR/g4-serve.json" \
    2>&1 | tee "$RESULTS_DIR/g4-serve.log"
  log "G4 compare vs recorded baseline"
  "$PY" "$HERE/serve_gates.py" g4-compare \
    /home/bruno/AI/bench/flint-p8/g4-8.0-q4.json "$RESULTS_DIR/g4-serve.json" \
    2>&1 | tee "$RESULTS_DIR/g4-compare.log"
  log "GX capture (needs serve, restart it)"
  stop_all
  start_serve "$GUFO_BIN" "$MODEL_Q4" 131072 flint-q4 "$RESULTS_DIR/serve-q4-gx.log" || return 1
  "$PY" "$HERE/gx_capture.py" \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" --model flint-q4 \
    --out "$RESULTS_DIR/gx-q4.json" 2>&1 | tee "$RESULTS_DIR/gx.log"
  stop_all
  log "full tier done"
}

case "$TIER" in
  quick|stage)
    start_serve "$GUFO_BIN" "$MODEL_Q4" 65536 flint-q4 "$RESULTS_DIR/serve-q4-$TIER.log" || { stop_all; exit 1; }
    run_common "$TIER"
    stop_all
    ;;
  full)
    run_full
    ;;
  *)
    echo "usage: $0 [quick|stage|full]" >&2
    exit 2
    ;;
esac

log "all done; results in $RESULTS_DIR"
