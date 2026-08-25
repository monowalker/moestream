#!/usr/bin/env bash
# =============================================================================
# Spike S20 -- what is a byte of decode I/O actually worth?
#
# Several proposals reduce the bytes decode reads:
#   S17 weight-aware miss skipping   -> -10..12% of read bytes
#   B1  low-precision streaming tier -> -50% (tier b) to -90% (tier a)
# Both were costed by multiplying the byte reduction by the measured I/O share.
# That assumes the relationship is linear, and nobody has checked.
#
# MOESTREAM_DROP_FROM=N is an existing diagnostic: at layers >= N a cache miss
# is pointed at the null expert instead of being fetched. Sweeping N therefore
# removes a known fraction of decode's read volume with no other change, and
# the decode time at each point IS the price curve.
#
# OUTPUT IS DELIBERATELY WRONG for any N below the layer count -- experts are
# skipped without renormalisation. This measures speed only. Quality is S17's
# job and needs a separate perplexity run.
#
# MOESTREAM_NOOP=3 (cache lookup, no reads) gives the zero-I/O bound and
# MOESTREAM=0 the no-streaming bound, reproducing RESULTS.md §10.12's method.
# =============================================================================
set -uo pipefail
# Only one spike may run at a time: they all contend for the single GPU, and a
# second one starting mid-run silently corrupts both sets of numbers.
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || {
    echo "another spike holds /tmp/moestream-spike.lock; refusing to run" >&2; exit 9; }
fi
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT INT TERM
NAME=ms-s20; PORT=18082
NL=${NL:-40}

boot() {
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$MODEL_FILE" -e CTX_SIZE="${CTX_SIZE:-32768}" -e UBATCH="${UBATCH:-1024}" \
    -e N_PARALLEL=1 -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 -e MOESTREAM_CACHE_FRAC="${MOESTREAM_CACHE_FRAC:-0.25}" \
    $1 -p "$PORT":8080 moestream/server:local >/dev/null || return 1
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done; return 1
}
ask() { curl -s -m 900 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' -d "$1"; }

run() {
  if ! boot "$2"; then printf "%-30s  FAILED TO BOOT\n" "$1"; docker logs "$NAME" 2>&1|tail -6; return; fi
  ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":700,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local D; D=$(for _ in 1 2 3; do
      ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":100,"temperature":0,"top_k":1,"cache_prompt":true}' \
      | python3 -c "import json,sys;print('%.2f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null
    done | sort -n | sed -n 2p)
  local BUG; BUG=$(docker logs "$NAME" 2>&1 | grep -ac 'BUG')
  # The layer index is NOT a proxy for read volume: early layers have more
  # concentrated expert usage and so miss less. Ask the runtime what it actually
  # read instead of inferring it from the drop_from setting.
  docker kill -s USR1 "$NAME" >/dev/null 2>&1; sleep 4
  local BPT; BPT=$(docker logs "$NAME" 2>&1 | grep -a 'bytes/token' | tail -1 \
                   | sed 's/.*bytes\/token \([0-9.]*\) MiB.*/\1/')
  local HR; HR=$(docker logs "$NAME" 2>&1 | grep -a 'hit rate' | tail -1 | sed 's/.*hit rate \([0-9.]*\)%.*/\1/')
  local OUT; OUT=$(ask '{"prompt":"The capital of France is","n_predict":8,"temperature":0,"top_k":1}' \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['content'][:36].replace(chr(10),' '))" 2>/dev/null)
  printf "%-30s  %8s ms/tok  %8s MiB/tok  BUG=%-3s hit=%-7s out='%s'\n" \
    "$1" "${D:-FAIL}" "${BPT:-n/a}" "$BUG" "${HR:-n/a}" "$OUT"
}

echo "## S20 -- decode time vs decode read volume   $(date -Iseconds)"
echo "   model $MODEL_FILE / frac ${MOESTREAM_CACHE_FRAC} / ub ${UBATCH} / ctx ${CTX_SIZE}"
echo "   $NL layers; drop_from=N removes misses in layers N..$((NL-1))"
echo "   NOTE: output for drop_from<$NL is intentionally corrupt. Speed probe only.
   MiB/tok is the runtime's own counter, so the curve is decode-time against
   MEASURED read volume, not against the drop_from setting."
echo
printf "%-30s  %8s\n" "configuration" "decode"
run "MOESTREAM=0 (no streaming)"     "-e MOESTREAM=0"
run "NOOP=3 (zero I/O)"              "-e MOESTREAM=1 -e MOESTREAM_NOOP=3"
run "drop_from=0  (-100% reads)"     "-e MOESTREAM=1 -e MOESTREAM_DROP_FROM=0"
run "drop_from=10 (-75% reads)"      "-e MOESTREAM=1 -e MOESTREAM_DROP_FROM=10"
run "drop_from=20 (-50% reads)"      "-e MOESTREAM=1 -e MOESTREAM_DROP_FROM=20"
run "drop_from=30 (-25% reads)"      "-e MOESTREAM=1 -e MOESTREAM_DROP_FROM=30"
run "normal (full reads)"            "-e MOESTREAM=1"
docker rm -f "$NAME" >/dev/null 2>&1
