#!/usr/bin/env bash
# =============================================================================
# Spike S25 -- perplexity on the DECODE path
#
# S24 measured PPL for MOESTREAM_SKIP_RANK at 7, 6 and 5 and got 5.1040 every
# time, identical to four decimals and identical to the no-skip reference --
# while the generated text visibly degraded. Both cannot be true.
#
# The explanation is in prefill_exps():
#
#     if (!g_pf_ready || n_tokens <= g_pf_threshold) -> slab path
#     else                                           -> arena path
#
# The threshold here is 6. llama-perplexity was run with -ub 512, so every
# evaluation took the ARENA path, which does not go through remap_exec and
# therefore never reaches the skip predicate. The PPL numbers were real; they
# just measured a code path the change does not touch.
#
# This runs the same corpus with -ub 4, below the threshold, so evaluation goes
# through the slab and remap_exec exactly as decode does. Slower (128 micro-
# batches per 512-token chunk) but it measures the thing under test.
#
# Both references are re-taken at -ub 4: batching changes the floating-point
# reduction order, so a -ub 512 number is not a valid baseline for these.
#
# ---------------------------------------------------------------------------
# Note for RESULTS.md §8: the published PPL figures use -c 512 -b 512 -ub 512,
# which by the same argument exercised the arena path. The decode/slab path's
# quality evidence rests on the token-identity check, not on those PPL numbers.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || {
    echo "another spike holds /tmp/moestream-spike.lock; refusing to run" >&2; exit 9; }
fi
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
# The patch is kept as research/spikes/s24_skiprank/skiprank.patch rather than
# in src/, because S25 rejected the idea (+10.2% PPL for -3.2% decode). To
# reproduce: apply it to src/llama-moestream.cpp, rebuild the normal image, and
# point IMG at it.
IMG=${IMG:-moestream/server:local}
UB=${UB:-4}
CHUNKS=${CHUNKS:-10}

ppl() {  # $1=label $2=env
  local OUT P X
  # shellcheck disable=SC2086
  OUT=$(timeout 5400 docker run --rm --entrypoint /opt/llama.cpp/build/bin/llama-perplexity \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -v "$PWD/research/bench":/bench:ro \
    -e XDG_RUNTIME_DIR=/tmp -e MOESTREAM_GGUF="/models/$MODEL_FILE" $2 "$IMG" \
    -m "/models/$MODEL_FILE" -f /bench/ppl.txt --chunks "$CHUNKS" \
    -c 512 -b 512 -ub "$UB" -ngl 99 2>&1)
  P=$(echo "$OUT" | grep -oE 'Final estimate: PPL = [0-9.]+' | grep -oE '[0-9.]+$')
  # prove the path: slab-path builds should dominate, and the skip must fire
  X=$(echo "$OUT" | grep -a 'graph builds' | tail -1 | sed 's/moestream: //')
  local S; S=$(echo "$OUT" | grep -a '\[S17\] rank skip' | tail -1 | sed 's/moestream: //')
  printf "%-24s PPL = %-11s  %s\n" "$1" "${P:-FAILED}" "$X"
  [ -n "$S" ] && echo "                         $S"
}

echo "## S25 -- perplexity through the slab/decode path   $(date -Iseconds)"
echo "   $MODEL_FILE / frac ${MOESTREAM_CACHE_FRAC} / -ub $UB (threshold is 6) / $CHUNKS chunks"
echo "   at -ub 512 every config gave 5.1040 because evaluation went via the arena"
echo
ppl "MOESTREAM=0"    "-e MOESTREAM=0"
ppl "no skip"        "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25"
ppl "SKIP_RANK=7"    "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25 -e MOESTREAM_SKIP_RANK=7"
ppl "SKIP_RANK=6"    "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25 -e MOESTREAM_SKIP_RANK=6"
ppl "SKIP_RANK=5"    "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25 -e MOESTREAM_SKIP_RANK=5"
