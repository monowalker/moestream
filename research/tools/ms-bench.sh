#!/usr/bin/env bash
# =============================================================================
# ms-bench.sh — measure MoEStream in a form other people can reproduce
#
#   Usage:
#     research/tools/ms-bench.sh                     # measure using the current settings
#     research/tools/ms-bench.sh --baseline          # also A/B against MOESTREAM=0
#     research/tools/ms-bench.sh --frac 0.25 --ub 1024
#     research/tools/ms-bench.sh --model x.gguf --ctx 32768
#     research/tools/ms-bench.sh --dense-frac 0.00    # a dense model's FFN
#     research/tools/ms-bench.sh --spec              # with speculative decoding
#
#   This script never reports speed on its own. Alongside it, it always reports
#   (a) the environment, (b) the actual slot count, (c) the page-cache state and
#   (d) whether the output is correct.
#   A speed-only table can look excellent while the implementation is broken --
#   which is exactly what happened on 2026-08-06 (docs/RESULTS.md §10.8).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root: .env and research/bench live there

FRAC=""; UB=""; BASELINE=0; PORT=18080; NAME=ms-bench; DFRAC=""; SPEC=0; MODEL_ARG=""; CTX_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --frac)     FRAC="$2"; shift 2 ;;
    --ub)       UB="$2";   shift 2 ;;
    --baseline) BASELINE=1; shift ;;
    --dense-frac) DFRAC="$2"; shift 2 ;;
    --model)    MODEL_ARG="$2"; shift 2 ;;
    --ctx)      CTX_ARG="$2"; shift 2 ;;
    --spec)     SPEC=1; shift ;;
    --port)     PORT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -f .env ] || { echo "no .env found; run: cp .env.example .env" >&2; exit 1; }
set -a
. ./.env
# `make launch` writes its answers to .env.launcher, and compose.yaml layers that
# file over .env. Read it here too, or this measures a different model from the
# one actually running.
[ -f .env.launcher ] && . ./.env.launcher
set +a

# Sourcing .env overwrites anything the caller set, so an explicit flag has to be
# applied afterwards. Measuring a different model from the one named on the
# command line is the kind of error that produces a whole table of wrong numbers
# without a single visible failure.
[ -n "$MODEL_ARG" ] && MODEL_FILE="$MODEL_ARG"
[ -n "$CTX_ARG" ]   && CTX_SIZE="$CTX_ARG"
[ -n "$FRAC" ] && MOESTREAM_CACHE_FRAC="$FRAC"
[ -n "$UB" ]   && UBATCH="$UB"

# ---------------------------------------------------------------- environment
echo "============================================================"
echo " MoEStream benchmark  $(date -Iseconds)"
echo "============================================================"
echo "## Environment"
printf "  CPU        : %s\n" "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
printf "  cores      : %s physical (%s logical)\n" \
    "$(lscpu -p=CORE 2>/dev/null | grep -vc '^#' >/dev/null && lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l || echo '?')" \
    "$(grep -c '^processor' /proc/cpuinfo)"
printf "  RAM        : %.1f GiB\n" "$(awk '/MemTotal/{print $2/1048576}' /proc/meminfo)"
for d in /sys/class/drm/card*/device; do
  [ -f "$d/mem_info_gtt_total" ] || continue
  printf "  GTT limit  : %.1f GiB   <- the fits/does-not-fit boundary; always quote it\n" \
      "$(echo "scale=3;$(cat "$d/mem_info_gtt_total")/1073741824" | bc)"
  printf "  VRAM limit : %.1f GiB\n" \
      "$(echo "scale=3;$(cat "$d/mem_info_vram_total")/1073741824" | bc)"
  break
done
MODELPATH="$MODEL_DIR/$MODEL_FILE"
printf "  model      : %s (%.2f GiB)\n" "$MODEL_FILE" \
    "$(du -cb "${MODELPATH%-00001-of-*}"* 2>/dev/null | tail -1 | awk '{print $1/1073741824}')"
# Resolve the block device holding the models (partition -> parent disk)
_src=$(df --output=source "$MODEL_DIR" 2>/dev/null | tail -1)
_pk=$(lsblk -ndo PKNAME "$_src" 2>/dev/null); [ -z "$_pk" ] && _pk=$(basename "$_src")
printf "  SSD        : %s (rotational=%s)\n" \
    "$(cat "/sys/block/$_pk/device/model" 2>/dev/null || echo unknown)" \
    "$(cat "/sys/block/$_pk/queue/rotational" 2>/dev/null || echo '?')"
echo
echo "## Settings (the knobs a human chooses)"
printf "  MOESTREAM_CACHE_FRAC = %s\n" "${MOESTREAM_CACHE_FRAC:-1.0}"
printf "  UBATCH               = %s\n" "${UBATCH:-512}"
printf "  CTX_SIZE             = %s\n" "${CTX_SIZE:-4096}"
printf "  N_PARALLEL           = %s / KV = %s,%s / FA = %s\n" \
    "${N_PARALLEL:-1}" "${CACHE_TYPE_K:-f16}" "${CACHE_TYPE_V:-f16}" "${FLASH_ATTN:-auto}"
echo

# ---------------------------------------------------------------- boot
boot() {  # $1 = MOESTREAM 0/1
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 3
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$MODEL_FILE" -e CTX_SIZE="${CTX_SIZE:-32768}" -e UBATCH="${UBATCH:-1024}" \
    -e N_PARALLEL="${N_PARALLEL:-1}" -e FLASH_ATTN="${FLASH_ATTN:-on}" \
    -e CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}" -e CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}" \
    -e ENABLE_WEBUI=0 -e MOESTREAM="$1" -e MOESTREAM_CACHE_FRAC="${MOESTREAM_CACHE_FRAC:-0.25}" \
    ${DFRAC:+-e MOESTREAM_DENSE_FRAC=$DFRAC} \
    ${SPECARG:+--env-file $SPECFILE} \
    -p "$PORT":8080 moestream/server:local >/dev/null || return 1
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done
  return 1
}

# Speculative decoding is one env var whose value contains spaces, so it cannot
# ride on the -e word-splitting the rest uses; give it a file of its own.
SPECFILE=""; SPECARG=""
if [ "$SPEC" = 1 ]; then
  SPECFILE=$(mktemp); SPECARG=1
  TYPES=$(docker run --rm -v "$MODEL_DIR":/models:ro \
      --entrypoint /usr/local/bin/moestream-spec-probe moestream/server:local \
      "/models/$MODEL_FILE" 2>/dev/null | paste -sd, -)
  if [ -z "$TYPES" ]; then
    echo "  --spec: llama.cpp reports this model cannot self-speculate; ignoring"
    SPECARG=""; rm -f "$SPECFILE"; SPECFILE=""
  else
    echo "SPEC_DECODING=--spec-type $TYPES --spec-draft-n-max 3 --spec-draft-p-min 0.7" > "$SPECFILE"
    echo "  --spec: llama.cpp reports $TYPES"
  fi
fi
trap '[ -n "$SPECFILE" ] && rm -f "$SPECFILE"' EXIT

ask() { curl -s -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$1"; }

measure() {  # $1 = label
  local L="$1"
  # (c) page-cache state: cold vs warm swings prefill by over 20%
  local PC; PC=$(awk '/^Cached:/{printf "%.1f", $2/1048576}' /proc/meminfo)
  local GPU; GPU=$(docker logs "$NAME" 2>&1 | grep -am1 'ggml_vulkan: 0 = ' | sed 's/.*0 = //;s/ *|.*//')
  [ -n "$GPU" ] && echo "  GPU            : $GPU"
  # Recover the automatically derived values (slot count, arena, strategy)
  local SLOTS; SLOTS=$(docker logs "$NAME" 2>&1 | grep -a 'slots/layer' | tail -1 | sed 's/^moestream: //')
  local ARENA; ARENA=$(docker logs "$NAME" 2>&1 | grep -a 'prefill arena' | tail -1 | sed 's/^moestream: //')
  local PFMODE; PFMODE=$(docker logs "$NAME" 2>&1 | grep -a 'read strategy' | tail -1 | sed 's/.*= //')

  # (d) output correctness, checked before any speed number
  local OUT; OUT=$(ask '{"prompt":"The capital of France is","n_predict":24,"temperature":0,"top_k":1}' \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['content'][:80])" 2>/dev/null)
  local BUG; BUG=$(docker logs "$NAME" 2>&1 | grep -ac 'BUG')

  # Device memory
  local V G MEM
  V=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used  2>/dev/null | head -1)
  MEM=$(echo "scale=2;($V+$G)/1073741824" | bc)

  # prefill: use real text. Synthetic tokens destroy expert locality and make
  #          MoEStream look unfairly slow (llama-bench uses std::rand).
  #   The prompt is ~13900 tokens. With a smaller context the request fails and
  #   the result silently reads "n/a" with no reason given -- which is exactly
  #   how a measurement was lost during development. Say so instead.
  local PF="n/a"
  local PROMPT_TOK=13900
  if [ -f research/bench/prompt_long.txt ] && [ "${CTX_SIZE:-32768}" -lt "$PROMPT_TOK" ]; then
    PF="skipped"
    echo "  NOTE: CTX_SIZE=${CTX_SIZE} is below the ~${PROMPT_TOK}-token benchmark"
    echo "        prompt, so prefill cannot be measured. Raise CTX_SIZE to compare."
  elif [ -f research/bench/prompt_long.txt ]; then
    PF=$(python3 -c "
import json;print(json.dumps({'prompt':open('research/bench/prompt_long.txt').read(),
'n_predict':1,'temperature':0,'top_k':1,'cache_prompt':False}))" \
      | curl -s -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @- \
      | python3 -c "
import json,sys
try:
    print('%.1f' % json.load(sys.stdin)['timings']['prompt_per_second'])
except Exception as e:
    print('FAILED')            # do not hide it behind an empty value
" 2>/dev/null)
    [ -z "$PF" ] && PF="FAILED"
  fi

  # decode: median of 3 warm runs. A single sample is unreliable (§13.3).
  ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":200,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local DEC; DEC=$(for _ in 1 2 3; do
      ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":100,"temperature":0,"top_k":1,"cache_prompt":true}' \
        | python3 -c "import json,sys;print('%.1f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null
    done | sort -n | sed -n 2p)

  echo "### $L"
  [ -n "$SLOTS" ]  && echo "  auto: $SLOTS"
  [ -n "$ARENA" ]  && echo "  auto: $ARENA"
  [ -n "$PFMODE" ] && echo "  auto: prefill read strategy = $PFMODE"
  printf "  page cache at start : %s GiB\n" "$PC"
  printf "  device memory       : %s GiB\n" "$MEM"
  printf "  prefill             : %s tok/s   (13877-token real document)\n" "$PF"
  printf "  decode              : %s ms/tok  (%.2f tok/s, median of 3)\n" "$DEC" \
      "$(python3 -c "print(1000/$DEC)" 2>/dev/null || echo 0)"
  printf "  output correctness  : '%s'\n" "$OUT"
  printf "  [BUG] log lines     : %s  %s\n" "$BUG" \
      "$([ "$BUG" -eq 0 ] && echo '(clean)' || echo 'READS ARE BEING DISCARDED; these numbers are invalid')"
  echo
  # Internal statistics (hit rate, I/O share)
  docker kill -s USR1 "$NAME" >/dev/null 2>&1; sleep 3
  docker logs "$NAME" 2>&1 | grep -aE '\[stats\]|\[mrc\]|\[io\]' | tail -4 | sed 's/^/  /'
  echo
}

echo "## Results"
echo
if [ "$BASELINE" = 1 ]; then
  if boot 0; then measure "baseline: plain llama.cpp (MOESTREAM=0)"
  else echo "### baseline: plain llama.cpp"; echo "  failed to start -- this model does not fit in this machine's memory"; echo; fi
fi
if boot 1; then measure "MoEStream (${DFRAC:+dense_frac=$DFRAC, }frac=${MOESTREAM_CACHE_FRAC}, ub=${UBATCH})"
else echo "### MoEStream"; echo "  failed to start"; docker logs "$NAME" 2>&1 | tail -5; fi

docker rm -f "$NAME" >/dev/null 2>&1
echo "============================================================"
echo "note: llama-bench is bundled too, but (§12.2):"
echo "    - its default -p 512 overstates MoEStream (113% of plain llama.cpp,"
echo "      i.e. 'streaming is faster'). Use -p 4096 or more."
echo "    - without MOESTREAM_GGUF, top_k silently falls back to a default"
echo "    - decode (tg) is measured correctly by llama-bench (74% vs 71% here)"
echo "============================================================"
