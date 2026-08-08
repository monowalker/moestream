#!/usr/bin/env bash
# MoEStream — llama-server launch wrapper
#
#   Policy: anything derivable from the model or the host is never written to
#   .env. A stale value left behind after swapping models is a silent hazard,
#   so .env holds only the knobs a human actually chooses.
#
#   Derived automatically:
#     top_k             read from GGUF metadata (llama-moestream.cpp: gguf_topk)
#     slab slot count   MOESTREAM_CACHE_FRAC x n_expert
#     arena threshold   largest token count the slab can serve without eviction
#     sampling defaults taken from the model itself when present in the GGUF
#     thread count      physical core count
#     BATCH             raised to at least UBATCH
set -euo pipefail

MODEL_PATH="/models/${MODEL_FILE:?MODEL_FILE not set}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[moestream] ERROR: model not found: $MODEL_PATH" >&2
  echo "[moestream]   check MODEL_DIR / MODEL_FILE in .env" >&2
  exit 1
fi

# ---- GPU access check -------------------------------------------------------
# Without this check a missing GID silently falls back to the CPU backend,
# which merely looks "unexplainably slow". Fail fast and name the fix instead.
DRI_NODE=$(ls /dev/dri/renderD* 2>/dev/null | head -1 || true)
if [[ -z "$DRI_NODE" ]]; then
  echo "[moestream] ERROR: /dev/dri is not visible; check 'devices:' in compose.yaml" >&2
  exit 1
fi
if [[ ! -r "$DRI_NODE" || ! -w "$DRI_NODE" ]]; then
  echo "[moestream] ERROR: cannot access $DRI_NODE (missing group membership)" >&2
  echo "[moestream]   run these on the host and set RENDER_GID / VIDEO_GID in .env:" >&2
  echo "[moestream]     getent group render | cut -d: -f3" >&2
  echo "[moestream]     getent group video  | cut -d: -f3" >&2
  exit 1
fi

# Pin the RADV ICD. Without it Vulkan may select llvmpipe (software rasterizer).
RADEON_ICD=$(ls /usr/share/vulkan/icd.d/radeon_icd*.json 2>/dev/null | head -1 || true)
if [[ -n "$RADEON_ICD" ]]; then
  export VK_ICD_FILENAMES="$RADEON_ICD"
  echo "[moestream] Vulkan ICD: $RADEON_ICD"
else
  echo "[moestream] WARNING: RADV ICD not found; may fall back to CPU" >&2
fi

# MoEStream reads expert tensors straight from the GGUF, so it needs the path.
export MOESTREAM_GGUF="${MOESTREAM_GGUF:-$MODEL_PATH}"

# ---- derived values ---------------------------------------------------------
UBATCH="${UBATCH:-1024}"

# UBATCH=learn: try candidates once each, then keep the fastest.
#   The optimum is strongly model-dependent (measured: 1024 for Ornith-35B,
#   8096 for Laguna -- a 45% difference in the opposite direction) and cannot be
#   extrapolated from a single run, because per-token compute grows
#   super-linearly with UBATCH (docs/RESULTS.md §10.11). Comparing measured
#   points across runs is the only sound way to find it.
if [[ "$UBATCH" == "learn" ]]; then
  # Best effort only. `set -e` is off here on purpose: this is a convenience,
  # and no failure in it may stop the server from starting.
  set +e
  ST="${MOESTREAM_STATE_DIR:-/state}"
  # awk exits non-zero on a missing file, which would be indistinguishable from
  # "already measured". Point at /dev/null instead of writing anything: the
  # state directory may not be writable yet on a first start.
  TUN="$ST/tuning.tsv"; [[ -f "$TUN" ]] || TUN=/dev/null
  UBT="$ST/ubatch.tsv"; [[ -f "$UBT" ]] || UBT=/dev/null
  # The frac this run will use. Measurements are only comparable within one
  # frac: on Ornith the best ubatch is 1024 at frac 0.25 but 2048 at frac 0.15,
  # so mixing them converges on the wrong answer.
  FRAC="${MOESTREAM_CACHE_FRAC:-0.25}"
  [[ "$FRAC" == "learn" ]] && FRAC=$(awk -F'\t' -v m="$MODEL_FILE" \
      '$1==m {printf "%.2f", $2}' "$TUN" 2>/dev/null)
  [[ -z "$FRAC" || "$FRAC" == "learn" || "$FRAC" == "auto" ]] && FRAC=0.15
  FRAC=$(printf '%.2f' "$FRAC" 2>/dev/null || echo 0.15)

  UB=""
  for c in ${UBATCH_CANDIDATES:-1024 2048 4096 8192}; do
    # Untried at this frac? Take it. Rows are only written for runs that served
    # at least two requests, so every row present is a real measurement.
    awk -F'\t' -v m="$MODEL_FILE" -v f="$FRAC" -v u="$c" \
        '$1==m && $2==f && $3==u {found=1} END{exit found}' "$UBT" \
      && { UB="$c"; echo "[moestream] UBATCH=learn: measuring $c at frac=$FRAC"; break; }
  done
  if [[ -z "$UB" ]]; then
    read -r UB RATE <<< "$(awk -F'\t' -v m="$MODEL_FILE" -v f="$FRAC" \
        '$1==m && $2==f && $4+0>0 {if ($4+0>b) {b=$4+0; p=$3}} END{if (p) print p, b}' \
        "$UBT")"
    [[ -n "$UB" ]] && echo "[moestream] UBATCH=learn: using $UB at frac=$FRAC ($RATE tok/s prefill)"
  fi
  [[ -z "$UB" ]] && { UB=1024; echo "[moestream] UBATCH=learn: nothing measured yet; starting at $UB"; }

  UBATCH="$UB"
  export MOESTREAM_UBATCH="$UBATCH"      # recorded against the configured value
  export MOESTREAM_CACHE_FRAC="${MOESTREAM_CACHE_FRAC:-learn}"
  set -e
fi

# BATCH must be >= UBATCH. Leaving that to the operator invites misconfiguration.
BATCH_AUTO=4096
(( UBATCH > BATCH_AUTO )) && BATCH_AUTO=$UBATCH
BATCH="${BATCH:-$BATCH_AUTO}"
if (( BATCH < UBATCH )); then
  echo "[moestream] BATCH($BATCH) < UBATCH($UBATCH); raising BATCH to $UBATCH"
  BATCH=$UBATCH
fi
# Use physical cores, not nproc. nproc counts SMT siblings, and passing that
# to llama.cpp is measurably slower (e.g. 16 on an 8C/16T part).
PHYS_CORES=$(awk '/^cpu cores/{print $4; exit}' /proc/cpuinfo 2>/dev/null || true)
[[ -z "$PHYS_CORES" || "$PHYS_CORES" -lt 1 ]] && PHYS_CORES=$(nproc)
THREADS="${THREADS:-$PHYS_CORES}"

if [[ "${MOESTREAM:-0}" != "0" ]]; then
  if [[ "${MOESTREAM_PREFILL_ARENA:-1}" == "0" ]]; then
    # Without the arena the slab must hold the union of one full ubatch.
    # Deriving it beats asking the operator to keep two values in sync.
    export MOESTREAM_MAX_UBATCH="$UBATCH"
    export MOESTREAM_PREFILL_THRESHOLD=0
    echo "[moestream] prefill arena disabled; growing slab for ubatch=$UBATCH (uses more memory)"
  fi
  echo "[moestream] streaming enabled (cache_frac=${MOESTREAM_CACHE_FRAC:-0.25}, ubatch=$UBATCH, arena=${MOESTREAM_PREFILL_ARENA:-1})"
fi

# ---- sampling ---------------------------------------------------------------
# Pass through only what was explicitly set. Anything omitted falls back to the
# model's own defaults from the GGUF. Hardcoding values here would carry one
# model's settings over to the next one silently.
SAMPLING=()
[[ -n "${TEMP:-}"             ]] && SAMPLING+=(--temp "$TEMP")
[[ -n "${TOP_P:-}"            ]] && SAMPLING+=(--top-p "$TOP_P")
[[ -n "${TOP_K:-}"            ]] && SAMPLING+=(--top-k "$TOP_K")
[[ -n "${MIN_P:-}"            ]] && SAMPLING+=(--min-p "$MIN_P")
[[ -n "${PRESENCE_PENALTY:-}" ]] && SAMPLING+=(--presence-penalty "$PRESENCE_PENALTY")
if (( ${#SAMPLING[@]} )); then
  echo "[moestream] sampling overridden by .env: ${SAMPLING[*]}"
else
  echo "[moestream] sampling: using model defaults"
fi

# ---- Web UI -----------------------------------------------------------------
UI_ARGS=()
UI_DIR=/opt/llama.cpp/build/tools/ui/dist
[[ -d "$UI_DIR" && "${ENABLE_WEBUI:-1}" != "0" ]] && UI_ARGS=(--path "$UI_DIR")

# shellcheck disable=SC2086
exec /opt/llama.cpp/build/bin/llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  ${API_KEY:+--api-key "$API_KEY"} \
  -c "${CTX_SIZE:-32768}" \
  -ngl "${N_GPU_LAYERS:-99}" \
  -b "$BATCH" \
  -ub "$UBATCH" \
  -t "$THREADS" \
  --parallel "${N_PARALLEL:-1}" \
  --cont-batching \
  ${FLASH_ATTN:+-fa "$FLASH_ATTN"} \
  --cache-type-k "${CACHE_TYPE_K:-q8_0}" \
  --cache-type-v "${CACHE_TYPE_V:-q8_0}" \
  --ctx-checkpoints "${CTX_CHECKPOINTS:-4}" \
  --cache-ram "${CACHE_RAM:-2048}" \
  --jinja \
  ${CHAT_TEMPLATE:+--chat-template-file "$CHAT_TEMPLATE"} \
  --reasoning-format "${REASONING_FORMAT:-auto}" \
  --chat-template-kwargs "{\"enable_thinking\":${ENABLE_THINKING:-false}}" \
  "${SAMPLING[@]}" \
  --metrics \
  "${UI_ARGS[@]}" \
  ${SPEC_DECODING:-} \
  ${EXTRA_ARGS:-} \
  "$@"
