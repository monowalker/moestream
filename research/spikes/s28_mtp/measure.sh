#!/usr/bin/env bash
# =============================================================================
# Spike S28 -- does MTP self-speculation help or hurt a streaming model?
#
# Finding S13 rejected speculative decoding for MoE: a verification pass with K
# tokens references union(K) experts instead of top_k, so I/O per ACCEPTED token
# rises. It measured ngram-cache (-35%) and a separate 4B draft model (-79%).
#
# It did not measure MTP, and could not: Ornith-1.0 has no nextn_predict_layers
# in its GGUF, so the model has no MTP head. Ornith-1.5 does, and so does
# Qwen3.8-27B -- which is what the stock llama.cpp server on this machine runs
# in production (--spec-type draft-mtp --spec-draft-n-max 3).
#
# The prediction, from the same arithmetic S13 used:
#
#   dense : a pass reads the same bytes whatever the token count, so K tokens
#           per pass divides streamed bytes per token by the acceptance count.
#           Speculation should HELP, and help more the more is streamed.
#   MoE   : union(4)/union(1) = 30.5/8 = 3.8x on Ornith, so ~3.8 of 4 tokens
#           must be accepted just to break even. Should still HURT.
#
# If both hold, the rule is: speculation pays exactly to the extent that a
# pass's read volume is independent of how many tokens are in it.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || {
    echo "another spike holds the lock; refusing to run" >&2; exit 9; }
fi
trap 'docker rm -f ms-s28 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
NAME=ms-s28; PORT=18091
MTP='--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7'

boot() {  # $1 = model  $2 = ctx  $3 = plain env  $4 = spec string ("" = none)
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # $3 is intentionally unquoted (word-split into flags); $4 must stay one argv
  # element, which is what broke the first attempt at this measurement.
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$1" -e CTX_SIZE="$2" -e UBATCH=1024 \
    -e N_PARALLEL=1 -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 $3 ${4:+-e SPEC_DECODING="$4"} \
    -p "$PORT":8080 moestream/server:local >/dev/null || return 1
  for i in $(seq 1 500); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done; return 1
}
ask() { curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$1"; }

run() {  # $1=label $2=model $3=ctx $4=env $5=spec
  if ! boot "$2" "$3" "$4" "$5"; then printf "%-38s FAILED TO BOOT\n" "$1"; docker logs "$NAME" 2>&1|tail -6; return; fi
  local V G MEM
  V=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null|head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null|head -1)
  MEM=$(echo "scale=2;($V+$G)/1073741824"|bc)
  ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":40,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local D; D=$(for _ in 1 2 3; do
      ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":60,"temperature":0,"top_k":1,"cache_prompt":true}' \
      | python3 -c "import json,sys;print('%.1f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null
    done | sort -n | sed -n 2p)
  local OUT; OUT=$(ask '{"prompt":"The capital of France is","n_predict":10,"temperature":0,"top_k":1}' \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['content'][:34].replace(chr(10),' '))" 2>/dev/null)
  local ACC; ACC=$(docker logs "$NAME" 2>&1 | grep -aoE "accept[^,]*" | tail -1)
  printf "%-38s %7s GiB %9s ms/tok  '%s' %s\n" "$1" "$MEM" "${D:-FAIL}" "$OUT" "$ACC"
  docker rm -f "$NAME" >/dev/null 2>&1
}

echo "## S28 -- MTP against a streaming model   $(date -Iseconds)"
echo "   MTP = $MTP"
echo
echo "### dense: Qwen3.8-27B-IQ4_NL   (a pass reads the same bytes at any token count)"
echo "   from S27: plain 211.0 / frac0.40 512.5 / frac0.00 690.8 ms per token"
run "dense plain, no MTP"        Qwen3.8-27B-IQ4_NL.gguf 16384 "-e MOESTREAM=0" ""
run "dense plain + MTP"          Qwen3.8-27B-IQ4_NL.gguf 16384 "-e MOESTREAM=0" "$MTP"
run "dense frac=0.40, no MTP"    Qwen3.8-27B-IQ4_NL.gguf 16384 "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.40" ""
run "dense frac=0.40 + MTP"      Qwen3.8-27B-IQ4_NL.gguf 16384 "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.40" "$MTP"
run "dense frac=0.00, no MTP"    Qwen3.8-27B-IQ4_NL.gguf 16384 "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" ""
run "dense frac=0.00 + MTP"      Qwen3.8-27B-IQ4_NL.gguf 16384 "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "$MTP"
echo
echo "### MoE: Ornith-1.5-35B-Q4_K_M   (union grows with token count)"
echo "   Ornith-1.0 has no nextn_predict_layers, so S13 could not test MTP at all"
run "MoE expert streaming, no MTP" Ornith-1.5-35B-Q4_K_M.gguf 32768 "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.40" ""
run "MoE expert streaming + MTP"   Ornith-1.5-35B-Q4_K_M.gguf 32768 "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.40" "$MTP"
