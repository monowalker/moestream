#!/usr/bin/env bash
# =============================================================================
# Spike S31 -- does MTP help a streaming MoE model? (S28 left this unresolved)
#
# S28 ran Ornith-1.5 with --spec-type draft-mtp and got acceptance = 0.00000
# (0 accepted / 3 generated). Decode was unchanged, which is what you would
# expect if no speculation actually happened. That is not an answer to "does
# MTP help MoE streaming" -- it is a measurement that did not run.
#
# The control S28 never took: does MTP accept anything on this model with
# MoEStream OFF? If acceptance is zero there too, the model or the runtime is
# the cause and MoEStream is not implicated. If it works with streaming off and
# not on, that is a MoEStream bug and matters far more than the speed question.
#
# Prediction from S13's arithmetic, if speculation does run: union(4)/union(1)
# = 30.5/8 = 3.8x more experts per pass on Ornith, so ~3.8 of 4 tokens must be
# accepted to break even. MTP should still lose.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || { echo locked >&2; exit 9; }
fi
trap 'docker rm -f ms-s31 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
NAME=ms-s32; PORT=18104
MTP='--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7'

run() {  # $1=label $2=model $3=ctx $4=env $5=spec
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$2" -e CTX_SIZE="$3" -e UBATCH=1024 -e N_PARALLEL=1 \
    -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 $4 ${5:+-e SPEC_DECODING="$5"} \
    -p "$PORT":8080 moestream/server:local >/dev/null || { printf "%-40s BOOT FAILED\n" "$1"; return; }
  local ok=0
  for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    docker ps -q -f name="$NAME" | grep -q . || break; sleep 3; done
  if [ $ok = 0 ]; then printf "%-40s DIED\n" "$1"; docker logs "$NAME" 2>&1|tail -4; return; fi
  local V G MEM; V=$(cat /sys/class/drm/card*/device/mem_info_vram_used|head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used|head -1)
  MEM=$(echo "scale=2;($V+$G)/1073741824"|bc)
  # longer generation: MTP acceptance needs a real sample, not 40 tokens
  curl -s -m 1800 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a detailed essay about distributed consensus.","n_predict":200,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local D; D=$(curl -s -m 1800 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a detailed essay about distributed consensus.","n_predict":200,"temperature":0,"top_k":1,"cache_prompt":true}' \
    | python3 -c "import json,sys;print('%.1f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null)
  local ACC; ACC=$(docker logs "$NAME" 2>&1 | grep -aoE "acceptance = [0-9.]+ \([^)]*\)" | tail -1)
  printf "%-40s %7s GiB %9s ms/tok  %s\n" "$1" "$MEM" "${D:-FAIL}" "${ACC:-no acceptance line}"
  docker rm -f "$NAME" >/dev/null 2>&1
}

echo "## S32 -- MoE MTP: does a larger draft help or widen the union?   $(date -Iseconds)"
echo "   Ornith-1.5-35B-Q4_K_M (nextn_predict_layers = 1). Ornith-1.0 has no MTP head."
echo "   200-token generations, so acceptance is sampled properly."
echo
echo "   S31 settled the sign at n_max=3: MTP loses 6-11% on MoE at 88.7% acceptance."
echo "   The open question is whether a larger draft helps. union(K) grows with K,"
echo "   so the prediction is that it gets worse -- the opposite of dense (S29),"
echo "   where n_max 1->7 improved 394.7 -> 291.9 ms."
echo
run "frac=0.40, no MTP (reference)" Ornith-1.5-35B-Q4_K_M.gguf 8192 "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.40" ""
for n in 1 3 5 7; do
  run "frac=0.40 + MTP n_max=$n" Ornith-1.5-35B-Q4_K_M.gguf 8192 \
      "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.40" \
      "--spec-type draft-mtp --spec-draft-n-max $n --spec-draft-p-min 0.7"
done
