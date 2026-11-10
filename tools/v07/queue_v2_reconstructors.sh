#!/usr/bin/env bash
# v2 reconstructor sweep: more categories, real-mask mixing, learned vs telea
# diagnostics and qualitative panels.
set -u

REPO="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型"
RUNS="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs"
PY="/home/asus/anaconda3/envs/cad-v0/bin/python"
LOG="$RUNS/v07/reconstructors/queue_v2.log"
DONE="$RUNS/v07/reconstructors/queue_v2.done"

mkdir -p "$RUNS/v07/reconstructors"
exec >>"$LOG" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

latest_checkpoint() {
  local category="$1"
  find "$RUNS/v07/reconstructors/$category" -name checkpoint.pt 2>/dev/null | sort | tail -1
}

train_and_eval() {
  local gpu="$1"
  local category="$2"
  local name="$3"
  local cache="$4"
  log "START train ${category}/${name} on ${gpu}"
  if [ -n "$cache" ]; then
    "$PY" tools/v07/train_reconstructor.py \
      --category "$category" --device "$gpu" --name "$name" --steps 6000 \
      --real-mask-fraction 0.5 --supports-cache "$cache"
  else
    "$PY" tools/v07/train_reconstructor.py \
      --category "$category" --device "$gpu" --name "$name" --steps 6000 \
      --real-mask-fraction 0.5
  fi
  local rc=$?
  log "END train ${category}/${name} rc=${rc}"
  local checkpoint
  checkpoint=$(latest_checkpoint "$category")
  if [ -z "$checkpoint" ]; then
    log "NO CHECKPOINT for ${category}; skip diag"
    return
  fi
  log "DIAG ${category} checkpoint=${checkpoint}"
  "$PY" tools/v07/diag_proposal.py \
    --categories "$category" \
    --backends neighbor telea ring learned \
    --learned-root "$RUNS/v07/reconstructors" \
    --normals-per-type 2 --sources-per-type 2 --max-probes-per-type 4 \
    --atlas-references 4 --device "$gpu" --name "${category}_learned_diag_v2"
  log "END diag ${category} rc=$?"
  log "VISUAL ${category} checkpoint=${checkpoint}"
  "$PY" tools/v07/visualize_reconstruction.py \
    --category "$category" --checkpoint "$checkpoint" --samples 6 --device "$gpu"
  log "END visual ${category} rc=$?"
}

cd "$REPO" || exit 1
export PYTHONPATH=src
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

(
  train_and_eval cuda:0 bottle bottle_v5 "$RUNS/v07/reconstructors/bottle/supports.npz"
  train_and_eval cuda:0 capsule capsule_v2 ""
  train_and_eval cuda:0 grid grid_v2 ""
) &
queue0=$!
(
  train_and_eval cuda:1 transistor transistor_v2 "$RUNS/v07/reconstructors/transistor/supports.npz"
  train_and_eval cuda:1 screw screw_v2 ""
  train_and_eval cuda:1 carpet carpet_v2 ""
) &
queue1=$!

wait "$queue0"; rc0=$?
wait "$queue1"; rc1=$?
log "all queues finished gpu0=${rc0} gpu1=${rc1}"
printf 'finished %s gpu0=%s gpu1=%s\n' "$(date '+%F %T')" "$rc0" "$rc1" >"$DONE"
log "done"
