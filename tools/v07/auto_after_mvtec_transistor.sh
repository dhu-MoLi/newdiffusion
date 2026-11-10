#!/usr/bin/env bash
# Wait for the running MVTec baseline drivers to exit, then train the
# transistor reconstructor on the newly freed GPU1. The bottle run already
# occupies GPU0.
set -u

REPO="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型"
RUNS="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs"
PY="/home/asus/anaconda3/envs/cad-v0/bin/python"
LOG="$RUNS/v07/reconstructors/transistor/auto_transistor.log"
TRAIN_LOG="$RUNS/v07/reconstructors/transistor/transistor_v1.log"
DONE="$RUNS/v07/reconstructors/transistor/auto_transistor.done"

mkdir -p "$RUNS/v07/reconstructors/transistor"
exec >>"$LOG" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [ -f "$DONE" ]; then
  log "done marker exists; refusing to run twice"
  exit 0
fi

log "watcher started (pid $$); baseline pids: $(pgrep -f '[r]un_mvtec_baseline.py' | tr '\n' ' ')"
deadline=$(( $(date +%s) + 24 * 3600 ))
while pgrep -f '[r]un_mvtec_baseline.py' >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "timed out waiting for the baseline"
    exit 1
  fi
  sleep 60
done
log "baseline exited; waiting 60s for files to flush"
sleep 60

cd "$REPO" || exit 1
export PYTHONPATH=src
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
log "start transistor reconstructor on cuda:1"
"$PY" tools/v07/train_reconstructor.py \
  --category transistor \
  --device cuda:1 \
  --name transistor_v1 \
  --steps 6000 \
  >"$TRAIN_LOG" 2>&1
rc=$?
log "transistor training finished rc=${rc}"
printf 'finished %s rc=%s\n' "$(date '+%F %T')" "$rc" >"$DONE"
log "watcher done"
