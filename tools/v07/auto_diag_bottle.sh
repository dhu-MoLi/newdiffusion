#!/usr/bin/env bash
# After the bottle reconstructor finishes, run the Stage-0 diagnostic with the
# learned backend and write a direct comparison against the zero-shot arms.
set -u

REPO="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型"
RUNS="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs"
PY="/home/asus/anaconda3/envs/cad-v0/bin/python"
LOG="$RUNS/v07/diag_proposal/bottle_learned_diag.log"
DONE="$RUNS/v07/diag_proposal/bottle_learned_diag.done"

mkdir -p "$RUNS/v07/diag_proposal"
exec >>"$LOG" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "watcher started (pid $$); bottle training pids: $(pgrep -f 'train_reconstructor.py --category bottle' | tr '\n' ' ')"
while pgrep -f 'train_reconstructor.py --category bottle' >/dev/null 2>&1; do
  sleep 30
done
log "bottle training exited; waiting 20s"
sleep 20

cd "$REPO" || exit 1
export PYTHONPATH=src
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
log "run learned diagnostic on bottle"
"$PY" tools/v07/diag_proposal.py \
  --categories bottle \
  --backends neighbor telea ring learned \
  --learned-root "$RUNS/v07/reconstructors" \
  --normals-per-type 2 --sources-per-type 2 --max-probes-per-type 4 \
  --atlas-references 4 --device cuda:0 --name bottle_learned_diag
rc=$?
log "diagnostic finished rc=${rc}"
printf 'finished %s rc=%s\n' "$(date '+%F %T')" "$rc" >"$DONE"
log "watcher done"
