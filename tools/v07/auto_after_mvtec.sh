#!/usr/bin/env bash
# Wait for the running MVTec baseline drivers to exit, then start the first
# v07 screening sweep. Safe to launch with setsid/nohup; all output goes to
# the runs directory, never into git.
set -u

REPO="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型"
RUNS="/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs"
BASELINE="$RUNS/mvtec_baseline/20261108"
OUT="$RUNS/v07"
PY="/home/asus/anaconda3/envs/cad-v0/bin/python"
LOG="$OUT/auto_after_mvtec.log"
DONE="$OUT/auto_after_mvtec.done"
CATEGORIES=(bottle carpet grid transistor)

mkdir -p "$OUT"
exec >>"$LOG" 2>&1

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [ -f "$DONE" ]; then
  log "done marker already exists; refusing to run twice"
  exit 0
fi

log "watcher started (pid $$); baseline pids: $(pgrep -f '[r]un_mvtec_baseline.py' | tr '\n' ' ')"
deadline=$(( $(date +%s) + 48 * 3600 ))
while pgrep -f '[r]un_mvtec_baseline.py' >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "timed out after 48h waiting for the baseline; exiting"
    exit 1
  fi
  sleep 60
done
log "baseline processes exited"
sleep 60

metrics_count=$(find "$BASELINE" -maxdepth 2 -name baseline_metrics.json 2>/dev/null | wc -l)
log "baseline metrics present: ${metrics_count}/15"

cd "$REPO" || exit 1
export PYTHONPATH=src
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_experiment() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  log "start ${name} on ${gpu}"
  "$PY" tools/v07/run_experiment.py \
    --config "$config" \
    --categories "${CATEGORIES[@]}" \
    --device "$gpu" \
    >"$OUT/${name}.driver.log" 2>&1
  local rc=$?
  log "finish ${name} on ${gpu} rc=${rc}"
  return "$rc"
}

# Two queues, one per GPU. A failure in one experiment does not stop the queue.
(
  run_experiment cuda:0 mini_baseline configs/experiments/v07/mini_baseline.yaml
  run_experiment cuda:0 mini_vis_rgb configs/experiments/v07/mini_vis_rgb.yaml
) &
queue0=$!
(
  run_experiment cuda:1 mini_vis_latent configs/experiments/v07/mini_vis_latent.yaml
  run_experiment cuda:1 mini_shape configs/experiments/v07/mini_shape.yaml
) &
queue1=$!

wait "$queue0"; rc0=$?
wait "$queue1"; rc1=$?

log "queues finished: gpu0=${rc0} gpu1=${rc1}"
for experiment in mini_baseline mini_vis_latent mini_vis_rgb mini_shape; do
  "$PY" tools/v07/aggregate.py \
    --root "$OUT" \
    --experiment "$experiment" \
    --output "$OUT/aggregate_${experiment}.md" \
    >>"$LOG" 2>&1
  log "aggregate ${experiment} rc=$?"
done

printf 'finished %s gpu0=%s gpu1=%s\n' "$(date '+%F %T')" "$rc0" "$rc1" >"$DONE"
log "watcher done"
