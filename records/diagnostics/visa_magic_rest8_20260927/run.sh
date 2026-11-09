#!/usr/bin/env bash
# Remaining 8 VisA classes, MAGIC single-label only.
# New R, 500 pairs on generate=yes types, floor(n/3)=0 stays in the localization test.
# Does not overwrite SeaS-500, official-500/1000, defect, or 20260926-magic-single.
set -euo pipefail

PY="/home/dell/2T/lm/new_projects/category_anomaly_diffusion/.conda/envs/cad-v0/bin/python"
KIT="/home/dell/2T/lm/new_projects/category_anomaly_diffusion/eval_kit/evaluate.py"
MAIN="/home/dell/2T/lm/new_projects/category_anomaly_diffusion"
REPO="/home/dell/.codex/worktrees/370b/main-design-v0.6"
RUNS="${MAIN}/runs"
VISA="${MAIN}/data/VisA"
STAMP="20260927-magic-500"
DIAG="${RUNS}/v06_c/diagnostics/visa_magic_rest8_20260927"
CATS=(fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)

cd "${REPO}"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
mkdir -p "${DIAG}"

echo "=== MANIFEST + LAYOUT $(date -Is) ==="
"${PY}" - <<'PY'
from pathlib import Path
from cadiff.v06_c.visa_pilot import (
    SOURCE_ROOT,
    load_visa_magic_inventory,
    stage_visa_eval_layout,
    write_visa_manifest,
)

runs = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs")
cats = ("fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
for cat in cats:
    generated = load_visa_magic_inventory(SOURCE_ROOT, cat, generate_only=True)
    full = load_visa_magic_inventory(SOURCE_ROOT, cat, generate_only=False)
    manifest = runs / "v06_b" / "visa" / cat / "splits" / "20260927-magic" / "manifest.jsonl"
    if manifest.exists():
        print("skip manifest", cat)
    else:
        write_visa_manifest(generated, manifest)
        print(
            "wrote", manifest,
            "fewshot", sum(len(item.fewshot) for item in generated.types),
            "types", [item.slug for item in generated.types],
        )
    layout = runs / "v06_c" / "visa" / cat / "mvtec_layout_magic_20260927"
    if (layout / cat / "train" / "good").is_dir():
        print("skip layout", cat)
    else:
        stage_visa_eval_layout(full, layout)
        print(
            "layout", layout,
            "train", len(full.normals),
            "test_good", len(full.test_normals or ()),
            "test_types", [item.slug for item in full.types],
        )
PY

for CAT in "${CATS[@]}"; do
  R_OUT="${RUNS}/v06_b/visa/${CAT}/exact-mask-r/seed-20260927-magic"
  if [[ -f "${R_OUT}/summary.json" ]]; then
    echo "[skip train] ${CAT}"
  else
    echo "=== TRAIN ${CAT} $(date -Is) ==="
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 "${PY}" -m cadiff.cli v06-c-visa-train-r \
      --paths configs/paths/omen-home.yaml \
      --manifest "${RUNS}/v06_b/visa/${CAT}/splits/20260927-magic/manifest.jsonl" \
      --dataset-root "${VISA}" \
      --output "${R_OUT}" \
      --category "${CAT}" \
      --split magic \
      --repository-root "${REPO}" \
      --device cuda:0
    echo "=== TRAIN ${CAT} DONE $(date -Is) ==="
  fi
done
date -Is > "${DIAG}/ALL_TRAIN_DONE"

for CAT in "${CATS[@]}"; do
  GEN="${RUNS}/v06_c/visa/${CAT}/generate/${STAMP}"
  if [[ -f "${GEN}/summary.json" ]]; then
    echo "[skip generate] ${CAT}"
  else
    echo "=== GENERATE ${CAT} $(date -Is) ==="
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONPATH=src:. PYTHONUNBUFFERED=1 \
    "${PY}" -m cadiff.cli v06-c-visa-place-bulk \
      --paths configs/paths/omen-home.yaml \
      --source-root "${VISA}" \
      --category "${CAT}" \
      --checkpoint "${RUNS}/v06_b/visa/${CAT}/exact-mask-r/seed-20260927-magic/checkpoint" \
      --output "${GEN}" \
      --repository-root "${REPO}" \
      --pairs-per-type 500 \
      --split magic \
      --device cuda:0
    echo "=== GENERATE ${CAT} DONE $(date -Is) ==="
  fi
done
date -Is > "${DIAG}/ALL_GENERATE_DONE"

echo "=== PACK $(date -Is) ==="
"${PY}" - <<'PY'
from pathlib import Path
from cadiff.v06_c.visa_place_bulk import pack_visa_eval_generated
runs = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c")
stamp = "20260927-magic-500"
for cat in ("fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"):
    pack_visa_eval_generated(
        runs / "visa" / cat / "generate" / stamp,
        runs / "visa" / cat / "eval" / stamp / "generated_layout",
        cat,
    )
    print("packed", cat)
PY

for CAT in "${CATS[@]}"; do
  OUT="${RUNS}/v06_c/visa/${CAT}/eval/${STAMP}"
  RESULTS="$(ls -1 "${OUT}/${CAT}"/results_*.json 2>/dev/null || true)"
  if [[ -n "${RESULTS}" ]]; then
    echo "[skip eval] ${CAT} ${RESULTS}"
    continue
  fi
  NAME_LINE="$("${PY}" - <<PY
from pathlib import Path
from cadiff.v06_c.visa_pilot import SOURCE_ROOT, load_visa_magic_inventory
gen = load_visa_magic_inventory(SOURCE_ROOT, "${CAT}", generate_only=True)
full = load_visa_magic_inventory(SOURCE_ROOT, "${CAT}", generate_only=False)
layout = Path("${OUT}/generated_layout/${CAT}")
trainable = []
for item in gen.types:
    composed = layout / item.slug / "composed"
    if composed.is_dir() and any(composed.iterdir()):
        trainable.append(item.slug)
if not trainable:
    raise SystemExit("no generated pairs to train ${CAT}")
print("|".join((
    " ".join(trainable),
    " ".join(item.slug for item in full.types),
)))
PY
)"
  GEN_NAMES="${NAME_LINE%%|*}"
  EVAL_NAMES="${NAME_LINE#*|}"
  echo "=== EVAL ${CAT} train=${GEN_NAMES} test=${EVAL_NAMES} $(date -Is) ==="
  if [[ "${GEN_NAMES}" == "${EVAL_NAMES}" ]]; then
    "${PY}" "${KIT}" \
      --class_name "${CAT}" \
      --anomaly_names ${GEN_NAMES} \
      --metrics localization classification \
      --generated_root "${OUT}/generated_layout" \
      --mvtec_path "${RUNS}/v06_c/visa/${CAT}/mvtec_layout_magic_20260927" \
      --output_dir "${OUT}" \
      --img_size 256 \
      --loc_epochs 200 \
      --cls_epochs 30 \
      --loc_bs 16 \
      --loc_accum 1
  else
    "${PY}" "${KIT}" \
      --class_name "${CAT}" \
      --anomaly_names ${GEN_NAMES} \
      --eval_anomaly_names ${EVAL_NAMES} \
      --metrics localization classification \
      --generated_root "${OUT}/generated_layout" \
      --mvtec_path "${RUNS}/v06_c/visa/${CAT}/mvtec_layout_magic_20260927" \
      --output_dir "${OUT}" \
      --img_size 256 \
      --loc_epochs 200 \
      --cls_epochs 30 \
      --loc_bs 16 \
      --loc_accum 1
  fi
  echo "=== EVAL ${CAT} DONE $(date -Is) ==="
done

"${PY}" - <<'PY'
import json
from pathlib import Path

runs = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c")
out = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c/diagnostics/visa_magic_rest8_20260927/eval256_compare.json")
cats = ("fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
stamp = "20260927-magic-500"
arms = {}
generate = {}
for cat in cats:
    gen = json.loads((runs / "visa" / cat / "generate" / stamp / "summary.json").read_text())
    generate[cat] = {
        "kept": gen.get("kept_rows"),
        "failed": gen.get("failed_rows"),
        "floor": gen.get("floor_rows"),
        "identity_count": gen.get("identity_count"),
    }
    results = sorted((runs / "visa" / cat / "eval" / stamp / cat).glob("results_*.json"))
    payload = json.loads(results[-1].read_text()) if results else {}
    loc = payload.get("localization") or {}
    arms[cat] = {
        "loc_auroc_image": loc.get("auroc_image") or payload.get("loc_auroc_image"),
        "loc_ap_image": loc.get("ap_image") or payload.get("loc_ap_image"),
        "loc_f1_image": loc.get("f1_image") or payload.get("loc_f1_image"),
        "loc_auroc_pixel": loc.get("auroc_pixel") or payload.get("loc_auroc_pixel"),
        "loc_ap_pixel": loc.get("ap_pixel") or payload.get("loc_ap_pixel"),
        "loc_f1_pixel": loc.get("f1_pixel") or payload.get("loc_f1_pixel"),
        "loc_pro_pixel": loc.get("pro_pixel") or payload.get("loc_pro_pixel"),
        "cls_accuracy": payload.get("cls_accuracy"),
        "results": str(results[-1]) if results else None,
    }
out.write_text(json.dumps({
    "schema_version": "v06_c_visa_magic_rest8_eval256_compare_v1",
    "split_id": "visa_magic_single_anomaly_amazon1cls_v1",
    "not_module_c": True,
    "not_official": True,
    "generate": generate,
    "arms": arms,
}, indent=2, sort_keys=True) + "\n")
print("wrote", out)
PY

date -Is > "${DIAG}/ALL_EVAL_DONE"
echo "ALL_EVAL_DONE $(cat "${DIAG}/ALL_EVAL_DONE")"
