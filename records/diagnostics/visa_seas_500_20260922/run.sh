#!/usr/bin/env bash
# VisA SeaS per-type 500 pairs: all image_anno types + combined, Amazon 1cls good.
# Does not overwrite official-500/1000, seed-20260913, defect, or extra-A mvtec_layout.
set -euo pipefail

PY="/home/dell/2T/lm/new_projects/category_anomaly_diffusion/.conda/envs/cad-v0/bin/python"
KIT="/home/dell/2T/lm/new_projects/category_anomaly_diffusion/eval_kit/evaluate.py"
MAIN="/home/dell/2T/lm/new_projects/category_anomaly_diffusion"
REPO="/home/dell/.codex/worktrees/370b/main-design-v0.6"
RUNS="${MAIN}/runs"
VISA="${MAIN}/data/VisA"
STAMP="20260922-seas-500"
DIAG="${RUNS}/v06_c/diagnostics/visa_seas_500_20260922"
CATS=(candle capsules cashew chewinggum)

cd "${REPO}"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
mkdir -p "${DIAG}"

echo "=== MANIFEST + LAYOUT $(date -Is) ==="
"${PY}" - <<'PY'
from pathlib import Path
from cadiff.v06_c.visa_pilot import SOURCE_ROOT, load_visa_seas_inventory, stage_visa_eval_layout, write_visa_manifest

runs = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs")
for cat in ("candle", "capsules", "cashew", "chewinggum"):
    inventory = load_visa_seas_inventory(SOURCE_ROOT, cat)
    manifest = runs / "v06_b" / "visa" / cat / "splits" / "20260922-seas" / "manifest.jsonl"
    if manifest.exists():
        print("skip manifest", cat)
    else:
        write_visa_manifest(inventory, manifest)
        print("wrote", manifest, "fewshot", sum(len(item.fewshot) for item in inventory.types), "types", [item.slug for item in inventory.types])
    layout = runs / "v06_c" / "visa" / cat / "mvtec_layout_seas_20260922"
    if (layout / cat / "train" / "good").is_dir():
        print("skip layout", cat)
    else:
        stage_visa_eval_layout(inventory, layout)
        print("layout", layout, "train", len(inventory.normals), "test_good", len(inventory.test_normals or ()))
PY

for CAT in "${CATS[@]}"; do
  R_OUT="${RUNS}/v06_b/visa/${CAT}/exact-mask-r/seed-20260922-seas"
  if [[ -f "${R_OUT}/summary.json" ]]; then
    echo "[skip train] ${CAT}"
  else
    echo "=== TRAIN ${CAT} $(date -Is) ==="
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 "${PY}" -m cadiff.cli v06-c-visa-train-r \
      --paths configs/paths/omen-home.yaml \
      --manifest "${RUNS}/v06_b/visa/${CAT}/splits/20260922-seas/manifest.jsonl" \
      --dataset-root "${VISA}" \
      --output "${R_OUT}" \
      --category "${CAT}" \
      --split seas \
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
      --checkpoint "${RUNS}/v06_b/visa/${CAT}/exact-mask-r/seed-20260922-seas/checkpoint" \
      --output "${GEN}" \
      --repository-root "${REPO}" \
      --pairs-per-type 500 \
      --split seas \
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
stamp = "20260922-seas-500"
for cat in ("candle", "capsules", "cashew", "chewinggum"):
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
  NAMES="$("${PY}" - <<PY
from cadiff.v06_c.visa_pilot import SOURCE_ROOT, load_visa_seas_inventory
print(" ".join(item.slug for item in load_visa_seas_inventory(SOURCE_ROOT, "${CAT}").types))
PY
)"
  echo "=== EVAL ${CAT} ${NAMES} $(date -Is) ==="
  "${PY}" "${KIT}" \
    --class_name "${CAT}" \
    --anomaly_names ${NAMES} \
    --metrics is ic_lpips localization classification \
    --generated_root "${OUT}/generated_layout" \
    --mvtec_path "${RUNS}/v06_c/visa/${CAT}/mvtec_layout_seas_20260922" \
    --output_dir "${OUT}" \
    --img_size 256 \
    --loc_epochs 200 \
    --cls_epochs 30 \
    --loc_bs 16 \
    --loc_accum 1
  echo "=== EVAL ${CAT} DONE $(date -Is) ==="
done

"${PY}" - <<'PY'
import json
from pathlib import Path

runs = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c")
out = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c/diagnostics/visa_seas_500_20260922/eval256_compare.json")
arms = {}
generate = {}
for cat in ("candle", "capsules", "cashew", "chewinggum"):
    gen = json.loads((runs / "visa" / cat / "generate" / "20260922-seas-500" / "summary.json").read_text())
    generate[cat] = {
        "kept": gen.get("kept_rows"),
        "failed": gen.get("failed_rows"),
        "floor": gen.get("floor_rows"),
        "identity_count": gen.get("identity_count"),
    }
    results = sorted((runs / "visa" / cat / "eval" / "20260922-seas-500" / cat).glob("results_*.json"))
    payload = json.loads(results[-1].read_text()) if results else {}
    loc = payload.get("localization") or {}
    arms[cat] = {
        "IS": (payload.get("is") or {}).get("IS") or payload.get("IS"),
        "IC_LPIPS": (payload.get("ic_lpips") or {}).get("IC_LPIPS") or payload.get("IC_LPIPS"),
        "loc_auroc_image": loc.get("auroc_image") or loc.get("AUROC-I"),
        "loc_ap_image": loc.get("ap_image") or loc.get("AP-I"),
        "loc_f1_image": loc.get("f1_image") or loc.get("F1-I"),
        "loc_auroc_pixel": loc.get("auroc_pixel") or loc.get("AUROC-P"),
        "loc_ap_pixel": loc.get("ap_pixel") or loc.get("AP-P"),
        "loc_f1_pixel": loc.get("f1_pixel") or loc.get("F1-P"),
        "loc_pro_pixel": loc.get("pro_pixel") or loc.get("AU-PRO"),
        "cls_accuracy": (payload.get("classification") or {}).get("accuracy") or payload.get("cls_accuracy"),
        "results": str(results[-1]) if results else None,
    }
out.write_text(json.dumps({
    "schema_version": "v06_c_visa_seas500_eval256_compare_v1",
    "split_id": "visa_seas_type_amazon1cls_v1",
    "not_module_c": True,
    "not_official": True,
    "generate": generate,
    "arms": arms,
}, indent=2, sort_keys=True) + "\n")
print("wrote", out)
PY

date -Is > "${DIAG}/ALL_EVAL_DONE"
echo "ALL_EVAL_DONE $(cat "${DIAG}/ALL_EVAL_DONE")"
