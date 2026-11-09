# VisA 十二类生成基线（可复现包）

这是 2026-09 那次 VisA 全 12 类生成测试用过的代码、R 权重、划分和评估入口。迁到这台机器之后，`eval_inputs/` 里的图是普通文件。同级目录 `runs/` 是空的，生成图不在那里。

方法名：`tokencut_a2_site_scale_place_then_independent_visa_r`。A 是 TokenCut 物体支撑，H 是 `instantiate_site_scale`（不是学出来的 planner），R 是每类独立的 exact-mask renderer。当时自标 `diagnostic_only` / `not_official_generate` / `not_learned_h`。

## 十二类和用的权重

| 类 | 划分 | R 权重 |
| --- | --- | --- |
| candle, capsules, cashew, chewinggum | SeaS，500 对/类 | `checkpoints/v06_b/visa/<类>/exact-mask-r/seed-20260922-seas` |
| fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum | MAGIC 单标签，500 对/类 | `checkpoints/v06_b/visa/<类>/exact-mask-r/seed-20260927-magic` |

正式表丢掉了 SeaS 四类里的 `combined`。更早的 official-200 / official-500、`seed-20260913`、`seed-20260922-defect`，以及 10 月的单类诊断，都不在这个包里。

十二类 pixel AP 表：`records/diagnostics/visa_regime_visibility_20261010/summary.json`。SeaS 四类按缺陷类型拆开的 pixel AP 只在 `records/diagnostics/visa_seas_pixel_ap_buckets_20260924/`。后八类没有按类型拆开的检测结果。

## 这台机器

- 项目：`/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型`
- Python：`/home/asus/anaconda3/envs/cad-v0/bin/python`（3.11。训练依赖按 `requirements/lock.omen-home.txt`，其中 PyTorch 是 `2.6.0+cu124`。下游检测另外装了 `opencv-python-headless` 和 `scikit-learn`，锁文件里没有这两项）
- VisA：`/mnt/C4400D13400D0DBC/lm/code/newproject/data/VisA`（manifest 里的相对路径从这里起，例如 `cashew/Data/Images/Anomaly/001.JPG`）
- Stable Diffusion 1.5：`/mnt/C4400D13400D0DBC/lm/code/newproject/pretrained/stable-diffusion-v1-5-full`（`configs/paths/asus-data.yaml`）
- TokenCut 的 DINOv2：`~/.cache/torch/hub/facebookresearch_dinov2_main`，权重 `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`，模型名 `dinov2_vits14`
- 已有生成图和真实图布局：`eval_inputs/<类>/`

迁出机器的路径还留在 `configs/paths/omen-home.yaml`，这台机器上不存在：Python 曾是 `/home/dell/2T/lm/new_projects/category_anomaly_diffusion/.conda/envs/cad-v0/bin/python`，VisA 曾是 `/home/dell/2T/数据集/AD/VisA`，SD 1.5 曾是 `/home/dell/2T/lm/DualAnoDiff-main/stable-diffusion-v1-5-full`。

在本目录跑代码前：

```bash
cd /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型
export PYTHONPATH=src
PY=/home/asus/anaconda3/envs/cad-v0/bin/python
```

## 重训 R

训练拒绝写入已存在的输出目录。下面的 `--output` 必须是空的新目录，不要指回 `checkpoints/`。

SeaS 四类把 `--split seas`，manifest 用 `splits/v06_b/visa/<类>/20260922-seas/manifest.jsonl`。MAGIC 八类把 `--split magic`，manifest 用 `20260927-magic`。

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
"$PY" -m cadiff.cli v06-c-visa-train-r \
  --paths configs/paths/asus-data.yaml \
  --manifest splits/v06_b/visa/cashew/20260922-seas/manifest.jsonl \
  --dataset-root /mnt/C4400D13400D0DBC/lm/code/newproject/data/VisA \
  --category cashew \
  --split seas \
  --output /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs/v06_c/visa/cashew/exact-mask-r/retrain
```

训练读完整异常 RGB，VAE 编码后再在 latent 里按真值 mask 填高斯噪声，只遮条件，不遮监督目标。

## 重新生成

```bash
"$PY" -m cadiff.cli v06-c-visa-place-bulk \
  --paths configs/paths/asus-data.yaml \
  --source-root /mnt/C4400D13400D0DBC/lm/code/newproject/data/VisA \
  --category cashew \
  --split seas \
  --checkpoint checkpoints/v06_b/visa/cashew/exact-mask-r/seed-20260922-seas/checkpoint \
  --pairs-per-type 500 \
  --output /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs/v06_c/visa/cashew/generate/rerun
```

MAGIC 类把 `--split` 改成 `magic`，checkpoint 改成 `seed-20260927-magic/checkpoint`。`--checkpoint` 要指到里面有 `metadata.json` 的那一层，不是 seed 目录本身。`--output` 同样不能是已有目录。

## 重跑下游检测

检测器是 DualAnoDiff，入口 `eval_kit/evaluate.py`。真实图布局和生成图布局都已经按 MVTec 目录摆好。

- 真实图：`eval_inputs/<类>/mvtec_layout_seas_20260922` 或 `mvtec_layout_magic_20260927`。这个目录的下一级才是类名。
- 生成图：`eval_inputs/<类>/generated_layout_20260922-seas-500/generated_layout` 或 `generated_layout_20260927-magic-500/generated_layout`。多一层 `generated_layout` 是复制时 rsync 留下的，里面才是 `<类>/<缺陷>/composed`。

`--metrics` 只接受 `localization`、`classification`，以及生成质量用的 `is`、`ic_lpips`。正式表用前两个。`localization` 内部报告 AUROC-I、AUROC-P、AP-I、AP-P、F1-I、F1-P、PRO，并在每个 epoch 按 AUROC-I + AUROC-P + AP-P + PRO 留 checkpoint。

```bash
"$PY" eval_kit/evaluate.py \
  --class_name cashew \
  --anomaly_names burnt combined corner_or_edge_breakage different_colour_spot middle_breakage same_colour_spot small_holes small_scratches stuck_together \
  --eval_anomaly_names burnt corner_or_edge_breakage different_colour_spot middle_breakage same_colour_spot small_holes small_scratches stuck_together \
  --metrics localization classification \
  --generated_root eval_inputs/cashew/generated_layout_20260922-seas-500/generated_layout \
  --mvtec_path eval_inputs/cashew/mvtec_layout_seas_20260922 \
  --image_subdir composed --mask_subdir mask_defect \
  --loc_epochs 200 --cls_epochs 30 --img_size 256 --loc_bs 16 --loc_accum 1 \
  --output_dir /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs/eval/cashew
```

正式表的口径：单标签、Amazon `1cls`、每类 `floor(n/3)`、去掉 `combined`、256、每个 epoch 按 AUROC-I + AUROC-P + AP-P + PRO 选 checkpoint。这是看过测试集的开发，不是未见测试。MAGIC 类没有 `combined`，`--anomaly_names` 和 `--eval_anomaly_names` 用该类 `test/` 下的类型名。

原结果数字在 `records/<类>/results_*.json`，以及 `records/diagnostics/`。
