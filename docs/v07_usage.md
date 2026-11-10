# v07 使用说明

v07 是基线之上的实验层：`src/cadiff/v06` 保持不变，所有新组件在 `src/cadiff/v07`。每个实验的输出写到项目的 `runs/v07/<实验名>/<类别>/seed-<种子>/`（项目同级 `runs/`，不在 git 仓库内），不影响现有 MVTec 基线。

## 1. 一条命令跑完一个实验

```bash
cd /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型
export PYTHONPATH=src
PY=/home/asus/anaconda3/envs/cad-v0/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 先确认哪张卡空闲（MVTec 基线运行期间可能两张卡都在用）
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

"$PY" tools/v07/run_experiment.py \
  --config configs/experiments/v07/mini_baseline.yaml \
  --categories bottle carpet grid transistor \
  --device cuda:0
```

阶段可以单跑或续跑：

```bash
... --stage train
... --stage generate
... --stage pack
... --stage eval
```

已有输出会自动跳过，所以中断后重跑同一条命令即可续上。

## 2. 实验配置

`configs/experiments/v07/` 下每个 YAML 是一个实验：

| 配置 | 内容 |
| --- | --- |
| `mini_baseline.yaml` | 200 对 / 50 epoch，SD1.5，无改动 |
| `mini_vis_latent.yaml` | 潜空间低频可见性损失 |
| `mini_vis_rgb.yaml` | RGB@256 可见性损失 |
| `mini_shape.yaml` | 参数化形状先验（弹性形变 + 多组件） |
| `backbone_sd2_inpaint.yaml` | SD2 inpainting 9 通道骨干 |
| `full_baseline.yaml` | 500 对 / 200 epoch，正式协议 |

关键字段：

- `visibility`: `none` / `latent` / `rgb`
- `shape_prior`: `none` / `parametric`
- `backbone`: `sd15` / `sd2-inpaint`
- `pairs_per_type`、`loc_epochs`、`img_size`、`loc_bs`
- `seed`：默认 20261109

## 3. 输出结构

```
runs/v07/<实验名>/<类别>/seed-<种子>/
├── run_meta.json          # git commit、配置哈希、seed、GPU
├── run.log
├── train/
│   ├── summary.json
│   ├── training.json
│   └── checkpoint/{adapter,conditioner}.safetensors + metadata.json
├── generate/
│   ├── summary.json
│   ├── synthetic_manifest.jsonl
│   └── images/ masks/
├── generated_layout/<类别>/<类型>/composed|mask_defect
├── eval/
├── metrics.json           # 下游指标汇总
└── shape_prior.json       # 仅形状先验实验
```

## 4. 汇总与诊断

```bash
# 跨实验/类别/seed 汇总成 markdown + csv
"$PY" tools/v07/aggregate.py --root runs/v07 --output runs/v07/aggregate.md

# 生成 mask 与参考 GT 的形状分布对比（KS 检验）
"$PY" tools/v07/mask_stats.py \
  --generated runs/v07/mini_shape/bottle/seed-20261109/generate/masks \
  --reference runs/v07/mini_baseline/bottle/seed-20261109/generate/masks \
  --output runs/v07/mask_stats_bottle.json

# Stage-0 候选区诊断：重建后端 × proposal × 站点 × 位置
"$PY" tools/v07/diag_proposal.py \
  --categories bottle carpet grid transistor \
  --backends neighbor telea ns ring atlas \
  --normals-per-type 4 --sources-per-type 2 --max-probes-per-type 8 \
  --device cuda:0 --name stage0_v1
```

诊断的指标定义、使用注意事项（尤其是"不要用缺陷图当 target 算 proposal 召回"）见 `docs/proposal_diagnostic.md`。

## 5. 首次运行前的冒烟测试（不占 GPU）

```bash
"$PY" -m compileall -q src/cadiff/v07 tools/v07
"$PY" tools/v07/smoke_v07.py
```

`smoke_v07.py` 只做张量级检查：可见性损失有限性、形状先验采样、指标聚合，不加载扩散模型。

## 6. 注意

- 不要删除或覆盖 `runs/mvtec_baseline/20261108`，那是当前论文对照的基线。
- 每次正式实验前确认 `git status` 干净；`run_meta.json` 会记录 dirty 标记。
- 论文阶段需把 `loc` 的 checkpoint 选择改成固定 epoch 或独立验证集，当前仍是测试集选点（探索阶段允许）。
