# 类别专用重建器（Self-supervised normal completion）

> 目标：用正常图 + B0 缺陷形状先验自监督地训练一个类别专用补全器，替代 A2 里的逐像素最近邻填充。
> 代码：`src/cadiff/v07/reconstructor.py`、`tools/v07/train_reconstructor.py`

## 1. 训练数据

- 来源：MVTec / VisA 的 `train/good`，不读取任何缺陷图。
- 洞：`ParametricShapePrior`（B0）从 few-shot GT mask 估计形状/面积/弹性/多组件分布，采样后放置到正常图的 TokenCut 物体支撑内。
- 监督：同一张正常图在洞内的原始像素（以及 DINOv2 特征）。
- 训练/验证：按正常图随机 9:1 划分；验证洞固定 seed。

## 2. 损失

```
L = masked_L1(pred, target) inside hole
  + 0.05 * gradient_L1(composite, target)
  + 0.1  * (1 - cos(DINOv2(composite)_hole, DINOv2(target)_hole))   every 10 steps
```

- DINOv2 特征项是核心：A2 的 prototype 就是洞内 token 均值，所以重建器要优化"洞内特征像正常材料"，而不只是像素。
- 特征项只在小 batch（默认 2）上每 10 步算一次，控制显存。
- 模型输出在洞外被原始像素覆盖，保证不影响上下文。

## 3. 模型

`MaskedUNet`：4 通道输入（RGB + hole mask），3 通道输出，U-Net（默认 base 48、depth 4，约 20M 参数），GroupNorm + SiLU，`tanh` 输出。

## 4. 用法

```bash
cd /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型
export PYTHONPATH=src
PY=/home/asus/anaconda3/envs/cad-v0/bin/python

"$PY" tools/v07/train_reconstructor.py \
  --category bottle \
  --device cuda:0 \
  --name bottle_v1 \
  --steps 6000 \
  --supports-cache /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs/v07/reconstructors/bottle/supports.npz
```

- 第一次会计算并缓存正常图的 TokenCut 支撑（`supports.npz`，约几分钟）；后续实验复用。
- 输出：`runs/v07/reconstructors/<category>/<name>/`
  - `config.json`、`run_meta.json`
  - `model/checkpoint.pt`（验证最优）
  - `model/final.pt`
  - `model/summary.json`（含 val 指标、baseline 对照、history）

## 5. 验证指标

| 指标 | 含义 |
|---|---|
| `masked_l1` | 洞内 L1（越低越好） |
| `psnr` | 洞内 PSNR（越高越好） |
| `feature_cos` | 洞内 DINOv2 token 余弦（越高越好；直接对应 A2 prototype） |

同一组验证洞上同时跑 `neighbor` 和 `telea` 两个零样本基线。

## 6. 当前基线（bottle 验证集，21 对）

| 方法 | masked_l1 | PSNR | feature_cos |
|---|---:|---:|---:|
| neighbor（A2 现状） | 0.00056 | 23.73 | 0.605 |
| telea | 0.00058 | 24.09 | 0.689 |

> 注意：masked_l1 在平滑区域会非常小，主要看 `feature_cos` 和后续 A2 proposal 指标。

## 7. 与 A2 / 诊断的衔接

- `LearnedReconstructor` 已实现，可直接作为 `v07/reconstruction.py` 的后端使用。
- 训练完成后，用 `LearnedReconstructor(checkpoint)` 替换 A2 的填充函数，再跑 `tools/v07/diag_proposal.py`，对比：
  - `fill_feature_cos`
  - `proposal_mean_similarity` / `similarity_gap`
  - `site_*` 位置指标
- 如果 proposal 指标提升，再进入 mini-protocol 下游。

## 8. 注意事项

- **按类型分层评估**：Stage 0 显示重建收益在 `bottle/broken_large`、`transistor` 这类大块/结构化缺陷上最明显；纹理类可能收益有限。
- B0 的洞分布决定了训练分布；B1 学习式 mask 生成器之后可以重训 v2。
- 如果显存不够：降低 `--base-channels` 到 32 或 `--batch-size` 到 2；特征项 batch 已单独限制。
- 验证时不要用逐像素 `neighbor_fill_source`（O(N²) 很慢）；训练脚本内部已改用 scipy 线性时间最近邻。
