# AnomalyDiffusion（VisA SeaS，剔除 combined）权重清单

> 复核时间：2026-11-17。全部路径已逐一验证存在；结构化版本见 `configs/ad/visa_seas_weights.json`。
> 约定：推理与下游评测只使用下列最终文件；各目录下的 `embeddings_gs-*.pt`、`last.ckpt` 等中途存档不可用。

## 1. 全局权重（12 类共用）

| 用途 | 路径 | 大小 |
| --- | --- | ---: |
| 预训练 LDM 主干 | `anomalydiffusion-portable/models/ldm/text2img-large/model.ckpt` | 5.7 GB |
| train_anomaly spatial encoder | `experiments/visa_seas/logs/anomaly-checkpoints/checkpoints/spatial_encoder.pt` | 1.4 GB |
| train_anomaly embeddings | `experiments/visa_seas/logs/anomaly-checkpoints/checkpoints/embeddings.pt` | 1.4 MB |

## 2. train_mask：58 个（类, 缺陷类型）最终 embedding

前缀：`experiments/visa_seas/logs/mask-checkpoints/`，每个缺陷对一份 `checkpoints/embeddings.pt`。

- 全部 **58/58 存在**，与本文档的类型表一一对应
- 已排除 8 个 `*-combined` 目录（candle/capsules/cashew/chewinggum/fryum/macaroni1/macaroni2/pcb1）
- 类型名与仓库 split 完全一致：`splits/v06_b/visa/<class>/<split>/manifest.jsonl` 的 `defect_type` 归一化后无差异（split 里额外的 `combined`、`good` 符合预期）
  - SeaS 四类：`20260922-seas`；其余八类：`20260927-magic`

## 3. 下游检测器（per-category）

- 定位：`experiments/visa_seas/eval/eval_kit/per_category/<class>/ckpt_localization/<class>.pckl` —— **12/12 存在**
- 分类：`.../ckpt_classification/<class>.pckl` —— **11/12 存在（capsules 没有分类权重）**

## 4. 按类型生成的 mask 与质量

目录：`experiments/visa_seas/outputs/generated_masks/<class>/<type>/`，每类型 500 张 256×256 JPG。

质量差异极大（AD 官方也报告过大量空 mask），筛选标准建议：**面积 ≥0.2% 图像且非空的有效 mask ≥150 张**。满足的类型：

| 类别 | 可用类型（有效数） |
| --- | --- |
| candle | damaged_corner_of_packaging (376)、extra_wax_in_candle (209)、wax_melded_out_of_the_candle (417) |
| capsules | 无（仅 bubble，有效极少） |
| cashew | corner_or_edge_breakage (174)、middle_breakage (500)、stuck_together (453) |
| chewinggum | chunk_of_gum_missing (498)、corner_missing (356)、similar_colour_spot (190) |
| fryum | burnt (377)、fryum_stuck_together (500)、similar_colour_spot (402) |
| pcb1 | missing (488) |
| pcb2 | melt (388)、missing (485) |
| pcb3 | bent (346)、melt (213)、missing (383) |
| pcb4 | extra (461)、missing (476) |
| pipe_fryum | corner_and_edge_breakage (288)、stuck_together (438) |

共 **22 个可用缺陷对**。

## 5. 与之前 v08 阶段 B 的关系（重要）

v08 阶段 B 用的是 **per-class "ko"** 权重（`experiments/visa/logs/mask-checkpoints/<class>-ko` 与 `<class>/ko/` 下的 mask），训练时把类型混合成一个标签，导致大量退化 mask（capsules 98.5% 面积 <0.2%），A/B 对比被数据质量问题主导。

**应改用本文件列出的 per-type 权重/mask 重做**：
- 每类按类型采样，R 的类型条件与实际类型一致（不再循环）
- 只使用第 4 节的可用类型
- A/B 仍按索引严格对应，Ω 约束组仅在放不下时重采样并记录

## 6. 磁盘与备份

- `anomalydiffusion-portable` 总体约 **100 GB**
- 其中 `visa_seas/logs/mask-checkpoints` 约 **2.2 GB**（58 份 embedding）
- 如需迁移/备份，最小可用集合：LDM 主干 + spatial_encoder + 58 份 embeddings + `visa_seas/outputs/generated_masks`（124 MB）+ per-category 下游权重

## 7. 项目内最小复现包（2026-11-17 建立）

位置：`/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/ad_repro/`（约 13 G）
说明：`ad_repro/README.md`；机器可读清单：`ad_repro/manifest.json`（路径已重映射到新位置）。

只包含对比需要的内容，**不包含原项目的其他实验**：

| 内容 | 大小 | 校验 |
| --- | ---: | --- |
| `code/`（generate_mask.py、train_mask.py、ldm/、configs/ 等） | 1.9 M | 复制 |
| `models/ldm/text2img-large/model.ckpt` | 5.8 G | 复制 |
| `logs/anomaly-checkpoints/checkpoints/`（spatial_encoder + embeddings） | 1.4 G | 复制 |
| `logs/mask-checkpoints/<class>-<type>/checkpoints/embeddings.pt` | ~5 M | **58/58** |
| `outputs/generated_masks/` | 124 M | **29,000 张，与源一致** |
| `eval/per_category/`（12 loc + 11 cls） | 2.2 G | 23 个 pckl |
| `dataset/visa/`（MVTec 风格） | 1.9 G | **24,055 文件，与源一致** |
| `dataset/visa_seas/`（split 数据集） | 1.8 G | **14,337 文件，与源一致**；`inventory.json`、`fewshot_train/` 58 份、类型目录链接均保留 |

数据划分与原复现**完全一致**：`inventory.json`（74 对 / 58 可训练 / combined 排除 / 每对 500 mask）、`fewshot_train/*.txt`、`<type>` 与 `<type>_mask` 相对链接。
