# H 站点与位置先验（L0/L1）

> 目标：解决 Stage 0 发现的 `site_proposal_ratio = 1.0`、`site_count = 1`、`site/支撑 ≈ 0.40` 问题。
> 代码：`src/cadiff/v07/site_selection.py`、`v07/location.py`、`v07/placement.py`、`tools/v07/diag_proposal.py`

## 1. 根因

基线 `candidate_tokencut_sites(target_support, proposal)` 切的是 **target support 的连通域**，再与 proposal 求交。大多数物体只有一个连通域，所以：

```
site_count = 1
site ∩ proposal = proposal
site_proposal_ratio = 1.0
site / support ≈ 0.40
```

`pick_tokencut_site` 的随机选择在单连通物体上没有意义。而且 proposal 本身是"材料相似区"，偏向物体内部；真实缺陷常出现在边缘/薄结构（transistor 真实径向 0.79 vs proposal 0.60）。

## 2. L0：proposal 连通域站点

`proposal_sites()` 直接切 **proposal 的 8 连通域**，每个站点记录：

- pixels / mean_similarity
- SDF 均值 / 归一化径向位置

`select_site()` 支持：

- 相似度加权；
- 可选的位置先验分数；
- softmax 温度采样（保留多样性）。

**bottle 初步结果（neighbor/telea，3 probes/backend）**：

| 后端 | site_count | site/proposal | site/支撑 | site_sdf | site_radial |
|---|---:|---:|---:|---:|---:|
| 基线 support_cc | 1.0 | 1.0000 | 0.42 | 88.1 | 0.578 |
| L0 proposal_cc | 4.3–5.7 | 0.33–0.67 | 0.14–0.29 | 86.2–91.3 | 0.557–0.567 |

站点从"整个 proposal"变成局部块，`site/proposal` 明显下降。

## 3. L1：物体坐标系缺陷位置先验

`ObjectLocationPrior`：

1. 对每张 few-shot 缺陷图，用 TokenCut support 建立物体坐标系；
2. 把 GT mask 像素映射到归一化坐标 `u=(x-cx)/sx, v=(y-cy)/sy`；
3. 每个 type 累积 2D 直方图（每张 mask 先归一化再平均），Gaussian 平滑；
4. 生成时对候选站点计算平均 log density，作为 `select_site` 的先验分数。

接口：

- `score_mask(mask, support, defect_type)`
- `score_site(site, support, defect_type)`
- `heatmap(support, defect_type)`（用于可视化）
- `from_inventory(inventory, encoder)`（直接从 few-shot 构建）

**bottle 初步结果（3 probes/backend）**：

| 后端 | site_method | site_prior_score | site_sdf | site_radial |
|---|---|---:|---:|---:|
| neighbor | proposal_cc | n/a | 86.2 | 0.567 |
| neighbor | proposal_cc_prior | -6.51 | 86.2 | 0.567 |
| telea | proposal_cc | n/a | 91.3 | 0.557 |
| telea | proposal_cc_prior | -6.65 | 78.8 | 0.604 |

先验让 telea 的站点 SDF 从 91.3 降到 78.8（更接近真实缺陷的 102？注意真实 SDF 是 102，方向上仍需更大样本确认），径向从 0.557 到 0.604。样本很小（3 probes），只能说明管线通了，不能下结论。

## 4. 下一步（L2/L3）

- **L2 学习式站点排序**：用 DINOv2 上下文 + SDF/径向 + 面积 + 类型 embedding 训练一个小 scorer，正样本为 few-shot 缺陷位置、负样本为随机站点；对比直方图先验。
- **L3 联合位置-形状 B**：条件 mask 生成器，输入目标特征 + type + L1 热图，直接输出局部 mask；这是最强的论文主线。
- **评价**：
  - site_count、site/proposal、site/support；
  - 位置 NLL（held-out 真实缺陷 under few-shot prior）；
  - SDF/径向分布 MMD/KS；
  - 下游 AP-P/PRO；
  - 可视化：真实缺陷热图 vs 生成 site 热图。
- **数据质量**：L0 的 `_safe_scale` 会按站点 bbox 缩小形状，应能缓解 wood/tile 的 `scaled GT does not fit the canvas`。

## 5. 使用

```bash
# L0
"$PY" tools/v07/diag_proposal.py --categories bottle --backends neighbor telea \
  --site-method proposal_cc --normals-per-type 2 --sources-per-type 2 \
  --max-probes-per-type 4 --device cuda:0 --name bottle_site_proposal_cc

# L1
"$PY" tools/v07/diag_proposal.py --categories bottle --backends neighbor telea \
  --site-method proposal_cc_prior --normals-per-type 2 --sources-per-type 2 \
  --max-probes-per-type 4 --device cuda:0 --name bottle_site_prior
```

输出表新增 `site_prior_score`、`site_count`、`site_proposal_ratio`、`site_support_ratio`。
