# Stage-0 候选区诊断

> 目标：在训练任何生成模型之前，先量化 A2 入口各环节的损失。
> 代码：`src/cadiff/v07/reconstruction.py`、`src/cadiff/v07/proposal_probe.py`、`tools/v07/diag_proposal.py`

## 1. 为什么需要它

基线 A2 的链路是：

```
源缺陷图 + GT mask → 逐像素最近邻填充 → DINOv2 洞内 token 均值(prototype)
→ 目标正常图相似度 → proposal → TokenCut 站点 → H 放置
```

如果直接换重建模型、或直接换 mask 生成方式，很难判断收益来自哪一层。这个诊断把链路拆成可度量的四段：

1. **填充质量**：补出来的洞像不像周围正常材料；
2. **proposal 材料一致性**：相似度是否真的在 proposal 内更高、区域是否过大；
3. **站点定位**：H 选中的 site 是否还是一个"大块"，而不是局部区域；
4. **位置合理性**：真实缺陷、proposal、site 在物体坐标系中的 SDF/径向位置是否一致。

## 2. 一个必须避免的错误指标

**不要把一张缺陷图当 target，然后看 proposal 是否覆盖它的 GT。**

proposal 的定义是"target 中与**正常材料** prototype 相似的区域"。缺陷本身是异常的，按设计就应该被排除。用缺陷图评估会得出反向结论。

本诊断改用：

- 正常 target 上的 proposal 材料统计；
- 用真实缺陷 mask 的 SDF/径向分布做位置对照；
- 需要像素级召回时，再用"缺陷图配准到正常图"的伪 GT（后续再加）。

## 3. 指标

| 指标 | 含义 | 期望 |
|---|---|---|
| `fill_feature_cos` | 洞内填充 token 均值 vs 洞外正常环的余弦 | 越高越好 |
| `fill_color_l1` | 洞内 vs 正常环的平均颜色差（0–1） | 越低越好 |
| `proposal_support_ratio` | proposal 占物体支撑比例 | 不要接近 1 |
| `proposal_components` | proposal 连通域数 | 有结构更好 |
| `proposal_mean_similarity` | proposal 内平均相似度 | 越高越好 |
| `similarity_gap` | proposal 内 vs 支撑内 proposal 外的相似度差 | 越大越好 |
| `site_support_ratio` | 选中站点占支撑比例 | 越小越"局部" |
| `site_proposal_ratio` | 选中站点占 proposal 比例 | 应明显 < 1 |
| `site_count` | 候选站点数 | > 1 才有选择意义 |
| `real_sdf_median` | 真实缺陷像素的有符号距离中位数 | 位置先验参考 |
| `proposal_sdf_median` / `site_sdf_median` | proposal/site 的 SDF 中位数 | 应接近 real |
| `real_radial_median` / `proposal_radial_median` | 归一化径向位置 | 应接近 real |

## 4. 重建后端

| 名称 | 说明 | 依赖 |
|---|---|---|
| `neighbor` | 逐像素最近邻（当前基线） | 无 |
| `telea` | `cv2.INPAINT_TELEA`（仓库已有但未用） | OpenCV |
| `ns` | `cv2.INPAINT_NS` | OpenCV |
| `ring` | 不重建；用洞外正常环 token 当 prototype | 无 |
| `atlas` | ECC 配准同类正常参考，拷贝对应区域 | OpenCV |
| `sd2` | SD2-inpainting 生成式补全 | 已缓存权重，需 GPU |

> `sd2` 后端代码已接入并做了懒加载缓存，但尚未在本机跑过完整推理；建议在 Stage 0 的 GPU 阶段单独验证一次。

`ring` 是强对照：如果它不比任何重建差，说明"重建"本身可能不是必要模块。

## 5. 用法

```bash
cd /mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/基线模型
export PYTHONPATH=src
PY=/home/asus/anaconda3/envs/cad-v0/bin/python

# 小规模冒烟（CPU 可跑）
"$PY" tools/v07/diag_proposal.py \
  --categories bottle \
  --backends neighbor ring \
  --normals-per-type 1 --sources-per-type 1 --max-probes-per-type 1 \
  --device cpu --name smoke

# 正式 Stage 0（建议等 MVTec 基线跑完再用 GPU）
"$PY" tools/v07/diag_proposal.py \
  --categories pcb1 pcb4 transistor capsule tile carpet \
  --backends neighbor telea ns ring atlas \
  --normals-per-type 4 --sources-per-type 2 --max-probes-per-type 8 \
  --device cuda:0 --name stage0_v1
```

输出在 `runs/v07/diag_proposal/<name>/`：`summary.json`、`summary.md`。

VisA 类别需要把 `--paths` 指向 `configs/paths/asus-data.yaml`，`--split magic`（或 `seas`）；MVTec 用默认 `--split mvtec`。

## 6. 已知的早期观察（bottle, 1 sample/type）

- `fill_feature_cos`：neighbor 0.85，broken_large 只有 0.70，broken_small 0.98；
- `proposal_support_ratio` ≈ 0.42，说明 proposal 覆盖了物体支撑的 42%；
- `proposal_components` ≈ 11，但 **`site_count = 1`、`site_proposal_ratio = 1.0`、`site_support_ratio = 0.42`**：
  H 选中的"站点"就是整个 proposal，没有定位作用；
- `ring` 的 proposal 相似度（0.78）高于 neighbor（0.66），但站点仍然是大块。

这提示：**重建可能影响 prototype，但位置/站点选择是更直接的瓶颈**。需要用完整的 Stage 0 在更多类别上确认。

## 7. Stage 0 首次结果（4 类，136 probes，CPU）

类别：bottle / carpet / grid / transistor；后端：neighbor / telea / ring / atlas；每类每类型 1 个源、2 个正常 target、最多 4 个 probe。

| 后端 | fill_feature_cos | fill_color_l1 | proposal_sim | similarity_gap | proposal_support_ratio | site_support_ratio | site_proposal_ratio | site_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neighbor(hole) | 0.9349 | 0.0289 | 0.7048 | 0.2009 | 0.4072 | 0.4072 | 1.0000 | 1.0 |
| telea(hole) | 0.9453 | 0.0233 | 0.7276 | 0.2038 | 0.4100 | 0.4100 | 1.0000 | 1.0 |
| atlas(hole) | 0.9475 | 0.0517 | 0.7470 | 0.2169 | 0.4101 | 0.4101 | 1.0000 | 1.0 |
| ring(ring) | n/a | n/a | 0.7449 | 0.1842 | 0.3949 | 0.3949 | 1.0000 | 1.0 |

### 结论

1. **重建质量是类型相关的，不是全局瓶颈。**
   - `bottle/broken_large`：neighbor 的 fill_feature_cos 只有 0.6998，atlas 0.9347、telea 0.8145；proposal 相似度 0.479 vs 0.818。大块缺失类明显受益于更好的填充。
   - `carpet/*`、`grid/*` 的 fill_feature_cos 普遍 0.94–0.98，neighbor 与 atlas/telea 差距很小。纹理类换重建收益有限。
   - 因此"训练一个统一的重建器"不是第一优先级；至少应该按缺陷类型/大小分层评估。

2. **零成本的 Telea 已经优于当前 baseline。**
   - 平均 fill_feature_cos 0.9453 vs 0.9349，颜色 L1 0.0233 vs 0.0289，proposal 相似度 0.728 vs 0.705。
   - 仓库里本来就有 `inpaint_source`，只是 A2 没用；这是一个可以直接消融的免费 baseline。

3. **H 的"站点选择"是更大的结构性问题。**
   - 136 个 probe 里，`site_proposal_ratio = 1.0`、`site_count = 1`、`site_support_ratio ≈ 0.41`：**选中的站点就是整个 proposal**，不存在"在多个站点里选一个"。
   - 原因在代码：`candidate_tokencut_sites` 切的是 **target support 的连通域**，再与 proposal 求交。大多数物体只有一个连通域，所以永远只有一个 site，`pick_tokencut_site` 的随机性没有意义。
   - 这意味着：即使重建/prototype 完美，缺陷仍然会被放在约 40% 的物体区域里，位置先验缺失。

4. **位置分布存在可测的偏差。**
   - `grid/glue`、`grid/thread`：真实缺陷 SDF 中位数约 -18（靠近/越过支撑边界），proposal 约 +18（支撑内部）；`bottle/broken_large` 的 neighbor 也出现 -9 vs +85。
   - 说明 proposal 偏向物体内部，而部分真实缺陷发生在边缘/薄结构处。

### 对路线的含义

- **先做**：把 `telea` 接进 A2（一行切换）作为免费 baseline；同时做"位置/站点"模块，而不是先训练大重建器。
- **再做**：B（mask 生成）在 proposal 内生成**局部**掩码，替代"整个 support CC 当 site"；位置先验用物体坐标系的真实缺陷分布。
- **重建器训练的位置**：作为 B 之后的增强项，优先针对大块缺失/结构化类别（bottle broken_large、pcb、transistor），用 B0/B1 mask + 正常图自监督；纹理类可以先不训。

## 8. 下一步

- 跑完整 Stage 0（建议在 MVTec 基线结束后用 GPU，加入 `sd2`）；
- 若 `site_proposal_ratio` 普遍接近 1，优先做物体坐标系位置先验 + 更局部的站点/掩码生成；
- 若 `fill_feature_cos` 与 `similarity_gap` 在 atlas/ring 上明显更好，再考虑训练重建器；
- 训练重建器时用 `v07/shape_prior.py` 的 B0 生成洞，正常图做监督，损失加 DINOv2 特征项。
