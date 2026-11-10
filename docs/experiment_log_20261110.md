# 实验日志 2026-11-10

## 1. MVTec 基线：15 类完整结果

| 类别 | Image AUROC | Pixel AUROC | AP-P | PRO | 分类 |
|---|---:|---:|---:|---:|---:|
| bottle | 0.9919 | 0.9733 | 0.8461 | 0.9161 | 0.8372 |
| cable | 0.9784 | 0.9601 | 0.7192 | 0.9059 | 0.5000 |
| capsule | 0.9490 | 0.9519 | 0.4002 | 0.8867 | 0.4400 |
| carpet | 0.9147 | 0.9608 | 0.6842 | 0.8551 | 0.6613 |
| grid | 0.9964 | 0.8726 | 0.3286 | 0.7820 | 0.3750 |
| hazelnut | 0.9885 | 0.9968 | 0.9119 | 0.9717 | 0.8750 |
| leather | 0.9965 | 0.9891 | 0.6588 | 0.9735 | 0.4762 |
| metal_nut | 0.9858 | 0.9827 | 0.9218 | 0.9146 | 0.6250 |
| pill | 0.9295 | 0.9947 | 0.8833 | 0.9381 | 0.5833 |
| screw | 0.8531 | 0.9611 | 0.3817 | 0.8665 | 0.3704 |
| tile | 1.0000 | 0.9861 | 0.8852 | 0.9409 | 0.8070 |
| toothbrush | 1.0000 | 0.9734 | 0.5491 | 0.9164 | n/a |
| transistor | 0.9012 | 0.7865 | 0.2886 | 0.6008 | 0.6071 |
| wood | 0.9987 | 0.9800 | 0.8366 | 0.9639 | 0.7619 |
| zipper | 0.8925 | 0.9244 | 0.5303 | 0.7315 | 0.2927 |
| **均值** | **0.9584** | **0.9529** | **0.6550** | **0.8776** | 0.5866 |

最差：transistor（AP-P 0.289 / PRO 0.601）、screw（Image AUROC 0.853）、zipper（PRO 0.732）。

### 生成成功率

| 类别 | kept | failed | 失败率 |
|---|---:|---:|---:|
| wood | 1727 | 773 | **30.9%** |
| tile | 1986 | 514 | **20.6%** |
| zipper | 3160 | 340 | 9.7% |
| toothbrush | 445 | 55 | 11.0% |
| grid | 2415 | 85 | 3.4% |
| 其余 | ≥2469 | ≤38 | <2% |

失败原因主要是 `scaled GT does not fit the canvas`。这直接对应形状/放置模块的问题，后续 B 需要处理。

## 2. 重建器（C）

### 2.1 分布内验证（B0 洞，正常图）

| 类别 | 方法 | masked_l1 | PSNR | feature_cos |
|---|---|---:|---:|---:|
| bottle | neighbor | 0.00056 | 23.73 | 0.605 |
| bottle | telea | 0.00058 | 24.09 | 0.689 |
| bottle | **learned** | 0.00400 | **49.48** | **0.9987** |
| transistor | neighbor | 0.00023 | 32.40 | 0.770 |
| transistor | telea | 0.00025 | 32.23 | 0.844 |
| transistor | **learned** | 0.00506 | **49.40** | **0.9946** |

两个类别的学习式重建器在分布内都接近完美，远优于零样本基线。

### 2.2 A2 入口诊断（真实 few-shot mask，OOD）

**bottle（12 probes/backend）**

| 后端 | fill_feature_cos | proposal 相似度 | 相似度差 | site/proposal | site/支撑 |
|---|---:|---:|---:|---:|---:|
| neighbor | 0.8667 | 0.6447 | 0.1811 | 1.0000 | 0.4196 |
| telea | **0.9051** | **0.7036** | **0.1921** | 1.0000 | 0.4261 |
| learned | 0.8990 | 0.6987 | 0.1874 | 1.0000 | 0.4243 |
| ring | n/a | 0.7497 | 0.1549 | 1.0000 | 0.4159 |

**transistor（16 probes/backend）**

| 后端 | fill_feature_cos | proposal 相似度 | 相似度差 | site/proposal | site/支撑 |
|---|---:|---:|---:|---:|---:|
| neighbor | 0.8341 | 0.5736 | 0.2098 | 1.0000 | 0.3885 |
| telea | **0.8659** | 0.6033 | 0.2107 | 1.0000 | 0.3927 |
| learned | 0.8527 | **0.6409** | **0.2350** | 1.0000 | 0.4039 |
| ring | n/a | 0.6178 | 0.2373 | 1.0000 | 0.3540 |

### 2.3 结论

1. **学习式重建器在分布内极强，但 OOD 有落差。**
   - 训练洞来自 B0（参数化先验），诊断用的是真实 few-shot GT mask；形状/位置分布不同，导致 0.998 的验证 feature_cos 降到 0.85–0.90。
   - 在 bottle 上，learned 比 neighbor 好，但略低于 telea；在 transistor 上，learned 的 proposal 相似度/相似度差是所有后端中最好的（0.641/0.235 vs telea 0.603/0.211）。

2. **Telea 是极强的零成本基线。**
   - 两个类别上 fill_feature_cos 都是最高或接近最高；除 transistor 的 proposal 相似度外，与 learned 互有胜负。
   - 在投入更多重建器训练之前，应先把它作为 A2 的默认填充并跑下游对照。

3. **H 的站点/位置问题没有被重建器解决。**
   - 所有后端、所有 probe：`site_proposal_ratio = 1.0`、`site_count = 1`、`site/支撑 ≈ 0.39–0.43`。
   - transistor 上真实缺陷的归一化径向位置约 0.79，proposal 只有 0.60；真实 SDF 中位数 18.8，proposal 25.9。proposal 明显偏向物体内部，与真实缺陷位置分布不一致。
   - 这说明重建/prototype 改善后，**位置先验仍然是入口的下一个主瓶颈**。

## 3. 下一步建议

1. **重建器 v2（若继续 C）**：训练时混入真实 few-shot GT mask（直接放在正常图上），而不只用 B0 采样；目标是在 OOD 诊断上超过 Telea。可加 target 侧 context 或 B1 学习式 mask。
2. **H/位置模块（优先级更高）**：
   - 把 `candidate_tokencut_sites` 从切 target support 改成切 **proposal 的连通域**，让 `site_count > 1`、`site/proposal < 1`；
   - 用物体坐标系的真实缺陷 SDF/径向分布做位置先验；
   - 再用 B 在 proposal 内生成局部 mask。
3. **下游验证**：先把 Telea 接进 A2，跑 4 类 mini-protocol，确认入口填充改善能否传导到 AP-P；如果传导明显，再把 learned 接进去。
4. **数据质量**：wood/tile 的高失败率需要和形状/放置模块一起解决。

## 4. 产物

- MVTec 基线：`runs/mvtec_baseline/20261108/`
- bottle 重建器：`runs/v07/reconstructors/bottle/bottle_v4/`
- transistor 重建器：`runs/v07/reconstructors/transistor/transistor_v1/`
- 诊断：`runs/v07/diag_proposal/bottle_learned_diag/`、`runs/v07/diag_proposal/transistor_learned_diag/`
- 代码：`src/cadiff/v07/reconstructor.py`、`tools/v07/train_reconstructor.py`、`tools/v07/diag_proposal.py`
