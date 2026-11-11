# L2：学习式放置打分器（Learned Placement Prior）

> 目标：把「缺陷该放在哪」从手工直方图（L1）升级为从 few-shot 学出来的位置打分器。
> 形状尺寸仍沿用 few-shot 先验，保证与 baseline / L1 可比。
> 对应研究规划 `research_plan_v1.md` 的 B3 升级项（编号 E2.5），并为 L3（E2.2 条件掩码生成器）提供位置条件。

## 1. 为什么从 L1 升级到 L2

2026-11-11 的实测事实：

- proposal 的 8 连通域中，最大连通域占 93–98% 面积，L0「切 proposal 连通域」是空操作；
- L1 手工直方图：carpet AP-P +0.033 / PRO +0.027，bottle AP-P +0.029 但 PRO −0.023，grid AP-P 0.393 → 0.160；
- 入口改善到下游的传导不均匀：Telea 填充在 bottle 全面上涨，在 carpet/grid 下跌。

L1 的三个结构性弱点：

1. **只用位置**：同一个坐标，落在「材料相似、像缺陷易发区」和「平坦区」被同等对待，没有用外观上下文。
2. **每类一个 64×64 直方图**：few-shot 每类每类型只有 2–10 张 mask，直方图噪声大；也无法区分同类不同实例的尺寸/摆放差异。
3. **坐标系依赖支撑**：纹理类（如 carpet）TokenCut 支撑不可靠（真实缺陷 SDF 中位数为负），直方图被扭曲。

L2 把「位置」扩展成「DINOv2 上下文 + 几何 + 材料 + 类型」的联合打分。

## 2. 任务定义

- 输入：目标正常图 `x`（512×512）、缺陷类型 `t`、物体支撑 `S`、A2 材料相似度图 `sim`。
- 候选：proposal 内所有可行平移位置 `p`（与 `place_shape_inside_proposal` 的 feasible 网格一致）。
- 输出：`s(p) ∈ R`，越大表示越像真实缺陷会出现的位置。
- 标签：few-shot 真实缺陷 mask —— 正样本 = GT 内位置，负样本 = proposal 内其余位置。

**注意**：这是「位置打分」不是「站点分类」。proposal 是单连通大区域，按站点分类没有自由度（L0 的教训）。

## 3. 特征（每个候选位置）

| 组 | 特征 | 来源 | 维度 |
| --- | --- | --- | ---: |
| DINOv2 上下文 | patch token（36×36 网格双线性采样）+ 3×3 邻域均值 + 5×5 均值 | `load_official_dinov2_encoder` | 384×3 |
| 几何 | 归一化坐标 (u,v)、径向 r、归一化 SDF（clip 到 [-1,1]）、到支撑质心距离 | `support_frame` / `physical_sdf_px_512` | 5 |
| 材料 | `sim(p)`、`sim` 局部均值、是否在 proposal 内 | A2 `similarity` / `proposal` | 3 |
| 类型 | 可学习类型 embedding + 该类型 few-shot 统计（面积比、长宽比、实心度、组件数） | few-shot GT | 8 + d_t |
| 先验（可选） | L1 直方图对数密度（作为特征输入，而不是直接当权重） | `ObjectLocationPrior` | 1 |

DINOv2 384 维过大：每类 fit 一个 PCA 降到 32 维，或线性投影到 64 维；PCA 只在 few-shot 上 fit。

## 4. 模型三档（先验证信号，再上容量）

| 档 | 模型 | 目的 | 成本 |
| --- | --- | --- | --- |
| M0 | LightGBM / 逻辑回归 | 判断信号是否存在于特征中 | 分钟级，CPU |
| M1 | MLP 384→256→128→1 + 类型 embedding | 主模型，每类单独训练 | 分钟级，CPU/GPU |
| M2 | DINO 网格 36×36×384 → 1×1 conv → 36×36 → 上采样 512 | 稠密打分图，空间平滑，直接给 L3 当条件 | 中等 |

推荐：先 M0 → M1；只有当 M1 明显优于 L1 再做 M2。

## 5. 数据协议：Leave-One-Mask-Out

每类每类型只有 2–10 张 mask，必须避免自证：

- 每次留一张 GT mask 作测试，其余用于训练，逐张轮换；
- 正样本：GT 内均匀采样 64 点/张；
- 负样本：同图 proposal 内拒绝采样 640 点/张（1:10）；
- 增广：随机旋转/翻转（mask 与坐标同步变换）、跨图负样本；
- **禁止**用测试类别的下游指标调超参。

## 6. 训练目标

- 二分类 logloss（`pos_weight=10`）；或 pairwise ranking（正样本分数 > 负样本 + margin）；
- 早停用 held-out 的 LOMO NLL；
- 每类保存：模型权重、特征缓存、LOMO 指标。

## 7. 推理与集成

- 新增 `LearnedPlacementPrior`，与 `ObjectLocationPrior` 同接口：
  - `heatmap(target_support, defect_type) -> (512,512) float32`
  - `score_mask(mask, support, defect_type) -> float`
- `site_method="learned_prior"`：复用 `instantiate_site_scale_location_weighted` 的路径，只把权重换成学习打分图。
- 组合权重：`w = s_hat^α · sim^β`，默认 α=1、β=0（与 L1 严格可比）；α∈{0.5,1,2}、β∈{0,0.3,0.5} 做小网格。
- 产物：`runs/v07/placement_scorer/<category>/model.pt` + `features.npz`。

## 8. 评测（先入口，后下游）

入口级（分钟级，先做）：

1. LOMO 位置 NLL（bits）vs L1 直方图；
2. top-k 命中率：真实缺陷质心落在打分前 5%/10%/20% 区域的比例；
3. 预测 argmax 与真实质心的像素距离分布。

分布级：

- 生成样本位置分布（质心）与 held-out GT 的 MMD / KS（按类型）。

下游：

- 4 类 mini（bottle / carpet / grid / transistor）× 200 对 × 50 epoch；
- 对照：`mini_baseline`（已跑）、`mini_prior_weighted`（已跑）；
- 按类型拆 AP-P，重点看地毯/网格（纹理）与 transistor（位置偏差最大：真实径向 0.79 vs proposal 0.60）。

## 9. 代码落点

| 文件 | 内容 |
| --- | --- |
| `src/cadiff/v07/placement_scorer.py` | 特征提取、样本构造、LOMO 训练、`LearnedPlacementPrior` |
| `tools/v07/train_placement_scorer.py` | CLI：缓存特征 → 训练 → 输出指标 |
| `src/cadiff/v07/placement.py` | 复用加权放置，接 `learned_prior` |
| `tools/v07/run_experiment.py` | 新增 `learned_prior_root` 配置项 |

## 10. 时间与风险

| 项 | 估计 |
| --- | --- |
| 4 类特征缓存 | ~20 分钟 |
| M0 + M1 训练与 LOMO | 30–60 分钟 |
| 下游 mini（复用已训 R） | 6–8 小时（可与其它实验并行） |

| 风险 | 缓解 |
| --- | --- |
| few-shot 太少，LOMO 方差大 | 先 M0；按类型池化；用类型统计特征兜底 |
| 纹理类支撑不可靠 | SDF 特征可关闭；加「到 mask 边界距离」替代 |
| 与 L1 无显著差异 | 不作为独立方法主张，退化为 L3 的条件特征 |

## 11. 与 L3 的接口

- L2 的打分图作为 L3 的条件通道之一（与目标支撑、类型 embedding 并列）；
- 若 L2 明显优于 L1：L3 只需学形状，位置交给 L2；
- 若 L2 ≈ L1：L3 必须联合学位置 + 形状，L2 退化为输入特征。

## 12. 与现有实验编号的对应

| 本文件 | research_plan_v1 | 状态 |
| --- | --- | --- |
| L1 位置直方图 | E2.3 | 已跑（mini_prior_weighted） |
| **L2 学习式打分器** | E2.5（新增） | 本文档 |
| L3 条件掩码生成器 | E2.2 | 待 L2 结论 |
| 检测感知选择 | E4 | 排后 |
