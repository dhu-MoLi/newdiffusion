# v08：位置-形状解耦的目标条件 mask 合成

> 目标：在 Image / Pixel 级指标上全面超过 AnomalyDiffusion 与 MAGIC。
> 核心：先求「允许区 Ω」，再在 Ω 内学习生成形状 —— 位置与形状彻底解耦。
> 取代 PSMG v2（`docs/psmg_design_v2.md`），后者证明硬编码编辑族不可行。

## 0. 与 AD / MAGIC 的根本差异

| | AnomalyDiffusion / MAGIC | 本方案 |
| --- | --- | --- |
| mask 来源 | `p(mask \| type)` 采样 | `p(mask \| type, target)` —— 先求允许区 |
| 与正常图配对 | **随机配对** | 目标条件化（Ω 由目标图结构决定） |
| 位置 / 形状 | 耦合在一个生成器里 | **解耦**：Ω 管位置，形状模块管形状 |
| 适配性 | 无（MAGIC 仅小幅度调整） | Ω 随目标物体的位置/尺度/部件变化 |

这是范式差异，不是 trick，也是论文的核心贡献句：
> *Prior methods sample a defect mask and pair it with an arbitrary normal image; we first infer where a defect of this type may occur on the given target, then synthesise the shape inside that region.*

## 1. 允许区 Ω

### 1.1 定义

```
Ω_t(x) = Clean( σ( a·P_t(u,v) + b·D_boundary(x,S) + c·Sim_A2(x) ) > τ )
```

| 项 | 含义 | 来源 |
| --- | --- | --- |
| `P_t(u,v)` | 类型 t 的缺陷位置密度（物体坐标系） | 已实现的 L2 学习打分器 / L1 直方图 |
| `D_boundary` | 到物体边界的距离（边界类缺陷加权） | `physical_sdf_px_512` |
| `Sim_A2` | 材料相似度 | A2 proposal（可选，计算贵） |

### 1.2 清理规则（关键，解决"碎片"问题）

1. 在 **64×64** 网格上融合三路证据（像素级融合会产生噪声碎片）
2. 形态学 **闭运算 → 开运算**（半径 5–9 px）
3. 只保留面积 ≥ 1% 支撑面积的连通域，**最多 3 块**
4. 面积约束：目标面积 = clip(2× 该类型 few-shot 面积比中位数, 0.25, 0.75) × 支撑面积；
   超出则按密度排序裁剪
5. 边界外扩 ≤ 8 px（仅结构型倾向的类型启用）

### 1.3 验收（先过关，再往下做）

Ω 的定位是「宽但比物体小」，标准是**覆盖性**：

| 指标 | 目标 | 说明 |
| --- | --- | --- |
| few-shot LOMO GT 召回 | **≥ 0.95** | 用其余 mask 建先验，留一验证 |
| held-out 测试异常图 GT 召回 | **≥ 0.85** | 用异常图自身支撑，最诚实的检验 |
| 连通域数 ≤ 3 的样本比例 | ≥ 90% | 解决碎片问题 |
| Ω 面积 / 支撑面积 | 0.25–0.75 | 比整个物体明显小 |

达不到就继续调 Ω 规则（融合权重、面积、形态学半径），**不进入形状阶段**。

## 2. 形状模块

### 2.1 主设计：学习式形状生成 + 方差驱动的扰动强度

- 不再做解析旋转/平移（物体主体位置会变化，硬规则无效）
- 形状完全由模型从 few-shot mask 学习生成，**限制在 Ω 内采样**
- **扰动强度由类型内形状方差决定**：

```
σ_t = 1 - mean_{i<j} IoU(align(m_i), align(m_j))     # 物体坐标系内归一化后
```
  - `σ_t` 大（形状不固定，如 contamination / glue）→ 强随机扰动
  - `σ_t` 小（形状固定，如 cut_lead / pin）→ 几乎不扰动
  - 该值直接作为生成器的条件输入，而不是全局超参

### 2.2 阶段 A：复用 AD 的 mask 生成器（先做对照）

```
AD 原生：  m ~ p_AD(m | type)                  → 随机配对
本方案：   m ~ p_AD(m | type) ∩ Ω_t(x)         → 同一张正常图
```
- AD 的 mask 生成器与 per-class 权重已就绪（`anomalydiffusion-portable`，`train_mask.py` / `generate_mask.py`）
- **唯一变量是 Ω**，干净地验证"允许区是否提高 mask 与目标图的对齐度"

### 2.3 阶段 B：自研形状生成器

```
输入：Ω（64²）、目标 DINOv2 特征（37×37×384）、类型 embedding、σ_t、噪声 z
结构：条件 U-Net（64 或 128 分辨率，~2M 参数）
输出：mask logits → 与 Ω 相乘 → 可行性/真实性/多样性选择
训练：MVTec 393 + VisA ~430 张 mask，跨类型训练 + leave-one-type-out
```

**防泄漏（必须遵守）**：训练输入不能是带缺陷的 few-shot 图（L2 曾因此得到 AUC 0.998 的假信号）。
输入统一用**把缺陷洞填掉后的图**（Telea 或已训练的 reconstructor），或只用正常图 + 物体坐标投影标签。

## 3. 渲染 R

- 阶段一：完全复用基线 R（每类 SD1.5 adapter）
- 阶段二（mask 质量稳定超过 AD 后）：参考注入 / SD2-inpaint 9 通道
- AD 权重仅作为「同一 mask、不同渲染器」的对照实验

## 4. 评测

### 4.1 mask 级（分钟级）

- Ω：召回、连通域数、面积比
- 生成 mask：面积/长宽比/实心度/组件数的分布 KS vs held-out GT
- 对齐度：mask 质心到 Ω 质心的距离、mask ∩ Ω 比例

### 4.2 下游（seeded eval，已修复随机种子）

- 4 类 mini（bottle / carpet / grid / transistor）
- 全量 MVTec 15 类、VisA 12 类
- 与 AD / MAGIC 公开数字对比：Image AUROC、Pixel AUROC、AP-P、PRO、分类

### 4.3 核心消融

| 消融 | 验证 |
| --- | --- |
| 随机配对 vs Ω 约束 | Ω 的核心价值 |
| Ω 固定面积 vs 类型面积先验 | 面积控制 |
| σ_t 常数 vs 方差驱动 | 扰动强度设计 |
| 解析适配 vs 学习生成 | 你指出的硬规则问题 |

## 5. 阶段与 Gate

| 阶段 | 内容 | Gate |
| --- | --- | --- |
| P0 | Ω v0：L2 先验 + 边界项 + 清理，4 类验证 | 召回 ≥0.95（LOMO）/ ≥0.85（测试图） |
| P1 | Ω 全 15 类 + 调规则 | 同上 |
| P2 | AD mask 生成器 + Ω 对照 | mask 对齐度与分布 KS 明显优于 AD 原生 |
| P3 | 自研形状生成器 | mask 统计 KS ≤ copy-paste 的 0.7× |
| P4 | 4 类下游 | AP-P 均值 ≥ baseline |
| P5 | VisA 12 类对比 AD / MAGIC | 多数 Image/Pixel 指标超过 |

## 6. 风险与回退

| 风险 | 缓解 |
| --- | --- |
| Ω 召回不足 | 放宽面积/形态学参数；结构型类型允许边界外扩 |
| Ω 太宽导致 mask 无约束 | 面积上限 0.75 支撑；下游指标兜底 |
| few-shot 形状方差估计不稳 | 类型内至少 3 张才启用方差驱动，否则用类级中位数 |
| 学习式形状过拟合 | 跨类型预训练 + leave-one-type-out + 强增广 |
| P2 无效 | 说明"允许区"本身价值不足，回退到 copy-paste + 尺度校准 |

## 7. 代码落点

| 文件 | 内容 |
| --- | --- |
| `src/cadiff/v07/allowed_region.py` | Ω 构造、清理、面积控制 |
| `tools/v07/check_allowed_region.py` | LOMO / 测试图召回评测 |
| `src/cadiff/v07/shape_gen.py` | 阶段 B 条件形状生成器 |
| `tools/v07/run_ad_mask_with_omega.py` | 阶段 A：AD mask + Ω 对照 |
| `src/cadiff/v07/generate.py` | `site_method=omega_mask` 接线 |
