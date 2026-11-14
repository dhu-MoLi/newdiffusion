# PSMG v2：部件结构感知的 Mask 生成器

> 一句话：把 mask 生成为「对物体部件结构的编辑」，三层结构 Router → Family → Atom decoder；族与参数由 few-shot mask **解析逆推**得到，不依赖缺陷类型名称。
> 取代此前的手写算子方案（`structural.py` 的角色降级为解码器）。

## 0. 为什么需要重做

1. 手写算子（merge/split/displace）冒烟即暴露问题：`broken_large` 被名字映射成细裂缝、面积靠固定半径控制、每加一类就多一个分支——这是堆规则，不是方法。
2. 与 AD 的差距集中在大/结构型缺陷（stuck_together 真实占图 24.2%，我们只生成 5.9%），copy-paste 与位置/填充优化都够不到。
3. 尺度校准（`scale_calibrated_v4`）解决了「大缺陷被压小」，但**没有解决「大缺陷应该长什么样」**——这正是本模块要补的。

## 1. 统一抽象：mask 是「对部件结构的编辑」

| 缺陷类 | 作用域 | 编辑 |
| --- | --- | --- |
| 表面（contamination / glue / color） | 单部件内部区域 | 区域变形 + 面积缩放 |
| 结构（cut / broken / crack） | 单部件 | 切割 / 缺口 |
| 逻辑（stuck_together） | 两部件之间 | 桥接 |
| 逻辑（misplaced / flip） | 部件 + 目标位置 | 位移 |
| 逻辑（missing / broken_large） | 部件整体 | 移除（需背景渲染） |
| 形变（bent / dent） | 部件边界 | 位移场 |

表面与逻辑共享同一表示，只是作用域不同。**这是统一叙事，也是与 AD 的差异点。**

## 2. 问题设定与假设

- 设定：per-class few-shot，每个缺陷类型 2–10 张 GT mask（MVTec 平均 5.4 张/类型，VisA ~8.3）。
- 输出：512×512 mask，落在物体支撑附近，面积/形状服从该类型的 few-shot 分布。
- **显式假设 H1**：缺陷可表示为一组有限的结构编辑。假设失效时走 generic fallback（见 §7）。
- 不做：像素级 mask 扩散/VAE（每类 27 张 mask，必然过拟合）。

## 3. 结构解析器（Structure parser）

输入：目标正常图 `x`、支撑 `S`、A2 相似度 `sim`。

1. **部件分解**：DINOv2 patch 特征（37×37×384）→ 在 `S` 内做谱聚类（K 由 eigengap 决定，范围 1–6）→ 最小面积/连通性后处理。
2. **结构描述子**
   - 全局：部件数 K、纹理度（Laplacian 能量 + 特征谱各向异性）、支撑面积比、SDF 统计；
   - 每部件：面积比、长宽比、实心度、归一化位置/径向、边界复杂度、邻接数、DINOv2 均值 embedding；
   - 邻接图：部件对最小距离、接触长度。
3. **降级判定（关键）**：`纹理度 > τ` 或（K=1 且部件面积占比 > 0.8）→ 标记 `texture_mode`，Router 直接走 surface 路径，不使用部件图。

纹理类（carpet / grid / wood）没有部件概念，必须显式降级，而不是让谱聚类切出无意义碎片。

## 4. 类型 → 族：解析式、无名称

### 4.1 六个族

| 族 | 物理机制 | 判族特征（从 GT mask 解析） | atom 参数 |
| --- | --- | --- | --- |
| `surface_texture` | 材质表面覆盖 | 面积比 0.1–10%、长宽比 <5、不跨部件、SDF 中位数居中 | 位置、面积、长宽比、边界扰动 |
| `material_gain` | 额外材料 / 粘连 | 位于部件间或边界、接触 ≥2 个部件或外扩出支撑 | 桥宽、桥长、两端部件 |
| `material_loss` | 缺件 / 大片缺失 | 面积比 >10%，或与某部件的 IoU > 0.6 | 部件选择、收缩比 |
| `geometric_cut` | 切割 / 裂缝 | 长宽比 >5，或 面积/凸包面积 <0.3 | 线段方向、长度、宽度 |
| `deformation` | 弯曲 / 凹陷 | 位于边界、凸包差异大、面积/周长比低 | 边界弧段、位移场幅度 |
| `displacement` | 错位 / 翻转 | 两个分离区域（原位置+新位置），或 mask 与支撑形状高度相似 | 部件、位移向量 |

### 4.2 判族规则（不读名字）

对每张 GT mask 计算特征向量：

```
f = [area_ratio, aspect, solidity, n_components, crosses_parts,
     boundary_sdf_p65, hull_ratio, part_iou_max, centroid_uv, pair_gap]
```

- 用各族原型（典型特征区间）做最近原型分类，输出族概率；
- 低置信度（< τ_family）→ generic；
- 可选弱先验：类型名 → 文本 embedding → 族分类器，与特征分类器概率相乘后重归一化；
- 名称映射**只作为上限对照**，不作为唯一路径。

### 4.3 为什么这样对 OOD 更稳

族由 mask 的物理统计决定，而不是数据集约定。新数据集里名字不同（`merged`/`stuck_together`）或没有名字，判族仍然可用。

## 5. 逆推（analytic inversion，不做搜索优化）

对每张 GT mask：

1. 判族（§4.2）；
2. 计算族内参数（全部解析，<10ms/张）：
   - `surface_texture`：质心 (u,v)、面积比、长宽比、边界扰动幅度；
   - `material_gain`：最近的两个部件、接触点、桥长/桥宽；
   - `material_loss`：IoU 最大的部件、收缩比；
   - `geometric_cut`：PCA 主方向、长度、宽度；
   - `deformation`：边界弧段端点、法向位移幅度；
   - `displacement`：部件、位移向量（两个区域质心差）。
3. 输出 `(family, part_indices, params)` 作为训练标签；
4. 多解处理：若两个族都能重建，保留 top-2 作为软标签。

800 张 mask 的逆推总耗时分钟级，且随类型线性扩展，无需逐 mask 优化。

## 6. 生成模型（共享，跨类）

- 输入：结构描述子（全局 + 每部件）+ 族 embedding + 可选类型 embedding（文本）；
- 模型：MLP（必要时小型 VAE），输出：
  - 族 logits（surface 内额外输出子类型先验）
  - 部件选择分布
  - 参数均值/方差（高斯 NLL）
- 训练：逆推标签 + 强增广（旋转/翻转/缩放，部件索引同步变换）；
- 损失：族 CE + 部件 CE + 参数高斯 NLL；
- 采样：K=8 候选 → 可行性（落在支撑内、面积范围）→ 真实性分数（与类型统计的 KS）+ 多样性（两两 IoU）→ 选 1。

**与现在手写算子的本质区别**：算子只是解码器；选择哪个族、作用在哪个部件、参数多大，全部是学出来的。

## 7. Router 与 generic fallback

```
Router（学习 + 规则混合）
  ├ texture_mode → surface_texture 路径（无部件图）
  ├ 有部件图     → structure 路径（5 个结构族）
  └ 族置信度 < τ → generic fallback
generic fallback = 尺度校准的局部形状采样 + L1/L2 位置先验（即现有 baseline 的稳定路径）
```

generic fallback 保证「机制未见过」时**退化而不崩溃**，这是 OOD 的第一道保险。

## 8. 与现有组件的接口

- 新增 `site_method=psmg`；
- `scale_calibration`：作为 generic 路径和参数幅度先验；
- `location.py` / `placement_scorer.py`：作为 surface 族的位置先验；
- `structural.py`：降级为 atom decoder（merge/split/displace 三个原子）；
- R（SD infill）与 D（DRAEM）不变。
- **已知缺口**：`material_loss` 需要背景渲染（当前 R 只会填充材料）。短期用「从图像背景区域复制粘贴」的解析渲染器，长期再考虑条件渲染。

## 9. 评测设计

### 9.1 代理指标（分钟级，先做）

- 判族准确率（leave-one-type-out，按族混淆矩阵）；
- mask 统计 KS/MMD：面积比、长宽比、实心度、组件数（按族与按类型）；
- **逆推重建 IoU**：用逆推参数重建 GT 的 IoU，衡量 atom 表达力（若 <0.6，说明原子库不够）。

### 9.2 下游（Gate）

- 试点 6 个类型：覆盖 3 族
  - surface：bottle/contamination、carpet/color
  - cut：transistor/cut_lead、grid/broken
  - displacement：transistor/misplaced
  - gain：transistor/damaged_case（若判为 gain）或 VisA stuck_together
- 全量：MVTec 15 类、VisA 12 类；
- 汇总：按族报告 AP-P，并给每类型 3 seed 的均值±标准差（后期）。

## 10. 审稿对策实验

| 潜在质疑 | 对策实验 | 通过标准 |
| --- | --- | --- |
| 族是手工定义的，本质仍是规则 | family-conditioned vs family-agnostic 消融；数据驱动聚类与手工族的一致性 | 族条件带来 >0.05 AP-P 提升 |
| 类型名映射是人工的 | 无名称自动判族（统计 + 文本）准确率；名称映射只作上限 | 自动判族 ≥80% |
| 只在 MVTec/VisA 有效 | 留类交叉 + 跨数据集（MVTec→VisA / MPDD / BTech），按族报告 | 跨数据集族级指标不崩（KS 增幅 <20%） |
| 假设失效怎么办 | 报告 generic fallback 触发率与对应样本的下游表现 | 触发时有明确降级，不产生坏样本 |
| 大缺陷靠什么赢 | 与 AD 逐类型对比（stuck_together / missing） | 面积比接近 1，AP-P 提升 |

## 11. 实施计划与 Gate

| 步骤 | 内容 | 估计 | Gate |
| --- | --- | --- | --- |
| 0 | 部件分解 + 降级判定（15 类） | 0.5 天 | G0：纹理类 100% 正确降级，物体类 K 分布合理 |
| 1 | 判族 + 解析逆推标签 | 0.5 天 | G1：leave-one-type-out 判族 ≥80%；重建 IoU ≥0.6 |
| 2 | 共享生成器 + 采样选择 | 1 天 | G2：族级 mask 统计 KS < baseline 的 0.7 倍 |
| 3 | 试点 6 类型下游 | 1 天 | G3：结构型类型 AP-P ≥ baseline +0.10，整体不低于 baseline |
| 4 | 全量评测 + 审稿对策实验 | 2 天 | 论文主表 |

每关不过就停下来分析，不进入下一步。

## 12. 风险与回退

| 风险 | 缓解 |
| --- | --- |
| 部件分解在纹理/低对比度失败 | `texture_mode` 显式降级；G0 专门验证 |
| 逆推多解 | top-2 软标签 + 族先验 + 低维参数 |
| few-shot 太少（2–3 张/类型） | 跨类共享预训练 + 低维参数 + 强增广 |
| material_loss 渲染错误 | 背景复制渲染器；单独评估该类 |
| G2/G3 不达标 | 回退到已验证有效的 scale_calibrated_v4 + surface 族 |

## 13. 代码落点

| 文件 | 内容 |
| --- | --- |
| `src/cadiff/v07/structure_parts.py` | 部件分解、结构描述子、纹理度与降级判定 |
| `src/cadiff/v07/psmg.py` | 判族、解析逆推、生成器、采样与选择 |
| `tools/v07/train_psmg.py` | 数据构建、训练、代理指标 |
| `src/cadiff/v07/generate.py` | `site_method=psmg` 接入 |
| 产物 | `runs/v07/psmg/<class>/{model.pt, summary.json}` |

## 14. 与现有工作的差异（novelty 声明）

- vs copy-paste / CutPaste / DRAEM 增广：学的是「编辑分布」，不是复制形状；
- vs AnomalyDiffusion / DualAnoDiff：它们在扩散模型里生成外观 + 简单 mask，我们在**结构层面**编辑 mask，且覆盖逻辑型缺陷；
- vs 参数化形状先验（B1）：那只是形状统计；本模块的算子对应物体部件与逻辑关系，且选择/参数是学出来的；
- vs 手写算子：算子降级为解码器，方法在于「结构 → 编辑」的映射。
