# 研究规划 v1：从基线到可发表的工业异常生成方法

> 版本：2026-11-09
> 基线提交：`511942d`（tag `baseline-v1`）
> 目标：在 MVTec AD / VisA 的下游检测指标上形成可验证的提升，并沉淀为高质量论文

---

## 0. 结论先行

当前基线是一个**三段式 few-shot 异常生成框架**：TokenCut 物体支撑（A）→ 手工站点缩放放置（H）→ 每类独立 SD1.5 adapter 掩码内渲染（R），最后用 DRAEM 风格 U-Net 做下游检测（D）。

它的优势是工程完整、可复现、每类只需 few-shot GT；短板也很明确：**像素级 AP 塌陷**（VisA 36.5，MVTec 70.0），且塌在纹理类与细粒度缺陷上。根因不是渲染器不够强，而是：

1. 掩码是 few-shot GT 的复制 + 缩放 + 6 档旋转，形状分布严重受限；
2. 放置是手工规则，纹理类的 TokenCut 站点退化为任意位置；
3. 生成在 512、评测在 256，小缺陷被重采样抹平；
4. 外观条件太弱，没有利用 few-shot 里的真实缺陷外观。

因此规划的核心不是"换更大的扩散模型"，而是**把"在哪、什么形状、多明显、长什么样"这四件事分别做对**。

---

## 1. 现状诊断

### 1.1 方法链路（代码落点）

| 阶段 | 作用 | 关键实现 |
| --- | --- | --- |
| A | TokenCut 物体支撑 + few-shot 洞内材料原型 | `src/cadiff/v06_a/a2.py` `_material_proposal_maps` |
| H | 在 TokenCut 连通域内缩放/旋转放置 GT mask | `src/cadiff/v06_b/a2_component_place.py:389` `instantiate_site_scale` |
| R | 冻结 SD1.5 + 可训练 adapter，掩码内 50 步 DDIM infill | `src/cadiff/v06_c/can_exact_mask_r.py` |
| D | DRAEM U-Net 下游定位 + ResNet-34 分类 | `eval_kit/evaluation/localization.py` |

关键参数：渲染 512×512；每类 500 对；R 训练 1200 步；下游 200 epoch、256×256；few-shot = 每类每缺陷类型按字典序前 `floor(n/3)`。

### 1.2 当前结果

**VisA 12 类（已完成）**：Image AUROC 0.9192 / Pixel AUROC 0.9389 / **Pixel AP 0.3653** / PRO 0.8491 / 分类 0.5203。

**MVTec（9/15 类已完成，cable 收尾中，其余在跑）**：Image AUROC 0.9820 / Pixel AUROC 0.9641 / **Pixel AP 0.7001** / PRO 0.9042。

典型塌陷：grid AP-P 0.329、toothbrush 0.549、leather 0.659、carpet 0.684；VisA 的 macaroni2 0.047、macaroni1 0.150、pcb4 0.101、pcb2 0.180。

### 1.3 瓶颈排序（按对指标的直接影响）

1. **可见性/分辨率错位**：生成 512，评测双线性降到 256。项目已有诊断（`records/diagnostics/visa_regime_visibility_20261010/summary.json`）：macaroni1 的 mask 到 256 后中位数仅 18–24 像素，mask 内平均 |ΔRGB| 5.9，只有 30–40% 像素的 L∞ 变化 ≥ 8。这会同时压低 AP-P 和 Image AUROC。
2. **掩码形状先验过窄**：`sample_area_factor` 只在对数区间采面积因子，角度只有 0/30/…/150，形状完全来自 few-shot 的 2–10 张 GT。下游检测器直接拿这个 mask 当监督，形状不真实会直接封顶 AP-P。
3. **放置先验是手工的**：站点在 TokenCut 连通域里随机挑。纹理类 TokenCut 的"物体"接近整图，站点失去约束力。
4. **外观条件弱**：R 的 prompt 只有 `"a photo of a normal {category}"`，类型仅靠 128 维 embedding；mask 内条件是纯高斯噪声，没有引用 few-shot 的真实缺陷外观。分类准确率（VisA 52%）就是直接证据。
5. **排序损失不约束真实感**：只要求"正确类型 loss 低于错误类型/normal"，可以生成"易检测但不真实"的缺陷。
6. **评测选模型泄漏**：每个 epoch 在测试集上选 checkpoint。前期快速实验可接受，论文必须换协议。

---

## 2. 目标与成功标准

### 2.1 分阶段目标

| 阶段 | 时间 | 目标 | 判定 |
| --- | --- | --- | --- |
| S0 基建 | 1 天 | 快速筛选协议 + 实验管理 | 单类实验 ≤3 GPU·h |
| S1 可见性 | 第 1–2 周 | 细粒度缺陷 AP-P 明显回升 | 4 类均值 +5 AP-P 以上 |
| S2 形状先验 | 第 3–4 周 | 纹理类 AP-P 与 mask 真实感提升 | 3 类中至少 2 类超过 S1 |
| S3 外观条件 | 第 5–6 周 | 真实感、分类、图像级 AUROC 提升 | IS/IC-LPIPS 改善且检测不退化 |
| S4 检测感知 | 第 7 周（可选） | 同数据预算下检测收益更高 | 同 500 对下 AP-P/Image AUROC 提升 |
| S5 论文评测 | 第 8 周起 | 无泄漏协议 + 3 seed + 消融 | 完整表格与复现脚本 |

### 2.2 论文级成功标准

- MVTec 15 类、VisA 12 类上，Image AUROC / Pixel AUROC / AP / PRO 至少三项超过同协议生成类基线（DualAnoDiff、AnomalyDiffusion、AnoGen 等）。
- 至少两个组件有独立消融收益（不是靠堆模块）。
- 掩码/放置/可见性有可量化的真实性证据（分布距离、可视化）。
- 完整复现：3 seed、mean±std、无测试集选模型。

---

## 3. 技术路线

### 主线 A：分辨率一致的可见性渲染（优先，低风险）

**动机**：评测在 256，小缺陷在 512→256 的过程中消失。这是当前 AP-P 最大的确定性损失来源。

**方案 A1（便宜）**：潜空间可见性正则。训练 R 时，取预测的干净 latent 与目标 latent 的差 `Δ = x0_hat - z_target`，只在 mask 内计算；对 `Δ` 做 2× 平均池化得到低频分量，约束其能量不低于阈值（或直接对降采样后的 `x0_hat` 与降采样后的 `z_target` 做 L1）。这等价于要求"降采样后仍然看得出缺陷"。

**方案 A2（更强）**：RGB 可见性损失。每 k 步用冻结 VAE 解码 `x0_hat`，双线性降到 256，在 mask 内约束与真实目标的 L1/对比度，并加一个"mask 内 vs mask 外"的对比项。显存与耗时可控（3090 单样本解码约 0.1s）。

**方案 A3（评测侧）**：生成图与 mask 同步降采样时，mask 改用最近邻；评估 256 与 512 两档，论文明确指出协议。

**实验**：E1.1 仅改评测 mask 重采样（对照）；E1.2 潜空间可见性；E1.3 RGB 可见性；E1.4 多分辨率评测。每项在 4 类快速协议上跑。

**预期**：细粒度类 AP-P +5~15；Image AUROC 在 VisA 上 +1~3。风险低。

### 主线 B：可学习的形状与放置先验（核心创新点）

**动机**：copy-paste 形状是当前方法最"不学术"的部分，也是纹理类塌陷的直接原因。把它变成"从 few-shot 学到的分布"既是性能提升，也是最容易讲清楚的创新。

**B1（先做，便宜）**：参数化形状增广。从 few-shot GT mask 估计每类的面积比、长宽比、实心度、组件数分布；采样时按经验分布而非均匀区间；增加弹性形变、多组件组合（同类型 1–3 个 mask 叠加）、边界扰动。消融每一项。

**B2（核心）**：条件掩码生成器。在 64×64 或 128×128 上训练一个条件 VAE/小扩散：
- 条件：缺陷类型 embedding + 目标物体支撑降采样 + 可选源 mask；
- 训练数据：few-shot GT mask × 强增广（仿射、弹性、翻转、多组件）；
- 推理：采样 K=8 个候选 mask，经可行性检查后按"真实性评分 + 多样性"选一个。

**B3（配套）**：物体坐标系下的位置先验。用仓库已有的 `ObjectFrame` / SDF / 局部坐标（`src/cadiff/v06_a/structure.py`）把 few-shot 缺陷位置归一化到物体坐标系，估计每类的缺陷位置密度（KDE 或热图），再映射回目标物体坐标。它替代"随机挑 TokenCut 站点"，且对纹理类同样有效。

**实验**：E2.1 参数化增广消融；E2.2 条件 VAE；E2.3 位置先验；E2.4 三者组合。指标除下游检测外，增加 **mask 真实性指标**：面积/长宽比/实心度/组件数分布与 held-out GT 的 KS 或 MMD 距离。

**风险**：每类只有 2–10 张 GT mask，直接训练 VAE 会过拟合。缓解：强增广 + 低维 latent + 与参数化增广做 hybrid；先以 B1 保底。

### 主线 C：检索增强的外观条件（R 升级）

**动机**：few-shot 里有真实缺陷外观，但当前只有"正常图 + 噪声"作为条件，模型学成每类平均外观。

**C1（便宜，先做）**：用源缺陷 latent 初始化 mask 内的噪声。把 few-shot 对应类型的缺陷区域 VAE 编码，取一个 latent 作为去噪起点（加适当噪声），而不是纯高斯噪声。改动小，可能直接提升真实感。

**C2**：外观参考注入。把源缺陷区域编码成若干 token，通过一层 cross-attention 注入 adapter；或把参考 latent 拼到 context 通道（4→8 通道），只改 adapter stem。仓库已有 `src/cadiff/v06_b/appearance.py`、`src/cadiff/v06_r/losses.py` 的 gated appearance 地基。

**C3**：外观损失 + patch 判别器。解码预测 x0 到 RGB，在 mask 内做 Charbonnier/LPIPS；可选一个小 patch GAN 区分生成缺陷与 few-shot 真实缺陷。注意稳定性，判别器失败则回退到纯感知损失。

**实验**：E3.1 latent 起点；E3.2 参考注入；E3.3 外观损失；E3.4 +判别器。指标：IS、IC-LPIPS、分类准确率、下游检测。

### 增强 D：检测感知的样本选择（可选创新）

**动机**：最终目标是下游检测，不是生成质量本身。同样 500 对，选哪些样本对检测器更有用？

**方案**：先训一个 warmup 检测器（少量 epoch），对生成样本计算检测损失/梯度影响，选"难样本 + 多样性"子集；对比随机 500 对。可作为论文的一个小贡献（"generation-for-detection"），也可作为消融。

**风险**：实现与调参成本高；放在 S3 之后，视时间决定。

---

## 4. 快速筛选协议（mini-protocol）

前期不做完整 200 epoch、15 类。定义：

- 类别：`bottle`（物体）、`carpet`（纹理）、`grid`（细纹理）、`transistor`（结构小缺陷）。需要时加 `toothbrush`。
- 每类每类型生成 200 对（而非 500）。
- 下游定位 50 epoch，关闭分类。
- 1 个 seed。
- 指标：Pixel AP 为主，Image AUROC 为辅。

**协议校验**：在 3 类上比较 50 vs 200 epoch、200 vs 500 对的结果排序；Spearman ≥ 0.8 才用于筛选，否则提高 epoch。

单次实验成本估算（3090）：R 训练 0.5–1 h + 生成 0.7–1 h + 评测 0.4 h ≈ **2–2.5 GPU·h/类**。4 类约 8–10 GPU·h，双卡一天可跑 2–3 个变体。

---

## 5. 实验矩阵与时间表

| 编号 | 实验 | 对照 | 类别 | 预计 GPU·h | 周次 |
| --- | --- | --- | --- | ---: | --- |
| E0 | mini-protocol 校验 | 全量协议 | 3 | 12 | W1 |
| E1.1 | mask 重采样修正 | 基线 | 4 | 10 | W1 |
| E1.2 | 潜空间可见性 | 基线 | 4 | 10 | W1–2 |
| E1.3 | RGB 可见性 | E1.2 | 4 | 12 | W2 |
| E2.1 | 参数化形状增广 | E1.x | 4 | 10 | W2 |
| E2.2 | 条件掩码生成器 | E2.1 | 4 | 16 | W3 |
| E2.3 | 物体坐标位置先验 | E2.1 | 4 | 8 | W3 |
| E2.4 | 形状组合 | 最好单项 | 4 | 12 | W4 |
| E3.1 | latent 起点 | E2.4 | 4 | 8 | W5 |
| E3.2 | 参考注入 | E3.1 | 4 | 14 | W5 |
| E3.3 | 外观损失 | E3.2 | 4 | 12 | W6 |
| E4 | 检测感知选择 | 随机 500 | 4 | 16 | W7 |
| E5 | 全量评测 | — | 15+12 | 60+ | W8+ |

时间表是估计值；每个阶段结束开一次 gate（见 §7）。

---

## 6. 工程与版本管理

### 6.1 分支与目录

- `main`：只放已验证的代码；基线 tag `baseline-v1`。
- `exp/<name>`：每个实验一条分支，验证后合并。
- 新代码统一放 `src/cadiff/v07/`，**不修改 v06 基线**，保证基线可复现。
  - `v07/visibility.py`：可见性损失
  - `v07/shape_prior.py`：参数化增广 + 条件掩码生成器
  - `v07/placement.py`：物体坐标位置先验
  - `v07/appearance.py`：检索/参考注入
  - `v07/pipeline.py`：统一的 训练→生成→打包→评测 编排
- 实验配置放 `configs/experiments/v07_*.yaml`；输出放 `runs/v07/<exp>/<class>/<seed>/`（不入库）。

### 6.2 每次运行必写

`run_meta.json`：git commit、配置哈希、seed、数据 manifest 哈希、GPU、耗时、指标。这样任何表格都能回溯到提交。

### 6.3 工具

- `tools/v07/run_experiment.py`：一条命令跑完一个类的全链路，支持 `--stage` 与断点续跑。
- `tools/v07/quick_eval.py`：mini-protocol。
- `tools/v07/aggregate.py`：跨类/跨 seed 聚合，直接输出 markdown 表。
- `tools/v07/mask_stats.py`：mask 真实性指标。

### 6.4 纪律

- 不覆盖已有输出目录；实验失败也保留 log。
- 任何"提升"必须给出同协议对照，不接受跨协议比较。
- 前期允许用测试指标筛选；进入论文阶段必须换成固定 epoch/验证集协议并重跑决赛选手。

---

## 7. 决策门（Gate）与放弃标准

| Gate | 检查 | 通过标准 | 不通过 |
| --- | --- | --- | --- |
| G0 | mini-protocol 与全量排序一致性 | Spearman ≥ 0.8 | 提高 epoch/类别数 |
| G1 | 可见性方案 | 4 类均值 AP-P +3 以上且无类退化 | 换 A3/多分辨率，或跳过 |
| G2 | 形状先验 | 至少 2/4 类超过 G1，mask 分布距离下降 | 保留 B1 参数化增广，停 B2 |
| G3 | 外观条件 | IS/IC-LPIPS 改善且检测不退化 | 只保留 C1/C2，弃 GAN |
| G4 | 全量 | 三项指标达到论文级 | 收缩到 2 个贡献点，明确边界 |

---

## 8. 论文叙事

**暂定主题**：Shape- and Resolution-Consistent Few-Shot Anomaly Synthesis for Downstream Detection

**贡献点**（按最终结果取舍）：

1. 学习式缺陷形状/位置先验，替代 few-shot mask 复制粘贴；给出 mask 分布层面的真实性证据。
2. 分辨率一致的可见性渲染目标，解决"生成在 512、检测在 256"的系统性损失；给出可见性-指标关系分析。
3. 检索增强的外观条件，把 few-shot 真实缺陷外观显式注入生成器。
4. （可选）检测感知的样本选择。

**实验章节**：

- 主表：MVTec 15 类、VisA 12 类的 Image/Pixel AUROC、AP、PRO、F1；mean±std over 3 seeds。
- 消融：A/H/R 各组件、可见性损失、形状先验、外观条件、数据量 100/200/500。
- 真实性：IS、IC-LPIPS、FID/KID（可加）、mask 统计距离。
- 分析：可见性 vs AP-P 散点、失败案例、不同缺陷类型的增益热图。
- 效率：每 500 对的 GPU·h。

**对比基线**：DualAnoDiff、AnomalyDiffusion、AnoGen、CutPaste、DefectGAN；检测器统一用当前 DRAEM U-Net，确保同协议。

---

## 9. 近期任务清单（按顺序）

1. **现在（无 GPU）**
   - [x] git init + `.gitignore` + init commit + tag `baseline-v1`
   - [ ] 写 `tools/v07/` 骨架与 `run_meta.json` 规范
   - [ ] 写 mask 真实性统计脚本
   - [ ] 设计 `configs/experiments/v07_*.yaml` 模板
   - [ ] 等 GitHub 链接，配置 remote（不推送直到确认）
2. **MVTec 跑完后（第 1 周）**
   - [ ] 用当前基线跑 3 类 mini-protocol，与全量结果做排序校验（G0）
   - [ ] 跑 E1.1、E1.2
3. **第 2–4 周**
   - [ ] E1.3、E2.1、E2.3、E2.2
   - [ ] 每个 gate 出一次结论，记录到 `docs/decisions.md`
4. **第 5–7 周**
   - [ ] E3.x、E4（可选）
5. **第 8 周起**
   - [ ] 决赛组合 + 无泄漏协议 + 3 seed 全量评测
   - [ ] 论文表格与图

---

## 10. 明确不做的事

- 不换扩散骨干、不从零训练大模型（成本与收益不匹配）。
- 不追 3D/多模态，不在第一轮扩到更多数据集。
- 不在未验证前重构 v06 基线代码。
- 不把"测试集选 checkpoint"的结果写进论文主表。
- 不为了多样性牺牲检测指标；两者冲突时以检测为准，并单独报告。
