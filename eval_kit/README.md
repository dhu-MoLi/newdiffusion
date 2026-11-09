# 异常生成下游评测包（ad_gen_eval_kit）

独立可传输的评测包，协议对齐 **DualAnoDiff / AnomalyDiffusion** 官方 eval。  
用于：生成异常图像后，计算生成质量 + 下游定位/分类增益。

来源：`triple2stage` 项目 `evaluation/` + `scripts/evaluate.py` 的整合导出。

---

## 1. 包内容

```
eval_kit/
├── evaluate.py              # CLI 入口
├── requirements.txt
├── pack.sh                  # 一键打 tar.gz
├── README.md
└── evaluation/
    ├── datasets.py          # few-shot 协议数据集
    ├── gen_quality.py       # IS / IC-LPIPS
    ├── localization.py      # 下游定位（DRAEM U-Net）
    ├── classification.py    # 下游分类（ResNet-34）
    └── unet_utils/          # DiscriminativeSubNetwork / FocalLoss / AU-PRO
```

| 模块 | 指标 |
|------|------|
| `gen_quality` | IS、IC-LPIPS |
| `localization` | AUROC-I/P、AP-I/P、F1-I/P、AU-PRO |
| `classification` | 异常类型分类 accuracy（需 ≥2 种异常） |

---

## 2. 数据约定

### 生成数据（你的推理产物）

```
<generated_root>/
  <class_name>/
    <anomaly_name>/
      composed/          # 生成图：0.png, 1.png, ...
      mask_defect/       # 与图同名的 mask：0.png, 1.png, ...
```

子目录名可通过 `--image_subdir` / `--mask_subdir` 修改。

### MVTec AD

```
<mvtec_path>/
  <class_name>/
    train/good/
    test/<anomaly>/...
    test/good/...
    ground_truth/<anomaly>/...
```

### 协议要点（必须遵守，否则数字不可比）

1. MVTec 每个异常类型的 **test 前 1/3** = few-shot 支撑集（生成模型可见）；**后 2/3** = 下游评测集；`good` 全部保留。
2. 定位：用生成 (图, mask) + `train/good` 训 DRAEM `DiscriminativeSubNetwork`（默认 **200 epochs**），在 test 后 2/3 上算指标。
3. 分类：用生成图训 **ResNet-34**（与 DualAnoDiff / AnomalyDiffusion 官方一致），默认 **30 epochs**。
4. IC-LPIPS：以 test 前 1/3 为聚类中心，簇内两两 LPIPS 均值。

---

## 3. 安装

```bash
# 按本机 CUDA 安装 torch / torchvision 后：
pip install -r requirements.txt
```

---

## 4. 用法

### CLI

```bash
cd eval_kit   # 或解压后的目录

# 仅生成质量（较快）
python evaluate.py \
  --class_name hazelnut --anomaly_names cut \
  --metrics is ic_lpips \
  --generated_root /path/to/generated \
  --mvtec_path /path/to/MVTec \
  --output_dir ./eval_results

# 全套（定位约需数十分钟～数小时，视数据量与 GPU）
python evaluate.py \
  --class_name hazelnut --anomaly_names cut hole print crack \
  --metrics is ic_lpips localization classification \
  --generated_root /path/to/generated \
  --mvtec_path /path/to/MVTec \
  --output_dir ./eval_results \
  --loc_epochs 200 --cls_epochs 30

# 定位/分类改用 512（默认仍是 DualAnoDiff 的 256；IS/IC-LPIPS 不受此参数影响）
python evaluate.py \
  --class_name cable --anomaly_names bent_wire cut_inner_insulation \
  --metrics localization classification \
  --generated_root /path/to/generated \
  --mvtec_path /path/to/MVTec \
  --output_dir ./eval_results \
  --img_size 512
```

结果写入：`<output_dir>/<class_name>/results_<时间戳>.{json,csv}`。

### Python API

```python
import sys
sys.path.insert(0, '/path/to/eval_kit')

from evaluation import compute_is, compute_ic_lpips, train_localization

is_score = compute_is('/path/to/generated/hazelnut/cut/composed')
loc = train_localization(
    mvtec_path='/path/to/MVTec',
    generated_root='/path/to/generated',
    class_name='hazelnut',
    anomaly_names=['cut'],
    save_path='./ckpt_loc',
    epochs=200,
)
# loc: auroc_image, ap_image, f1_image, auroc_pixel, ap_pixel, f1_pixel, pro_pixel
```

---

## 5. 打包传输

在本目录执行：

```bash
bash pack.sh                 # → dist/ad_gen_eval_kit_<时间戳>.tar.gz
bash pack.sh /tmp/out        # 指定输出目录
```

目标机：

```bash
tar -xzf ad_gen_eval_kit_*.tar.gz
cd ad_gen_eval_kit_*
pip install -r requirements.txt
python evaluate.py --help
```

本包**不包含** MVTec 数据与生成图像，需在目标机自行提供路径。

---

## 6. 输出字段说明

| 字段 | 含义 |
|------|------|
| `IS` / `IS_per_anomaly` | Inception Score（越高越好） |
| `IC_LPIPS` / `IC_LPIPS_per_anomaly` | 簇内多样性（越高越好） |
| `loc_auroc_image` / `loc_ap_image` / `loc_f1_image` | 图像级定位 |
| `loc_auroc_pixel` / `loc_ap_pixel` / `loc_f1_pixel` | 像素级定位 |
| `loc_pro_pixel` | AU-PRO（积分上限 0.3） |
| `cls_accuracy` | 异常类型分类准确率 |

---

## 7. 注意事项

- 需要 GPU（`cuda`）；无 CUDA 时会回退 CPU，定位训练会极慢。
- 定位训练中途不计算 AU-PRO（慢），最终对最优 ckpt 补算全套指标。
- 单缺陷类型时自动跳过 classification。
- 本包与 triple2stage 仓库内 `eval_kit/` 同源；仓库根 `evaluation/` 为其软链，`scripts/evaluate.py` 为兼容包装入口。
