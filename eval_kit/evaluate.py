#!/usr/bin/env python3
"""
异常生成下游评测 CLI（协议对齐 DualAnoDiff / AnomalyDiffusion）。

指标：
- 生成质量：IS（图像级），IC-LPIPS（多样性）
- 下游定位（U-Net/DRAEM 判别网络）：图像级 AUROC/AP/F1-max，像素级 AUROC/AP/F1-max/AU-PRO
- 下游分类（ResNet-34）：异常类型分类准确率（需 >= 2 种异常类型）

用法示例：
  # 单缺陷类型，只算生成质量
  python evaluate.py --class_name hazelnut --anomaly_names cut \
      --metrics is ic_lpips \
      --generated_root /path/to/generated --mvtec_path /path/to/MVTec

  # 全套指标
  python evaluate.py --class_name hazelnut --anomaly_names cut hole \
      --metrics is ic_lpips localization classification \
      --generated_root /path/to/generated --mvtec_path /path/to/MVTec

生成数据目录约定：
  <generated_root>/<class>/<anomaly>/<image_subdir>/   # 默认 composed/
  <generated_root>/<class>/<anomaly>/<mask_subdir>/    # 默认 mask_defect/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

# 保证本包内 `import evaluation.*` 可用（解压到任意路径均可直接跑）
_KIT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _KIT_ROOT not in sys.path:
    sys.path.insert(0, _KIT_ROOT)

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="异常生成下游评测管线（IS / IC-LPIPS / 定位 / 分类）")
    parser.add_argument('--class_name', type=str, required=True, help='类别名，如 hazelnut')
    parser.add_argument('--anomaly_names', type=str, nargs='+', required=True,
                        help='有生成对、用于训练下游的异常类型，如 cut hole')
    parser.add_argument('--eval_anomaly_names', type=str, nargs='+', default=None,
                        help='定位测试类型。默认与 --anomaly_names 相同。'
                             'MAGIC VisA 可额外包含 floor(n/3)=0、未生成的单标签类型')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['is', 'ic_lpips', 'localization', 'classification'],
                        choices=['is', 'ic_lpips', 'localization', 'classification'])
    parser.add_argument('--generated_root', type=str, required=True,
                        help='生成数据根目录（<root>/<class>/<anomaly>/<image_subdir>）')
    parser.add_argument('--mvtec_path', type=str, required=True,
                        help='MVTec AD 根目录（含 <class>/train|test|ground_truth）')
    parser.add_argument('--image_subdir', type=str, default='composed',
                        help='生成图像子目录名（默认 composed）')
    parser.add_argument('--mask_subdir', type=str, default='mask_defect',
                        help='生成 mask 子目录名（默认 mask_defect）')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='结果输出根目录（默认 ./eval_results）')
    parser.add_argument('--loc_epochs', type=int, default=200)
    parser.add_argument('--cls_epochs', type=int, default=30)
    parser.add_argument('--img_size', type=int, default=256,
                        help='定位/分类训练与评测分辨率（默认 256，与 DualAnoDiff 官方一致；可设 512）')
    parser.add_argument('--loc_bs', type=int, default=16,
                        help='定位 U-Net batch size（默认 16；512 显存不够时可改小）')
    parser.add_argument('--loc_accum', type=int, default=1,
                        help='定位梯度累积步数（默认 1；与 --loc_bs 相乘等于有效 batch）')
    args = parser.parse_args(argv)

    if args.img_size < 32:
        raise ValueError('--img_size must be >= 32')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(args.output_dir, args.class_name)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = {
        'class_name': args.class_name,
        'image_subdir': args.image_subdir,
        'mask_subdir': args.mask_subdir,
        'generated_root': os.path.abspath(args.generated_root),
        'mvtec_path': os.path.abspath(args.mvtec_path),
        'img_size': args.img_size,
        'loc_bs': args.loc_bs,
        'loc_accum': args.loc_accum,
        'loc_effective_bs': args.loc_bs * max(1, args.loc_accum),
    }

    # ---------- 生成质量 ----------
    if 'is' in args.metrics:
        from evaluation.gen_quality import compute_is
        scores = {}
        for anomaly in args.anomaly_names:
            img_dir = os.path.join(
                args.generated_root, args.class_name, anomaly, args.image_subdir)
            print(f"[IS] {args.class_name}/{anomaly} ...")
            scores[anomaly] = compute_is(img_dir)
            print(f"[IS] {anomaly}: {scores[anomaly]:.4f}")
        results['IS_per_anomaly'] = scores
        results['IS'] = sum(scores.values()) / len(scores)

    if 'ic_lpips' in args.metrics:
        from evaluation.gen_quality import compute_ic_lpips
        scores = {}
        for anomaly in args.anomaly_names:
            img_dir = os.path.join(
                args.generated_root, args.class_name, anomaly, args.image_subdir)
            print(f"[IC-LPIPS] {args.class_name}/{anomaly} ...")
            scores[anomaly] = compute_ic_lpips(
                img_dir, args.mvtec_path, args.class_name, anomaly, device=device)
            print(f"[IC-LPIPS] {anomaly}: {scores[anomaly]:.4f}")
        results['IC_LPIPS_per_anomaly'] = scores
        results['IC_LPIPS'] = sum(scores.values()) / len(scores)

    # ---------- 下游定位 ----------
    if 'localization' in args.metrics:
        from evaluation.localization import train_localization
        print(
            f"[定位] 训练 U-Net（{args.loc_epochs} epochs, {args.img_size}px, "
            f"bs={args.loc_bs}, accum={args.loc_accum}）..."
        )
        loc = train_localization(
            args.mvtec_path, args.generated_root, args.class_name, args.anomaly_names,
            save_path=os.path.join(out_dir, 'ckpt_localization'),
            image_subdir=args.image_subdir, mask_subdir=args.mask_subdir,
            epochs=args.loc_epochs, bs=args.loc_bs, device=device,
            img_dim=args.img_size, accum_steps=args.loc_accum,
            eval_anomaly_names=args.eval_anomaly_names)
        results.update({f"loc_{k}": v for k, v in loc.items()})

    # ---------- 下游分类 ----------
    if 'classification' in args.metrics:
        from evaluation.classification import train_classification
        print(f"[分类] 训练 ResNet-34（{args.cls_epochs} epochs, {args.img_size}px）...")
        acc = train_classification(
            args.mvtec_path, args.generated_root, args.class_name, args.anomaly_names,
            save_path=os.path.join(out_dir, 'ckpt_classification'),
            image_subdir=args.image_subdir, epochs=args.cls_epochs, device=device,
            img_dim=args.img_size)
        if acc is not None:
            results['cls_accuracy'] = acc

    # ---------- 输出 ----------
    json_path = os.path.join(out_dir, f'results_{stamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, f'results_{stamp}.csv')
    flat = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(flat.keys()))
        writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in flat.values()])

    print("\n========== 评测汇总 ==========")
    for k, v in flat.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\n结果已保存: {json_path}\n            {csv_path}")
    return results


if __name__ == '__main__':
    main()
