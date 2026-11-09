"""
异常生成下游评测包（协议对齐 DualAnoDiff / AnomalyDiffusion）。

公开 API：
- gen_quality: compute_is, compute_ic_lpips
- localization: train_localization, evaluate_localization
- classification: train_classification, evaluate_classification
- datasets: 各类 Dataset
"""

from evaluation.gen_quality import compute_is, compute_ic_lpips
from evaluation.localization import train_localization, evaluate_localization
from evaluation.classification import train_classification, evaluate_classification

__all__ = [
    'compute_is',
    'compute_ic_lpips',
    'train_localization',
    'evaluate_localization',
    'train_classification',
    'evaluate_classification',
]
