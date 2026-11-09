"""
下游异常定位评测 —— 协议与 DualAnoDiff eval/train-localization.py + test-localization.py 一致。

用生成的 (异常图, mask) 对 + MVTec train/good 训练 DRAEM 判别式 U-Net
（DiscriminativeSubNetwork，FocalLoss），在 MVTec test 后 2/3 上评测：
图像级 AUROC / AP / F1-max，像素级 AUROC / AP / F1-max / AU-PRO。
"""

import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch import optim
from torch.utils.data import DataLoader

from evaluation.datasets import MVTecTestPartialDataset, GeneratedLocalizationDataset
from evaluation.unet_utils.model_unet import DiscriminativeSubNetwork
from evaluation.unet_utils.loss import FocalLoss
from evaluation.unet_utils.au_pro_util import calculate_au_pro


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


@torch.no_grad()
def evaluate_localization(model_seg, mvtec_path, class_name, anomaly_names=None,
                          img_dim=256, device='cuda', compute_pro=True):
    """返回 dict：图像级与像素级全套指标"""
    model_seg.eval()
    # 评测集包含 good + 指定异常类型（None = 全部），每类跳过前 1/3
    names = None
    if anomaly_names is not None:
        names = ['good'] + [a for a in anomaly_names if a != 'good']
    dataset = MVTecTestPartialDataset(mvtec_path, class_name,
                                      resize_shape=(img_dim, img_dim), anomaly_names=names)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    image_scores, image_gt = [], []
    pixel_scores, pixel_gt = [], []
    gt_masks, pred_masks = [], []

    for batch in dataloader:
        image = batch['image'].to(device)
        out_mask = torch.softmax(model_seg(image), dim=1)
        heat = out_mask[0, 1].cpu().numpy()

        # 图像级分数 = 21x21 平滑后的最大值（官方协议）
        smoothed = torch.nn.functional.avg_pool2d(out_mask[:, 1:], 21, stride=1, padding=10)
        image_scores.append(float(smoothed.max()))
        image_gt.append(float(batch['has_anomaly'][0, 0]))

        true_mask = (batch['mask'][0, 0].numpy() > 0.5).astype(np.float32)
        pixel_scores.append(heat.flatten())
        pixel_gt.append(true_mask.flatten())
        gt_masks.append(true_mask)
        pred_masks.append(heat)

    image_scores, image_gt = np.array(image_scores), np.array(image_gt)
    pixel_scores = np.concatenate(pixel_scores)
    pixel_gt = (np.concatenate(pixel_gt) > 0.5).astype(np.uint8)

    metrics = {
        'auroc_image': roc_auc_score(image_gt, image_scores),
        'ap_image': average_precision_score(image_gt, image_scores),
        'auroc_pixel': roc_auc_score(pixel_gt, pixel_scores),
        'ap_pixel': average_precision_score(pixel_gt, pixel_scores),
    }

    # 图像级 F1-max
    prec, rec, _ = precision_recall_curve(image_gt, image_scores)
    f1 = (2 * prec * rec) / np.maximum(prec + rec, 1e-8)
    metrics['f1_image'] = float(np.max(f1[np.isfinite(f1)]))

    # 像素级 F1-max（官方 0.05 步长扫阈值）
    eps = 1e-8
    gt_bool = pixel_gt.astype(bool)
    max_f1_px = 0.0
    for score in np.arange(0.0, 1.0 + 1e-3, 0.05):
        pr = pixel_scores > score
        inter = np.logical_and(gt_bool, pr).sum()
        precision = inter / (pr.sum() + eps)
        recall = inter / (gt_bool.sum() + eps)
        max_f1_px = max(max_f1_px, 2 * precision * recall / (precision + recall + eps))
    metrics['f1_pixel'] = float(max_f1_px)

    if compute_pro:
        pro, _ = calculate_au_pro(gt_masks, pred_masks)
        metrics['pro_pixel'] = float(pro)

    return metrics


def train_localization(mvtec_path, generated_root, class_name, anomaly_names,
                       save_path, image_subdir='composed', mask_subdir='mask_defect',
                       epochs=200, bs=16, lr=1e-4, epoch_length=500,
                       device='cuda', eval_every=1, img_dim=256, accum_steps=1,
                       eval_anomaly_names=None, log_fn=print):
    """训练 U-Net。每个 epoch 用 AUROC-I+AUROC-P+AP-P+PRO 留 checkpoint。

    anomaly_names 是有生成对的类型。eval_anomaly_names 默认与它相同；
    MAGIC VisA 表把 floor(n/3)=0 的单标签只放进测试、不生成。
    """
    os.makedirs(save_path, exist_ok=True)
    ckpt = os.path.join(save_path, f"{class_name}.pckl")

    model_seg = DiscriminativeSubNetwork(in_channels=3, out_channels=2).to(device)
    model_seg.apply(_weights_init)
    optimizer = torch.optim.Adam(model_seg.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [int(epochs * 0.8), int(epochs * 0.9)], gamma=0.2)
    loss_focal = FocalLoss()

    dataset = GeneratedLocalizationDataset(
        mvtec_path, generated_root, class_name, anomaly_names,
        length=epoch_length, image_subdir=image_subdir, mask_subdir=mask_subdir,
        resize_shape=(img_dim, img_dim))
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=8)
    accum_steps = max(1, int(accum_steps))

    best_sum, best_metrics = -1.0, None
    for epoch in range(epochs):
        model_seg.train()
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            out = torch.softmax(model_seg(image), dim=1)
            loss = loss_focal(out, mask) / accum_steps
            loss.backward()
            if step % accum_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step()

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            m = evaluate_localization(model_seg, mvtec_path, class_name,
                                      anomaly_names=eval_anomaly_names or anomaly_names,
                                      device=device, img_dim=img_dim, compute_pro=True)
            sum_metric = (m['auroc_image'] + m['auroc_pixel']
                          + m['ap_pixel'] + m['pro_pixel'])
            log_fn(f"[loc {class_name} {img_dim}] epoch {epoch+1}/{epochs} "
                   f"AUROC-I={m['auroc_image']:.3f} AUROC-P={m['auroc_pixel']:.3f} "
                   f"AP-P={m['ap_pixel']:.3f} PRO={m['pro_pixel']:.3f}")
            if sum_metric > best_sum:
                best_sum = sum_metric
                torch.save(model_seg.state_dict(), ckpt)

    # 加载最优权重，补齐含 AU-PRO 的完整指标
    model_seg.load_state_dict(torch.load(ckpt, map_location=device))
    best_metrics = evaluate_localization(model_seg, mvtec_path, class_name,
                                         anomaly_names=eval_anomaly_names or anomaly_names,
                                         device=device, img_dim=img_dim, compute_pro=True)
    return best_metrics
