"""
下游异常类型分类评测 —— 协议与 DualAnoDiff / AnomalyDiffusion 官方一致。

用生成数据训练 ResNet-34，在 MVTec test 后 2/3 上测分类准确率。
类别数 = 该物体的异常类型数（至少 2 种异常类型才有意义）。
"""

import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet34

from evaluation.datasets import GeneratedClassificationDataset, MVTecClassificationTestDataset


@torch.no_grad()
def evaluate_classification(model, mvtec_path, class_name, anomaly_names, device='cuda',
                           img_dim=256):
    model.eval()
    dataset = MVTecClassificationTestDataset(
        mvtec_path, class_name, anomaly_names, resize_shape=(img_dim, img_dim))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    correct, total = 0, 0
    for image, label in dataloader:
        pred = model(image.to(device)).argmax(1)
        correct += (pred == label.to(device)).sum().item()
        total += len(label)
    return correct / max(total, 1)


def train_classification(mvtec_path, generated_root, class_name, anomaly_names,
                         save_path, image_subdir='composed',
                         epochs=30, bs=8, lr=1e-4, device='cuda', img_dim=256,
                         log_fn=print):
    """训练 ResNet-34 异常类型分类器，返回最优测试准确率"""
    if len(anomaly_names) < 2:
        log_fn(f"[cls {class_name}] 异常类型少于 2 种，跳过分类评测")
        return None

    os.makedirs(save_path, exist_ok=True)
    ckpt = os.path.join(save_path, f"{class_name}.pckl")

    dataset = GeneratedClassificationDataset(
        generated_root, class_name, anomaly_names, image_subdir=image_subdir,
        resize_shape=(img_dim, img_dim))
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=8)

    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, dataset.class_num())
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [int(epochs * 0.8), int(epochs * 0.9)], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    max_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for image, label in dataloader:
            loss = criterion(model(image.to(device)), label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = evaluate_classification(
            model, mvtec_path, class_name, anomaly_names, device, img_dim=img_dim)
        log_fn(f"[cls {class_name} {img_dim}] epoch {epoch+1}/{epochs} acc={acc:.4f}")
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), ckpt)

    return max_acc
