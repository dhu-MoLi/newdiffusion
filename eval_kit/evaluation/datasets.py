"""
评测数据集 — 协议与 DualAnoDiff / AnomalyDiffusion 官方评测一致。

关键协议：MVTec 每个异常类型的 test 图像，前 1/3 供生成模型训练（few-shot 支撑集），
后 2/3 用于下游评测；good 图全部保留用于评测。

与官方实现的差异：生成数据目录为
runs/inference/<class>/<anomaly>/{composed, mask_defect}（可通过参数指定子目录名）。
定位测试读图与 DualAnoDiff `MVTecDRAEMTestDataset_partial` 一致：
`cv2.imread` + `cv2.resize` 默认双线性；网络输入再把 BGR 转回 RGB
（官方在 test() 里做 `[:,[2,1,0]]`）。生成图训练读入仍用 PIL。
"""

import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _sorted_images(dir_path):
    files = [f for f in os.listdir(dir_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    # 生成数据文件名是 "0.png"、"1.png"…，MVTec 是 "000.png"…，统一按数字排序
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    return files


class MVTecTestPartialDataset(Dataset):
    """MVTec 测试集（每个异常类型跳过前 1/3，与官方 MVTecDRAEMTestDataset_partial 一致）"""

    def __init__(self, mvtec_path, class_name, resize_shape=(256, 256), anomaly_names=None):
        self.resize_shape = resize_shape
        test_root = os.path.join(mvtec_path, class_name, 'test')
        gt_root = os.path.join(mvtec_path, class_name, 'ground_truth')
        self.samples = []  # (image_path, mask_path_or_None, has_anomaly, anomaly_name)

        names = anomaly_names if anomaly_names is not None else sorted(os.listdir(test_root))
        for anomaly_name in names:
            img_dir = os.path.join(test_root, anomaly_name)
            if not os.path.isdir(img_dir):
                continue
            files = _sorted_images(img_dir)
            skip = 0 if anomaly_name == 'good' else len(files) // 3
            for fname in files[skip:]:
                if anomaly_name == 'good':
                    self.samples.append((os.path.join(img_dir, fname), None, 0, anomaly_name))
                else:
                    stem = os.path.splitext(fname)[0]
                    mask_path = os.path.join(gt_root, anomaly_name, f"{stem}_mask.png")
                    if not os.path.exists(mask_path):
                        mask_path = os.path.join(gt_root, anomaly_name, f"{stem}.png")
                    self.samples.append((os.path.join(img_dir, fname), mask_path, 1, anomaly_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, has_anomaly, _ = self.samples[idx]
        width, height = self.resize_shape
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        bgr = cv2.resize(bgr, dsize=(width, height))
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            mask = cv2.resize(mask, dsize=(width, height)).astype(np.float32) / 255.0
        else:
            mask = np.zeros((height, width), dtype=np.float32)
        mask = mask[np.newaxis, ...]

        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'has_anomaly': np.array([has_anomaly], dtype=np.float32),
        }


class GeneratedLocalizationDataset(Dataset):
    """
    U-Net 定位训练集：50% 概率采样 MVTec train/good（mask 全零），
    50% 概率采样生成的异常图 + 生成 mask（与官方 MVTec_Anomaly_Detection 一致）。
    """

    def __init__(self, mvtec_path, generated_root, class_name, anomaly_names,
                 length=500, image_subdir='composed', mask_subdir='mask_defect',
                 resize_shape=(256, 256)):
        good_dir = os.path.join(mvtec_path, class_name, 'train', 'good')
        self.good_files = [os.path.join(good_dir, f) for f in _sorted_images(good_dir)]

        self.gen = []  # 每个 anomaly 一组 (img_path, mask_path) 列表
        for anomaly in anomaly_names:
            base = os.path.join(generated_root, class_name, anomaly)
            img_dir, mask_dir = os.path.join(base, image_subdir), os.path.join(base, mask_subdir)
            pairs = [(os.path.join(img_dir, f), os.path.join(mask_dir, f))
                     for f in _sorted_images(img_dir)
                     if os.path.exists(os.path.join(mask_dir, f))]
            if not pairs:
                raise ValueError(f"未找到生成的 图像/mask 对: {img_dir} + {mask_dir}")
            self.gen.append(pairs)

        self.resize_shape = tuple(resize_shape)
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(list(self.resize_shape), antialias=True),
        ])
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random() > 0.5:
            image = self.loader(Image.open(self.good_files[idx % len(self.good_files)]).convert('RGB'))
            mask = torch.zeros((1, image.size(-2), image.size(-1)))
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            group = self.gen[random.randint(0, len(self.gen) - 1)]
            img_path, mask_path = group[idx % len(group)]
            image = self.loader(Image.open(img_path).convert('RGB'))
            mask = (self.loader(Image.open(mask_path).convert('L')) > 0.5).float()
            has_anomaly = np.array([1 if mask.sum() > 0 else 0], dtype=np.float32)
        return {'image': image, 'mask': mask, 'has_anomaly': has_anomaly}


class GeneratedClassificationDataset(Dataset):
    """异常类型分类训练集（生成数据，label = 异常类型序号）"""

    def __init__(self, generated_root, class_name, anomaly_names,
                 image_subdir='composed', max_per_class=500, repeat=5,
                 resize_shape=(256, 256)):
        self.img_paths, self.labels = [], []
        self.anomaly_names = list(anomaly_names)
        for idx, anomaly in enumerate(self.anomaly_names):
            img_dir = os.path.join(generated_root, class_name, anomaly, image_subdir)
            files = _sorted_images(img_dir)[:max_per_class]
            self.img_paths += [os.path.join(img_dir, f) for f in files]
            self.labels += [idx] * len(files)
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(list(resize_shape), antialias=True),
        ])
        self.repeat = repeat

    def __len__(self):
        return len(self.img_paths) * self.repeat

    def class_num(self):
        return len(self.anomaly_names)

    def __getitem__(self, idx):
        i = idx % len(self.img_paths)
        return self.loader(Image.open(self.img_paths[i]).convert('RGB')), self.labels[i]


class MVTecClassificationTestDataset(Dataset):
    """异常类型分类测试集（MVTec test 后 2/3，与官方一致）"""

    def __init__(self, mvtec_path, class_name, anomaly_names, resize_shape=(256, 256)):
        test_root = os.path.join(mvtec_path, class_name, 'test')
        self.img_paths, self.labels = [], []
        for idx, anomaly in enumerate(anomaly_names):
            img_dir = os.path.join(test_root, anomaly)
            files = _sorted_images(img_dir)
            skip = len(files) // 3
            self.img_paths += [os.path.join(img_dir, f) for f in files[skip:]]
            self.labels += [idx] * len(files[skip:])
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(list(resize_shape), antialias=True),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.loader(Image.open(self.img_paths[idx]).convert('RGB')), self.labels[idx]
