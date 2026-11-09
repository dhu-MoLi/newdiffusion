"""
生成质量指标：IS 与 IC-LPIPS —— 协议与 DualAnoDiff / AnomalyDiffusion 一致。

- IS: 用 torch-fidelity 对生成图像目录计算 Inception Score
  （官方 compute-is.py 同样调用 fidelity 的 --isc）。
- IC-LPIPS: 官方 compute-ic-lpipis.py 的协议——
  以 MVTec 该异常类型 test 图像的前 1/3（few-shot 支撑集）为聚类中心，
  将每张生成图按最小 LPIPS 距离分配到某中心，
  在每个簇内（最多 50 张）计算两两平均 LPIPS，再对簇取平均。数值越高多样性越好。
"""

import os
import random

import torch
from PIL import Image
from torchvision import transforms


def compute_is(image_dir, gpu=0):
    """Inception Score（均值）"""
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1=image_dir,
        isc=True,
        cuda=torch.cuda.is_available(),
        verbose=False,
    )
    return float(metrics['inception_score_mean'])


_preprocess = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def compute_ic_lpips(generated_dir, mvtec_path, class_name, anomaly_name,
                     cluster_size=50, device='cuda', seed=0):
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    random.seed(seed)

    ori_dir = os.path.join(mvtec_path, class_name, 'test', anomaly_name)
    ori_files = sorted(os.listdir(ori_dir))
    n_centers = max(len(ori_files) // 3, 1)  # 前 1/3 = few-shot 支撑集

    def load(path):
        return _preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        centers = [load(os.path.join(ori_dir, ori_files[k])) for k in range(n_centers)]

        gen_files = [f for f in sorted(os.listdir(generated_dir))
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        clusters = [[] for _ in range(n_centers)]
        for fname in gen_files:
            img = load(os.path.join(generated_dir, fname))
            dists = torch.stack([lpips_fn(c, img).flatten() for c in centers])
            clusters[int(dists.argmin())].append(os.path.join(generated_dir, fname))

        cluster_means = []
        for members in clusters:
            random.shuffle(members)
            members = members[:cluster_size]
            if len(members) < 2:
                continue
            tensors = [load(p) for p in members]
            dists = []
            for i in range(len(tensors)):
                for j in range(i + 1, len(tensors)):
                    dists.append(lpips_fn(tensors[i], tensors[j]).flatten())
            cluster_means.append(torch.cat(dists).mean())

    if not cluster_means:
        return float('nan')
    return float(torch.stack(cluster_means).mean())
