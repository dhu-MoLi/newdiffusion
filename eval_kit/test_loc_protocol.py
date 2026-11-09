"""定位测试读图与选 checkpoint 合同。不训练。"""

import inspect
import tempfile
from pathlib import Path

import cv2
import numpy as np

from evaluation.datasets import MVTecTestPartialDataset
from evaluation.localization import train_localization


def test_test_loader_is_cv2_bilinear_rgb():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_dir = root / "toy" / "test" / "spot"
        gt_dir = root / "toy" / "ground_truth" / "spot"
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[0, 0] = (0, 0, 255)  # red in RGB
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[:, 0] = 255
        cv2.imwrite(str(img_dir / "000.png"), bgr)
        cv2.imwrite(str(gt_dir / "000_mask.png"), mask)

        sample = MVTecTestPartialDataset(str(root), "toy", resize_shape=(4, 4))[0]
        image = sample["image"].numpy()
        got = sample["mask"][0].numpy()
        assert image.shape == (3, 4, 4)
        assert image[0, 0, 0] > image[2, 0, 0]
        assert 0.0 < got[0, 1] < 1.0


def test_checkpoint_sum_uses_pro_every_epoch():
    signature = inspect.signature(train_localization)
    assert signature.parameters["eval_every"].default == 1
    assert "eval_anomaly_names" in signature.parameters
