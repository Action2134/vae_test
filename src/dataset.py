"""
MNIST 数据加载
直接解析 idx-ubyte.gz 原始文件, 不依赖 torchvision.datasets 的下载/校验逻辑
"""
import gzip
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _read_idx(path):
    """解析 MNIST idx 格式 (gzip 压缩), 自动区分图片(2051)/标签(2049)"""
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[:4], "big")
    n = int.from_bytes(data[4:8], "big")
    if magic == 2051:  # 图片
        rows = int.from_bytes(data[8:12], "big")
        cols = int.from_bytes(data[12:16], "big")
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    return np.frombuffer(data, dtype=np.uint8, offset=8)  # 标签(2049)


class MNISTDataset(Dataset):
    """
    从 data_dir/MNIST/raw/ 加载 idx 文件
    图片归一化到 [0, 1] 展平为 784 维
    """

    def __init__(self, data_dir, train=True):
        raw = os.path.join(data_dir, "MNIST", "raw")
        tag = "train" if train else "t10k"
        self.images = _read_idx(
            os.path.join(raw, f"{tag}-images-idx3-ubyte.gz")
        ).astype(np.float32) / 255.0
        self.labels = _read_idx(
            os.path.join(raw, f"{tag}-labels-idx1-ubyte.gz")
        ).astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx].reshape(-1))
        return x, int(self.labels[idx])


def make_loader(data_dir, batch_size, train=True, shuffle=None):
    ds = MNISTDataset(data_dir, train=train)
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=2, pin_memory=True, drop_last=train,
    )
