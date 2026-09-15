"""
未训练 VAE 推理: 随机初始化权重直接生成效果图, 作为训练前的对照基线
用法: python src/infer_untrained.py
输出: output/untrained_demo/{reconstruction,samples,interpolation}.png
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae import VAE
from dataset import make_loader
from visualize import generate_all

DATA_DIR = "/public/home/liuhuan/workspace_xd/data"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", "untrained_demo")


def main():
    torch.manual_seed(42)  # 固定随机初始化, 结果可复现
    model = VAE(image_size=784, hidden_dim=512, latent_dim=16)  # 不加载 checkpoint, 纯随机权重
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[infer_untrained] 随机初始化 VAE, 参数量 {n_params/1e6:.2f}M (未训练)")

    loader = make_loader(DATA_DIR, 32, train=False, shuffle=False)
    generate_all(model, loader, OUT_DIR)


if __name__ == "__main__":
    main()
