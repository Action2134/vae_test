"""
采样 N 张图: 随机 z ~ N(0,1) 解码, 每张单独保存 + 拼一张总览
用法: python src/sample_images.py [--n 10] [--checkpoint xxx.pt]
不指定 checkpoint 时用随机初始化权重 (未训练基线)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae import VAE
from visualize import _grid_to_image

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="生成张数")
    parser.add_argument("--checkpoint", default=None, help="可选, 训练好的 .pt")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)  # 固定采样, 可复现
    model = VAE(image_size=784, hidden_dim=512, latent_dim=16)
    tag = "untrained"
    if args.checkpoint:
        cfg = torch.load(args.checkpoint, map_location="cpu")["config"]
        model = VAE(image_size=784, hidden_dim=cfg["model"]["hidden_dim"],
                    latent_dim=cfg["model"]["latent_dim"])
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
        tag = "trained"
    model.eval()

    out_dir = args.out_dir or os.path.join(DATA_OUT, f"samples_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(args.n, model.latent_dim)
        imgs = model.decode(z)

    for i, img in enumerate(imgs):
        _grid_to_image(img.unsqueeze(0), nrow=1).save(os.path.join(out_dir, f"sample_{i:02d}.png"))
    _grid_to_image(imgs, nrow=args.n).save(os.path.join(out_dir, "overview.png"))
    print(f"[sample] {tag} 模型生成 {args.n} 张, 已保存到 {out_dir}")


if __name__ == "__main__":
    main()
