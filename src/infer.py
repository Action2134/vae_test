"""
VAE 推理脚本: 加载训练好的 checkpoint, 生成效果图
用法:
  python src/infer.py --checkpoint output/vae_mnist.pt
  python src/infer.py --checkpoint output/vae_mnist.pt --out_dir output/demo
"""
import argparse
import os

import torch

from vae import VAE
from dataset import make_loader
from visualize import generate_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="训练保存的 .pt 文件")
    parser.add_argument("--data_dir", default="/public/home/liuhuan/workspace_xd/data",
                        help="MNIST 数据目录 (取测试集图片做重建/插值)")
    parser.add_argument("--out_dir", default=None,
                        help="效果图输出目录, 默认和 checkpoint 同目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = VAE(image_size=784,
                hidden_dim=cfg["model"]["hidden_dim"],
                latent_dim=cfg["model"]["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[infer] 已加载 {args.checkpoint} (latent_dim={model.latent_dim})")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    loader = make_loader(args.data_dir, 32, train=False, shuffle=False)
    generate_all(model, loader, out_dir)


if __name__ == "__main__":
    main()
