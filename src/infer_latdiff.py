"""
Latent Diffusion 推理脚本
生成链路: 纯噪声 --(去噪100步)--> z --(冻结的 VAE decoder)--> 图
顺带出对照组: 老办法直接 randn 采 z, 两张图放一起比

用法:
    python src/infer_latdiff.py                            # 默认路径
    python src/infer_latdiff.py --n 64 --seed 7            # 换数量/种子
"""
import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae import VAE
from diffusion_latent import LatentDenoiser, GaussianDiffusion

VAE_CKPT = "/public/home/liuhuan/workspace_xd/ckpt/vae_test/vae_mnist.pt"
DIFF_CKPT = "/public/home/liuhuan/workspace_xd/ckpt/vae_test/latdiff/latdiff_mnist.pt"
OUT_DIR = "/public/home/liuhuan/workspace_xd/vae_test/output/images/latdiff"


def load_vae(ckpt_path, device):
    obj = torch.load(ckpt_path, map_location="cpu")
    m = obj.get("config", {}).get("model", {})
    vae = VAE(hidden_dim=m.get("hidden_dim", 512),
              latent_dim=m.get("latent_dim", 16)).to(device)
    vae.load_state_dict(obj["model"])
    vae.eval()
    return vae


def load_diffusion(ckpt_path, device):
    obj = torch.load(ckpt_path, map_location="cpu")
    d = obj["config"]["diffusion"]
    lat = obj["config"].get("latent_dim", None)  # 训练时从 VAE 读, 这里重建用
    denoiser = LatentDenoiser(latent_dim=lat or 16,
                              hidden_dim=d.get("hidden_dim", 128),
                              time_dim=d.get("time_dim", 64)).to(device)
    denoiser.load_state_dict(obj["model"])
    denoiser.eval()
    diff = GaussianDiffusion(denoiser, timesteps=d.get("timesteps", 100)).to(device)
    print(f"[infer_latdiff] 已加载 {ckpt_path} "
          f"(timesteps={d.get('timesteps', 100)}, "
          f"参数量 {sum(p.numel() for p in denoiser.parameters())/1e6:.3f}M)")
    return diff


def save_grid(tensor, path, nrow, cell=28, pad=1):
    n = tensor.shape[0]
    nrows = (n + nrow - 1) // nrow
    img = Image.new("L", (nrow * (cell + pad) + pad, nrows * (cell + pad) + pad), 0)
    for i in range(n):
        t = tensor[i].reshape(cell, cell).clamp(0, 1).cpu()  # GPU 张量先搬回 CPU 才能 .numpy()
        tile = Image.fromarray((t.numpy() * 255).astype("uint8"))
        r, c = divmod(i, nrow)
        img.paste(tile, (pad + c * (cell + pad), pad + r * (cell + pad)))
    img.save(path)
    print(f"[infer_latdiff] 已保存 {path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", default=VAE_CKPT)
    parser.add_argument("--diff_ckpt", default=DIFF_CKPT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--n", type=int, default=100, help="生成张数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    vae = load_vae(args.vae_ckpt, device)
    diff = load_diffusion(args.diff_ckpt, device)

    # 新办法: 纯噪声 -> 逐步去噪 -> z -> decoder
    z_diff = diff.sample(args.n, device)
    save_grid(vae.decode(z_diff), os.path.join(args.out_dir, "samples_diffusion.png"),
              nrow=10)

    # 老办法 (对照组): randn 直接采 z -> decoder
    z_randn = torch.randn(args.n, vae.latent_dim, device=device)
    save_grid(vae.decode(z_randn), os.path.join(args.out_dir, "samples_randn.png"),
              nrow=10)

    # 看一眼两条路采出来的 z 的统计, 对比谁更贴近真实数据
    print(f"[infer_latdiff] diffusion z: 均值={z_diff.mean():.3f} "
          f"标准差={z_diff.std():.3f}")
    print(f"[infer_latdiff] randn   z: 均值={z_randn.mean():.3f} "
          f"标准差={z_randn.std():.3f}")
    print("[infer_latdiff] 完成")


if __name__ == "__main__":
    main()
