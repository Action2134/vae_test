"""
条件 Latent Diffusion 推理脚本: 指定数字生成
生成链路: 指定数字 y + 纯噪声 --(条件去噪100步)--> z --(冻结 VAE decoder)--> 图
默认出 10x10 网格: 每行命令模型生成一个数字(0-9), 每列一个随机种子
    => 每行都是同一个数字 = 条件生效; 同列笔迹相近 = z 管风格, y 管内容

用法:
    python src/infer_latdiff_cond.py                  # 10x10 网格, 每行一个数字
    python src/infer_latdiff_cond.py --digit 8        # 只生成数字 8 (一排)
    python src/infer_latdiff_cond.py --per_class 10 --seed 7
"""
import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae import VAE
from diffusion_latent_cond import CondLatentDenoiser, CondGaussianDiffusion

VAE_CKPT = "/public/home/liuhuan/workspace_xd/ckpt/vae_test/vae_mnist.pt"
DIFF_CKPT = "/public/home/liuhuan/workspace_xd/ckpt/vae_test/latdiff_cond/latdiff_cond_mnist.pt"
OUT_DIR = "/public/home/liuhuan/workspace_xd/vae_test/output/images/latdiff_cond"


def load_vae(ckpt_path, device):
    """加载冻结 VAE: 推理只用它的 decoder 把 z 还原成图"""
    obj = torch.load(ckpt_path, map_location="cpu")
    m = obj.get("config", {}).get("model", {})
    vae = VAE(hidden_dim=m.get("hidden_dim", 512),
              latent_dim=m.get("latent_dim", 16)).to(device)
    vae.load_state_dict(obj["model"])
    vae.eval()
    return vae


def load_diffusion(ckpt_path, device):
    """按 ckpt 里存的 config 重建条件去噪器 + 扩散, 灌入训练好的权重"""
    obj = torch.load(ckpt_path, map_location="cpu")
    d = obj["config"]["diffusion"]
    lat = obj["config"].get("latent_dim", None)
    denoiser = CondLatentDenoiser(latent_dim=lat or 16,
                                  hidden_dim=d.get("hidden_dim", 128),
                                  time_dim=d.get("time_dim", 64),
                                  num_classes=d.get("num_classes", 10)).to(device)
    denoiser.load_state_dict(obj["model"])
    denoiser.eval()
    diff = CondGaussianDiffusion(denoiser, timesteps=d.get("timesteps", 100)).to(device)
    print(f"[infer_latdiff_cond] 已加载 {ckpt_path} "
          f"(timesteps={d.get('timesteps', 100)}, num_classes={d.get('num_classes', 10)}, "
          f"参数量 {sum(p.numel() for p in denoiser.parameters())/1e6:.3f}M)")
    return diff


def save_grid(tensor, path, nrow, cell=28, pad=1):
    """把 (n, 784) 的张量拼成 nrow 列的网格图存盘 (纯 PIL)"""
    n = tensor.shape[0]
    nrows = (n + nrow - 1) // nrow
    img = Image.new("L", (nrow * (cell + pad) + pad, nrows * (cell + pad) + pad), 0)
    for i in range(n):
        t = tensor[i].reshape(cell, cell).clamp(0, 1).cpu()  # GPU 张量先搬回 CPU 才能 .numpy()
        tile = Image.fromarray((t.numpy() * 255).astype("uint8"))
        r, c = divmod(i, nrow)
        img.paste(tile, (pad + c * (cell + pad), pad + r * (cell + pad)))
    img.save(path)
    print(f"[infer_latdiff_cond] 已保存 {path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", default=VAE_CKPT)
    parser.add_argument("--diff_ckpt", default=DIFF_CKPT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--per_class", type=int, default=10, help="每个数字生成几张")
    parser.add_argument("--digit", type=int, default=-1,
                        help="只生成指定数字(0-9); -1 表示 10 个数字全出(10行网格)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子, 换个值换一批笔迹")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    vae = load_vae(args.vae_ckpt, device)
    diff = load_diffusion(args.diff_ckpt, device)
    num_classes = diff.model.num_classes

    # ---- 组装条件 y: 决定每张图生成哪个数字 ----
    if args.digit >= 0:
        if not 0 <= args.digit < num_classes:
            raise SystemExit(f"[infer_latdiff_cond] --digit 必须在 0-{num_classes-1} 之间")
        ys = torch.full((args.per_class,), args.digit, dtype=torch.long)
        fname = f"digit{args.digit}.png"
    else:
        # 每行一个数字: [0 x per_class, 1 x per_class, ..., 9 x per_class]
        ys = torch.arange(num_classes).repeat_interleave(args.per_class)
        fname = "cond_grid.png"
    ys = ys.to(device)

    # ---- 条件生成: 指定 y + 纯噪声 -> 去噪 -> z -> decode -> 图 ----
    z = diff.sample(ys, device)
    img = vae.decode(z)
    save_grid(img, os.path.join(args.out_dir, fname), nrow=args.per_class)

    print(f"[infer_latdiff_cond] 生成 {len(ys)} 张, 指定数字标签: {ys.cpu().tolist()}")
    print(f"[infer_latdiff_cond] z 统计: 均值={z.mean():.3f} 标准差={z.std():.3f}")
    print("[infer_latdiff_cond] 完成: 每行应为同一数字 => 条件生效")


if __name__ == "__main__":
    main()
