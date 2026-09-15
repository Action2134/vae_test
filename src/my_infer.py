"""
自包含 infer 脚本: 模型定义 + 加载 + 推理全在这一个文件里, 不 import 项目其他代码
随便改, 不影响训练流程

用法:
    python my_infer.py                          # 用默认 ckpt, 出全部三张图
    python my_infer.py --ckpt /path/to/x.pt     # 指定 checkpoint
    python my_infer.py --mode sample --n 10     # 只采样 10 张

输出: workspace_xd/vae_test/output/images/my_infer/
"""
import argparse
import os
import sys

import torch
from PIL import Image

# 模型结构从定义文件导入 (type 从 ckpt 里存的 config 自动识别: fc/conv/cvae)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae import VAE
from vae_conv import ConvVAE
from vae_cvae import CVAE

# ---- 数据 ----
DATA_DIR = "/public/home/liuhuan/workspace_xd/data/MNIST/raw"
# ---- 默认 checkpoint (1步训练的 debug 模型) ----
DEFAULT_CKPT = "/public/home/liuhuan/workspace_xd/ckpt/vae_test/debug/vae_mnist.pt"
# ---- 输出目录 (效果图不进 ckpt, ckpt 只放 .pt) ----
OUT_DIR = "/public/home/liuhuan/workspace_xd/vae_test/output/images/my_infer"


# ========== 1. 加载模型 (结构从定义文件导入, 类型从 ckpt config 自动识别) ==========
MODEL_CLS = {"conv": ConvVAE, "cvae": CVAE}


def load_model(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    # ckpt 里随权重存了训练 config, 读出来判断类型 (旧 ckpt 没有就默认 fc)
    saved_cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    m = saved_cfg.get("model", {})
    mtype = m.get("type", "fc")
    model_cls = MODEL_CLS.get(mtype, VAE)
    # 按 ckpt 里存的 hidden_dim/latent_dim 构建, 兼容 2 维等非常规模型
    model = model_cls(hidden_dim=m.get("hidden_dim", 512),
                      latent_dim=m.get("latent_dim", 16))
    # 兼容两种保存方式: 直接存 state_dict / 存了 {"model": state_dict, ...} 的字典
    state = obj
    if isinstance(obj, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in obj:
                state = obj[key]
                break
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[my_infer] 已加载 {ckpt_path} (type={mtype}, 参数量 {n_params/1e6:.2f}M)")
    return model, mtype


# ========== 2. 读几张测试图 (手写解析 MNIST, 不依赖 torchvision) ==========
def load_test_images(n=16):
    import gzip
    with gzip.open(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        data = f.read()
    images = torch.frombuffer(data, dtype=torch.uint8, offset=16)
    images = images.reshape(-1, 784).float()[:n] / 255.0
    print(f"[my_infer] 读入 {n} 张 MNIST 测试图")
    return images


# ========== 3. 拼图保存 (纯 PIL) ==========
def save_grid(tensor, path, nrow, cell=28, pad=1):
    n = tensor.shape[0]
    nrows = (n + nrow - 1) // nrow
    img = Image.new("L", (nrow * (cell + pad) + pad, nrows * (cell + pad) + pad), 0)
    for i in range(n):
        t = tensor[i].reshape(cell, cell).clamp(0, 1)
        tile = Image.fromarray((t.numpy() * 255).astype("uint8"))
        r, c = divmod(i, nrow)
        img.paste(tile, (pad + c * (cell + pad), pad + r * (cell + pad)))
    img.save(path)
    print(f"[my_infer] 已保存 {path}")


# ========== 4. 三种推理, 想改哪个改哪个 ==========
@torch.no_grad()
def infer_reconstruction(model, images, path):
    """重建: 原图 -> encoder -> mu -> decoder (上排原图, 下排重建)"""
    mu, _ = model.encode(images)
    recon = model.decode(mu)
    save_grid(torch.cat([images, recon]), path, nrow=len(images))


@torch.no_grad()
def infer_sample(model, n, path, seed=0):
    """采样生成: z ~ N(0,1) -> decoder"""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, model.latent_dim, generator=g)
    save_grid(model.decode(z), path, nrow=n)


@torch.no_grad()
def infer_interpolate(model, img_a, img_b, path, steps=10):
    """插值: 两张图的 mu 之间连线 -> decoder"""
    mu_a, _ = model.encode(img_a.unsqueeze(0))
    mu_b, _ = model.encode(img_b.unsqueeze(0))
    alphas = torch.linspace(0, 1, steps).view(-1, 1)
    save_grid(model.decode(alphas * mu_b + (1 - alphas) * mu_a), path, nrow=steps)


@torch.no_grad()
def infer_conditional(model, path, per_class=10, seed=0, digit=None):
    """条件生成: 每行命令模型生成一个数字, 每行 per_class 张
    同一列用同一个 z, 能直观看到 z 管笔迹风格, y 管写哪个数字
    digit=None: 10 个数字全出; digit=9: 只生成数字 9 这一行"""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(per_class, model.latent_dim, generator=g)  # 每列共享的 z
    ys = torch.tensor([digit]) if digit is not None else torch.arange(model.num_classes)
    nrows = len(ys)
    # 展开成 (行数*per_class): 行=数字, 列=z
    z_grid = z.unsqueeze(0).expand(nrows, -1, -1).reshape(-1, model.latent_dim)
    y_grid = ys.unsqueeze(1).expand(-1, per_class).reshape(-1)
    save_grid(model.decode(z_grid, y_grid), path, nrow=per_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--mode", default="all",
                        choices=["all", "recon", "sample", "interp", "cond"])
    parser.add_argument("--n", type=int, default=10, help="sample 模式生成张数")
    parser.add_argument("--digit", type=int, default=-1,
                        help="cond 模式只生成指定数字 (0-9); -1 表示 10 个数字全出")
    parser.add_argument("--seed", type=int, default=0, help="随机种子, 换个值换一批笔迹")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, mtype = load_model(args.ckpt)

    # cvae 只能走 cond 模式 (decode 需要条件); 其他模型不能走 cond
    if mtype == "cvae":
        if args.mode not in ("all", "cond"):
            raise SystemExit("[my_infer] cvae 模型的 decode 需要条件, 请用 --mode cond")
        if args.digit >= 0:
            if not 0 <= args.digit <= 9:
                raise SystemExit("[my_infer] --digit 必须在 0-9 之间")
            fname = f"digit{args.digit}.png"
        else:
            fname = "cond_grid.png"
        infer_conditional(model, os.path.join(args.out_dir, fname),
                          digit=args.digit if args.digit >= 0 else None, seed=args.seed)
        return
    if args.mode == "cond":
        raise SystemExit("[my_infer] --mode cond 只支持 cvae 模型的 ckpt")

    images = load_test_images(16)

    if args.mode in ("all", "recon"):
        infer_reconstruction(model, images, os.path.join(args.out_dir, "reconstruction.png"))
    if args.mode in ("all", "sample"):
        infer_sample(model, args.n if args.mode == "sample" else 100,
                     os.path.join(args.out_dir, "samples.png"))
    if args.mode in ("all", "interp"):
        infer_interpolate(model, images[0], images[-1],
                          os.path.join(args.out_dir, "interpolation.png"))


if __name__ == "__main__":
    main()
