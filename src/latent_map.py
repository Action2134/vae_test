"""
latent_dim=2 流形可视化 (自包含, 纯 PIL 画图不依赖 matplotlib)
两张图:
  scatter.png   把 1 万张测试图 encode 成平面上的点, 按数字 0-9 上色 -> latent 地图
  manifold.png  在地图范围内铺 24x24 网格逐点 decode -> 平面上每个位置长什么样

用法:
    python latent_map.py                     # 用默认 ckpt (latent2)
    python latent_map.py --ckpt /path/x.pt   # 换 checkpoint
"""
import argparse
import gzip
import os
import sys

import torch
from PIL import Image, ImageDraw

# 模型结构从定义文件导入 (type 从 ckpt 里存的 config 自动识别)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae import VAE
from vae_conv import ConvVAE

DATA_DIR = "/public/home/liuhuan/workspace_xd/data/MNIST/raw"
DEFAULT_CKPT = "/public/home/liuhuan/workspace_xd/ckpt/vae_test/latent2/vae_mnist.pt"
OUT_DIR = "/public/home/liuhuan/workspace_xd/vae_test/output/images/latent2"

# 数字 0-9 的散点颜色 (R,G,B)
COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
]


def load_model(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    saved_cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    m = saved_cfg.get("model", {})
    mtype = m.get("type", "fc")
    model_cls = ConvVAE if mtype == "conv" else VAE
    # 必须用 ckpt 里存的 latent_dim/hidden_dim 构建, 否则权重形状对不上
    model = model_cls(hidden_dim=m.get("hidden_dim", 512),
                      latent_dim=m.get("latent_dim", 2))
    state = obj
    if isinstance(obj, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in obj:
                state = obj[key]
                break
    model.load_state_dict(state)
    model.eval()
    print(f"[latent_map] 已加载 {ckpt_path} (type={mtype}, latent_dim={model.latent_dim})")
    if model.latent_dim != 2:
        print("[latent_map] 警告: latent_dim 不是 2, 散点图无法绘制")
    return model


def load_test_set():
    """手写解析 MNIST 测试集: 图片 + 标签"""
    with gzip.open(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        images = torch.frombuffer(f.read(), dtype=torch.uint8, offset=16)
    images = images.reshape(-1, 784).float().clone() / 255.0
    with gzip.open(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        labels = torch.frombuffer(f.read(), dtype=torch.uint8, offset=8).clone()
    return images, labels.long()


@torch.no_grad()
def encode_all(model, images, batch=512):
    """分批 encode 全部测试图, 返回 mu (N, 2)"""
    mus = []
    for i in range(0, len(images), batch):
        mu, _ = model.encode(images[i:i + batch])
        mus.append(mu)
    return torch.cat(mus)


def draw_scatter(mus, labels, path, size=720):
    """把 (N,2) 的点按 1%/99% 分位裁剪到画布, 按数字上色画散点"""
    lo = torch.quantile(mus, 0.01, dim=0)
    hi = torch.quantile(mus, 0.99, dim=0)
    norm = (mus - lo) / (hi - lo).clamp(min=1e-6)  # 归一化到 [0,1] 附近

    img = Image.new("RGB", (size, size), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    for (x, y), lab in zip(norm.tolist(), labels.tolist()):
        px = int(max(0, min(1, x)) * (size - 1))
        py = int(max(0, min(1, 1 - y)) * (size - 1))  # y 轴翻转, 符合数学坐标
        c = COLORS[lab]
        draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=c)
    img.save(path)
    print(f"[latent_map] 散点图已保存 {path} ({len(mus)} 个点)")
    return lo, hi


@torch.no_grad()
def draw_manifold(model, lo, hi, path, grid=24, cell=28, pad=1):
    """在地图范围铺 grid x grid 个 z, 逐点 decode 拼成大图"""
    xs = torch.linspace(lo[0], hi[0], grid)
    ys = torch.linspace(lo[1], hi[1], grid)
    zz = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    out = model.decode(zz)

    w = h = grid * (cell + pad) + pad
    img = Image.new("L", (w, h), 0)
    for i in range(grid * grid):
        t = out[i].reshape(cell, cell).clamp(0, 1)
        tile = Image.fromarray((t.numpy() * 255).astype("uint8"))
        r, c = divmod(i, grid)
        img.paste(tile, (pad + c * (cell + pad), pad + r * (cell + pad)))
    img.save(path)
    print(f"[latent_map] 流形网格已保存 {path} ({grid}x{grid} 个采样点)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.ckpt)
    images, labels = load_test_set()
    print(f"[latent_map] 编码 {len(images)} 张测试图 ...")
    mus = encode_all(model, images)
    lo, hi = draw_scatter(mus, labels, os.path.join(args.out_dir, "scatter.png"))
    draw_manifold(model, lo, hi, os.path.join(args.out_dir, "manifold.png"))


if __name__ == "__main__":
    main()
