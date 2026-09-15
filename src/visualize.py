"""
效果可视化 (纯 PIL 实现, 不依赖 matplotlib)
生成三张效果图:
  reconstruction.png  原图 vs 重建图 (上下两行对比)
  samples.png         从 N(0,1) 采样 latent 解码出的新数字
  interpolation.png   两张图在 latent 空间的插值过渡
"""
import os

import torch
from PIL import Image


def _grid_to_image(tensor, nrow, cell=28, pad=1):
    """(N, 784) 张量拼成灰度大图"""
    n = tensor.shape[0]
    ncol = nrow
    nrows_grid = (n + ncol - 1) // ncol
    w = ncol * (cell + pad) + pad
    h = nrows_grid * (cell + pad) + pad
    img = Image.new("L", (w, h), 0)
    for i in range(n):
        t = tensor[i].reshape(cell, cell).clamp(0, 1)
        tile = Image.fromarray((t.cpu().numpy() * 255).astype("uint8"))
        r, c = divmod(i, ncol)
        img.paste(tile, (pad + c * (cell + pad), pad + r * (cell + pad)))
    return img


@torch.no_grad()
def save_reconstruction(model, images, path, n=16):
    """上排原图, 下排重建图"""
    model.eval()
    x = images[:n]
    recon, _, _ = model(x)
    grid = torch.cat([x, recon], dim=0)
    _grid_to_image(grid, nrow=n).save(path)


@torch.no_grad()
def save_samples(model, path, grid=12, seed=0):
    """latent ~ N(0,1) 随机采样解码"""
    model.eval()
    # CPU 发生器采样后搬到模型所在设备: CUDA Generator 兼容性差, 这样 CPU/GPU 都能跑
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(grid * grid, model.latent_dim, generator=g)
    z = z.to(next(model.parameters()).device)
    _grid_to_image(model.decode(z), nrow=grid).save(path)


@torch.no_grad()
def save_interpolation(model, img_a, img_b, path, steps=10):
    """两张图的 latent 之间线性插值, 输出过渡序列"""
    model.eval()
    mu_a, _ = model.encode(img_a.unsqueeze(0))
    mu_b, _ = model.encode(img_b.unsqueeze(0))
    alphas = torch.linspace(0, 1, steps, device=mu_a.device).view(-1, 1)
    z = alphas * mu_b + (1 - alphas) * mu_a
    _grid_to_image(model.decode(z), nrow=steps).save(path)


def generate_all(model, test_loader, output_dir):
    """训练结束后一次性生成全部效果图"""
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    x, y = next(iter(test_loader))
    x = x.to(device)

    save_reconstruction(model, x, os.path.join(output_dir, "reconstruction.png"))
    save_samples(model, os.path.join(output_dir, "samples.png"))
    # 取 batch 里第一张和最后一张做插值
    save_interpolation(model, x[0], x[-1],
                       os.path.join(output_dir, "interpolation.png"))
    print(f"[visualize] 效果图已保存到 {output_dir}")
