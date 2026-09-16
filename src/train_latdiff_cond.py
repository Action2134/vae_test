"""
条件 Latent Diffusion 训练脚本 (MNIST, class-conditional)
和 train_latdiff.py 的唯一区别: 编码时保留数字标签 y, 训练去噪器时把 (z, y) 一起喂
    => 训出的去噪器能"指定数字"生成
用法: python src/train_latdiff_cond.py --config configs/train_latdiff_cond.yaml
"""
import os
import time
import argparse

import yaml
import torch

from vae import VAE
from dataset import make_loader
from diffusion_latent_cond import CondLatentDenoiser, CondGaussianDiffusion


def load_frozen_vae(ckpt_path, device):
    """加载训好的 VAE 并冻结: 只借它的 encoder 把数据集压成 latent"""
    obj = torch.load(ckpt_path, map_location="cpu")
    m = obj.get("config", {}).get("model", {})
    vae = VAE(hidden_dim=m.get("hidden_dim", 512),
              latent_dim=m.get("latent_dim", 16)).to(device)
    vae.load_state_dict(obj["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[latdiff_cond] 已加载冻结 VAE: {ckpt_path} (latent_dim={vae.latent_dim})")
    return vae


@torch.no_grad()
def encode_dataset(vae, loader, device):
    """全量数据集过一遍 encoder, 取雾心 mu 作为扩散训练目标; 同时保留标签 y"""
    zs, ys = [], []
    for x, y in loader:                              # ← 保留标签(无条件版这里是 for x, _)
        mu, _ = vae.encode(x.to(device, non_blocking=True))
        zs.append(mu.cpu())
        ys.append(y)                                 # ← 收集标签
    zs = torch.cat(zs)
    ys = torch.cat(ys)
    print(f"[latdiff_cond] 数据集已压成 latent: {tuple(zs.shape)}, "
          f"标签: {tuple(ys.shape)}, 均值={zs.mean():.3f} 标准差={zs.std():.3f}")
    return zs, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[latdiff_cond] device={device}, cpu_threads={torch.get_num_threads()}")

    t = cfg["training"]
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- 第一阶段产物: 冻结的 VAE, 把 6 万张图压成 6 万个 16 维点 (连标签一起留) ----
    vae = load_frozen_vae(cfg["vae_ckpt"], device)
    enc_loader = make_loader(cfg["data_dir"], 1024, train=True, shuffle=False)
    latents, labels = encode_dataset(vae, enc_loader, device)

    # ---- 第二阶段: 在这些 (点, 标签) 上训条件去噪器 ----
    d = cfg["diffusion"]
    num_classes = d.get("num_classes", 10)
    denoiser = CondLatentDenoiser(latent_dim=vae.latent_dim,
                                  hidden_dim=d.get("hidden_dim", 128),
                                  time_dim=d.get("time_dim", 64),
                                  num_classes=num_classes).to(device)
    diff = CondGaussianDiffusion(denoiser, timesteps=d.get("timesteps", 100)).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"[latdiff_cond] 条件去噪器参数量 {n_params/1e6:.3f}M, "
          f"timesteps={d.get('timesteps', 100)}, num_classes={num_classes}")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(latents, labels),   # ← 数据集带上标签
        batch_size=t["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=t["learning_rate"])

    global_step = 0
    for epoch in range(t["epochs"]):
        denoiser.train()
        ep_loss, t0 = 0.0, time.time()
        for z0, y0 in loader:                          # ← 解包出 (z0, y0)
            z0 = z0.to(device, non_blocking=True)
            y0 = y0.to(device, non_blocking=True)
            loss = diff.training_loss(z0, y0)          # ← 传 y0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            global_step += 1
            if global_step % t.get("log_steps", 100) == 0:
                print(f"  step {global_step}: loss={loss.item():.4f}")
        print(f"[latdiff_cond] epoch {epoch+1}/{t['epochs']} "
              f"loss={ep_loss/len(loader):.4f} time={time.time()-t0:.1f}s")

    save_path = os.path.join(cfg["output_dir"], "latdiff_cond_mnist.pt")
    cfg["latent_dim"] = vae.latent_dim                 # 存进 config, 推理重建去噪器用
    torch.save({"model": denoiser.state_dict(), "config": cfg}, save_path)
    print(f"[latdiff_cond] 模型已保存 {save_path}")
    print("[latdiff_cond] 完成, 出图用 src/infer_latdiff_cond.py")


if __name__ == "__main__":
    main()
