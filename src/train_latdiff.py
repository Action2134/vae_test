"""
Latent Diffusion 训练脚本 (MNIST)
两阶段设计的第二阶段: 冻结训好的 VAE, 只在它的 16 维 latent 空间里训去噪器
用法: python src/train_latdiff.py --config configs/train_latdiff.yaml
"""
import os
import time
import yaml
import torch
import argparse

from vae import VAE
from dataset import make_loader
from diffusion_latent import LatentDenoiser, GaussianDiffusion


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
    print(f"[latdiff] 已加载冻结 VAE: {ckpt_path} (latent_dim={vae.latent_dim})")
    return vae


@torch.no_grad()
def encode_dataset(vae, loader, device):
    """全量数据集过一遍 encoder, 取雾心 mu 作为扩散的训练目标"""
    zs = []
    for x, _ in loader:
        mu, _ = vae.encode(x.to(device, non_blocking=True))
        zs.append(mu.cpu())
    zs = torch.cat(zs)
    print(f"[latdiff] 数据集已压成 latent: {tuple(zs.shape)}, "
          f"均值={zs.mean():.3f} 标准差={zs.std():.3f}")
    return zs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[latdiff] device={device}, cpu_threads={torch.get_num_threads()}")

    t = cfg["training"]
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- 第一阶段产物: 冻结的 VAE, 把 6 万张图压成 6 万个 16 维点 ----
    vae = load_frozen_vae(cfg["vae_ckpt"], device)
    enc_loader = make_loader(cfg["data_dir"], 1024, train=True, shuffle=False)
    latents = encode_dataset(vae, enc_loader, device)

    # ---- 第二阶段: 在这些点上训去噪器 ----
    d = cfg["diffusion"]
    denoiser = LatentDenoiser(latent_dim=vae.latent_dim,
                              hidden_dim=d.get("hidden_dim", 128),
                              time_dim=d.get("time_dim", 64)).to(device)
    diff = GaussianDiffusion(denoiser, timesteps=d.get("timesteps", 100)).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"[latdiff] 去噪器参数量 {n_params/1e6:.3f}M, "
          f"timesteps={d.get('timesteps', 100)}")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(latents),
        batch_size=t["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=t["learning_rate"])

    global_step = 0
    for epoch in range(t["epochs"]):
        denoiser.train()
        ep_loss, t0 = 0.0, time.time()
        for (z0,) in loader:
            z0 = z0.to(device, non_blocking=True)
            loss = diff.training_loss(z0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            global_step += 1
            if global_step % t.get("log_steps", 100) == 0:
                print(f"  step {global_step}: loss={loss.item():.4f}")
        print(f"[latdiff] epoch {epoch+1}/{t['epochs']} "
              f"loss={ep_loss/len(loader):.4f} time={time.time()-t0:.1f}s")

    save_path = os.path.join(cfg["output_dir"], "latdiff_mnist.pt")
    cfg["latent_dim"] = vae.latent_dim  # 存进 config, 推理重建去噪器用
    torch.save({"model": denoiser.state_dict(), "config": cfg}, save_path)
    print(f"[latdiff] 模型已保存 {save_path}")
    print("[latdiff] 完成, 出图用 src/infer_latdiff.py")


if __name__ == "__main__":
    main()
