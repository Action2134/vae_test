"""
VAE 训练脚本 (MNIST)
用法: python src/train.py --config configs/train_vae.yaml
训练结束后自动在 output_dir 生成效果图并保存模型
"""
import os
import time
import yaml
import torch
import argparse

from vae import VAE, vae_loss
from vae_conv import ConvVAE
from vae_cvae import CVAE
from dataset import make_loader
from visualize import generate_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    # 登录节点限 CPU 线程: 小模型开满 20+ 线程反而更慢(调度开销), 还挤占共享机器
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}, cpu_threads={torch.get_num_threads()}")

    t = cfg["training"]
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- 数据 ----
    train_loader = make_loader(cfg["data_dir"], t["batch_size"], train=True)
    test_loader = make_loader(cfg["data_dir"], t["batch_size"],
                              train=False, shuffle=False)
    print(f"[train] 训练集 {len(train_loader.dataset)} 张, "
          f"测试集 {len(test_loader.dataset)} 张")

    # ---- 模型 (type: fc=全连接 / conv=卷积 / cvae=条件VAE) ----
    m = cfg["model"]
    mtype = m.get("type", "fc")
    model_cls = {"conv": ConvVAE, "cvae": CVAE}.get(mtype, VAE)
    model = model_cls(image_size=784, hidden_dim=m["hidden_dim"],
                      latent_dim=m["latent_dim"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] type={mtype}, latent_dim={m['latent_dim']}, "
          f"参数量 {n_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    max_steps = t.get("max_steps", -1)  # >0 时提前终止, 用于调试
    log_steps = t.get("log_steps", 50)

    # ---- 训练 ----
    global_step = 0
    for epoch in range(t["epochs"]):
        model.train()
        ep_loss = ep_recon = ep_kl = 0.0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            if mtype == "cvae":  # 条件模型把标签也喂进去
                recon, mu, logvar = model(x, y.to(device))
            else:
                recon, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss(
                recon, x, mu, logvar, beta=t.get("kl_beta", 1.0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_recon += recon_l.item()
            ep_kl += kl_l.item()
            global_step += 1

            if global_step % log_steps == 0:
                print(f"  step {global_step}: loss={loss.item():.1f} "
                      f"recon={recon_l.item():.1f} kl={kl_l.item():.1f}")
            if 0 < max_steps <= global_step:
                break

        n_batch = len(train_loader)
        print(f"[train] epoch {epoch+1}/{t['epochs']} "
              f"loss={ep_loss/n_batch:.1f} recon={ep_recon/n_batch:.1f} "
              f"kl={ep_kl/n_batch:.1f} time={time.time()-t0:.0f}s")
        if 0 < max_steps <= global_step:
            print(f"[train] 达到 max_steps={max_steps}, 提前结束")
            break

    # ---- 保存模型 + 生成效果图 ----
    save_path = os.path.join(cfg["output_dir"], "vae_mnist.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, save_path)
    print(f"[train] 模型已保存 {save_path}")

    # 可视化用小 batch, 不走 drop_last 的测试 loader
    # 效果图存 image_dir (默认回退到 output_dir), ckpt 目录只放 .pt
    # cvae 的 decode 需要条件, 无条件效果图没意义, 改用 my_infer.py --mode cond
    if mtype == "cvae":
        print("[train] cvae 跳过无条件效果图, 看 my_infer.py --mode cond")
    else:
        image_dir = cfg.get("image_dir", cfg["output_dir"])
        viz_loader = make_loader(cfg["data_dir"], 32, train=False, shuffle=False)
        generate_all(model, viz_loader, image_dir)
    print("[train] 完成")


if __name__ == "__main__":
    main()
