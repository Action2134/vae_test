"""
Latent Diffusion 核心: 在 VAE 的 16 维 latent 空间里做扩散
这就是 Stable Diffusion 的骨架: VAE 管 图<->latent 压缩, diffusion 管在 latent 里"无中生有"

前向扩散 (加噪):  z0 -> z1 -> ... -> zT   每步加一点高斯噪声, zT 约等于纯噪声
反向去噪 (生成):  zT -> ... -> z1 -> z0   模型每步预测"上一步混进去的噪声", 减掉它
训练目标只有一个: 给一张被污染到第 t 步的 z_t, 把混进去的噪声预测出来 (MSE)
"""
import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """时间步 t -> 连续向量: 正弦位置编码 + 两层 MLP (SD 里也是这套)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t):
        # 正弦编码: 每个频率分量一对 (sin, cos)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        return self.mlp(emb)


class LatentDenoiser(nn.Module):
    """
    去噪网络 (epsilon-prediction): 输入 (z_t, t), 输出预测的噪声
    latent 只有 16 维, 一个带时间条件的 MLP 就够了
    """

    def __init__(self, latent_dim=16, hidden_dim=128, time_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t):
        temb = self.time_emb(t)                    # (n, time_dim)
        return self.net(torch.cat([z_t, temb], dim=1))  # (n, latent_dim)


class GaussianDiffusion(nn.Module):
    """
    DDPM: 管理加噪/去噪的全部数学
    所有 beta/alpha 序列在 __init__ 里预计算并注册为 buffer (随模型一起搬设备)
    """

    def __init__(self, model, timesteps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        # 线性噪声调度: beta 从 1e-4 线性爬到 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)          # alpha_bar: 累积留存率
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas", alphas.sqrt())
        self.register_buffer("sqrt_abar", abar.sqrt())
        self.register_buffer("sqrt_one_minus_abar", (1.0 - abar).sqrt())
        self.register_buffer("coef", betas / (1.0 - abar).sqrt())

    def q_sample(self, z0, t, noise):
        """前向扩散: 一步跳到第 t 步  z_t = sqrt(abar)*z0 + sqrt(1-abar)*noise"""
        sa = self.sqrt_abar[t].unsqueeze(1)
        sn = self.sqrt_one_minus_abar[t].unsqueeze(1)
        return sa * z0 + sn * noise

    def training_loss(self, z0):
        """训练一步: 随机抽时间步和噪声, 让模型把噪声猜出来"""
        n = z0.shape[0]
        t = torch.randint(0, self.timesteps, (n,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        pred = self.model(z_t, t)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n, device):
        """反向去噪: 从纯噪声出发, 逐步把噪声减掉, 走出一个干净的 z"""
        z = torch.randn(n, self.model.latent_dim, device=device)
        for t in range(self.timesteps - 1, -1, -1):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = self.model(z, tt)                        # 预测混进来的噪声
            # z_{t-1} = 1/sqrt(alpha) * (z_t - coef * eps) + 噪声项 (最后一步不加)
            z = (z - self.coef[t] * eps) / self.sqrt_alphas[t]
            if t > 0:
                z = z + self.betas[t].sqrt() * torch.randn_like(z)
        return z
