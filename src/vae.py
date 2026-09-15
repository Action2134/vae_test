"""
VAE 模型定义
经典全连接 VAE: 编码器 -> (mu, logvar) -> 重参数化采样 -> 解码器
"""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    MNIST VAE (28x28 灰度图)
    encoder: 784 -> hidden -> hidden -> (mu, logvar), 各 latent_dim 维
    decoder: latent_dim -> hidden -> hidden -> 784 (sigmoid 输出概率)
    """

    def __init__(self, image_size=784, hidden_dim=512, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(image_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_size),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        # 训练时 z = mu + std * eps; 推理时直接取 mu
        if torch.is_grad_enabled():
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    VAE ELBO = 重建项(BCE sum) + beta * KL 散度
    按样本求和后取 batch 均值; 返回 (总loss, 重建项, KL项)
    """
    recon_loss = nn.functional.binary_cross_entropy(
        recon, x, reduction="none").sum(dim=1)
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    loss = (recon_loss + beta * kl_loss).mean()
    return loss, recon_loss.mean(), kl_loss.mean()
