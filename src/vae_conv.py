"""
卷积版 VAE (MNIST 28x28)
与 vae.py 的全连接版接口完全一致: encode / decode / forward, 输入输出都是 784 维
卷积负责抓局部笔画结构, 解决全连接版生成图发糊的问题
"""
import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    """
    encoder: 28x28 --conv s2--> 14x14 --conv s2--> 7x7 -> flatten -> (mu, logvar)
    decoder: latent -> fc -> 7x7 --deconv s2--> 14x14 --deconv s2--> 28x28 (sigmoid)
    hidden_dim 控制中间全连接层宽度
    """

    def __init__(self, image_size=784, hidden_dim=256, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = 28  # 固定 MNIST 尺寸
        self.enc_channels = 64  # 最后一层卷积输出通道数

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),   # 28 -> 14
            nn.Conv2d(32, self.enc_channels, 3, stride=2, padding=1), nn.ReLU(),  # 14 -> 7
        )
        enc_flat = self.enc_channels * 7 * 7
        self.enc_fc = nn.Sequential(nn.Linear(enc_flat, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, enc_flat), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.enc_channels, 32, 3, stride=2,
                               padding=1, output_padding=1), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(32, 1, 3, stride=2,
                               padding=1, output_padding=1), nn.Sigmoid(),  # 14 -> 28
        )

    def encode(self, x):
        # x: (B, 784) -> (B, 1, 28, 28)
        h = self.encoder(x.view(-1, 1, self.img_size, self.img_size))
        h = self.enc_fc(h.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        # 训练时 z = mu + std * eps; 推理时直接取 mu
        if torch.is_grad_enabled():
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, self.enc_channels, 7, 7)
        return self.decoder(h).flatten(1)  # (B, 784)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
