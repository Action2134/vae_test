"""
条件 VAE (Conditional VAE) 模型定义
和普通 VAE 的唯一区别: encoder 和 decoder 都额外吃一个"条件" y (数字 0-9)
    encoder: (图, y) -> 分布参数      编码时告诉模型"这是几"
    decoder: (z, y) -> 图            生成时命令模型"给我画个几"
条件注入方式: one-hot 后直接 concat (最朴素, 也是理解 T2I 条件注入的起点)
loss 直接复用 vae.py 的 vae_loss (它只认张量, 不关心条件)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """
    MNIST 条件 VAE (28x28 灰度图, 条件 = 数字类别)
    encoder: 784+num_classes -> hidden -> hidden -> (mu, logvar)
    decoder: latent_dim+num_classes -> hidden -> hidden -> 784 (sigmoid)
    """

    def __init__(self, image_size=784, hidden_dim=512, latent_dim=16,
                 num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # 条件 one-hot 的维度, encoder/decoder 都要拼它
        self.encoder = nn.Sequential(
            nn.Linear(image_size + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_size),
            nn.Sigmoid(),
        )

    def onehot(self, y):
        """标签 (B,) long -> one-hot (B, num_classes) float"""
        return F.one_hot(y, self.num_classes).float()

    def encode(self, x, y):
        # 图和条件拼一起进 encoder
        h = self.encoder(torch.cat([x, self.onehot(y)], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        if torch.is_grad_enabled():
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, y):
        # z 和条件拼一起进 decoder: "在这颗随机种子上, 给我画个 y"
        return self.decoder(torch.cat([z, self.onehot(y)], dim=1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
