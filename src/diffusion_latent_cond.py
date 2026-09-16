"""
条件 Latent Diffusion (class-conditional) —— 在实验5无条件版基础上加"数字条件"
去噪器从 (z_t, t) 升级为 (z_t, t, y): y 是要生成的数字(0-9), one-hot 后拼进入口
    无条件版: 纯噪声 -> 去噪 -> z -> 图      (随机出数字)
    条件版:   指定y + 纯噪声 -> 去噪 -> z -> 图  (指定出哪个数字)
条件注入方式: one-hot concat, 和实验4 CVAE 完全一致, 也是通往文字生图的第一级台阶
复用实验5的 TimeEmbedding 和扩散调度数学, 只在去噪器入口多拼一个 one-hot(y)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_latent import TimeEmbedding, GaussianDiffusion


class CondLatentDenoiser(nn.Module):
    """
    条件去噪网络 (epsilon-prediction): 输入 (z_t, t, y), 输出预测的噪声
    和实验5 LatentDenoiser 的唯一区别: 入口多拼一个 one-hot(y)
        拼接维度: latent(16) + time(64) + onehot(10) = 90
    """

    def __init__(self, latent_dim=16, hidden_dim=128, time_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim + num_classes, hidden_dim),  # 16+64+10=90
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def onehot(self, y):
        """标签 (n,) long -> one-hot (n, num_classes) float (和 CVAE 一致)"""
        return F.one_hot(y, self.num_classes).float()

    def forward(self, z_t, t, y):
        temb = self.time_emb(t)                              # (n, time_dim)
        yemb = self.onehot(y).to(z_t.device)                 # (n, num_classes)
        return self.net(torch.cat([z_t, temb, yemb], dim=1))  # (n, latent_dim)


class CondGaussianDiffusion(GaussianDiffusion):
    """
    继承实验5的扩散数学: __init__(调度系数) 和 q_sample(前向加噪) 原样复用
    只重写 training_loss 和 sample, 让它们把条件 y 一路带给去噪器
    """

    def training_loss(self, z0, y):
        """训练一步: 和无条件版逻辑相同, 只是把 y 一起喂给去噪器"""
        n = z0.shape[0]
        t = torch.randint(0, self.timesteps, (n,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        pred = self.model(z_t, t, y)                         # ← 比无条件版多传 y
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, y, device):
        """
        条件生成: 从纯噪声出发去噪 timesteps 步, 每步都告诉去噪器"要画哪个数字 y"
        y: (n,) long, 每个元素指定对应这张图生成几 => 生成什么数字完全由 y 决定
        """
        n = y.shape[0]
        z = torch.randn(n, self.model.latent_dim, device=device)
        for t in range(self.timesteps - 1, -1, -1):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = self.model(z, tt, y)                       # ← 比无条件版多传 y
            z = (z - self.coef[t] * eps) / self.sqrt_alphas[t]
            if t > 0:
                z = z + self.betas[t].sqrt() * torch.randn_like(z)
        return z
