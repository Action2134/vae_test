# 实验记录: MNIST 生成模型系列 —— 从基础 VAE 到 Latent Diffusion

> 环境: CentOS 7.4 + SLURM + Tesla V100-SXM2-32GB | 数据: MNIST 28×28 (6万训练/1万测试)
> 项目路径: `/public/home/liuhuan/workspace_xd/vae_test/` | ckpt: `/public/home/liuhuan/workspace_xd/ckpt/vae_test/`
> 终点目标: 理解文字生图 (T2I / Stable Diffusion)

## 0. 实验路线图

五个实验是一条递进的路线, 每一步只加一个新概念:

```
实验1 基础VAE      学会"压缩+生成": 图 ↔ 16维latent, randn 一步出图
实验2 卷积VAE      换骨干网络: MLP → CNN, 生成图更锐利
实验3 latent=2     把瓶颈压到2维, 可视化 latent 空间"地图"
实验4 CVAE         加条件: one-hot 注入, 从"随机出图"到"指定数字出图"
实验5 LatDiff      换生成引擎: randn 一步采 → diffusion 逐步去噪 (= SD 骨架)
实验6 CondLatDiff  给去噪器加数字条件: one-hot 注入, 从"随机出数字"到"指定出数字"
下一步             把 one-hot 换成文本向量 → 迷你文字生图
```

### 完成状态总览 (截至当前)

| 实验 | 状态 | SLURM job | ckpt | 效果图 |
|------|------|-----------|------|--------|
| 1 基础 VAE | ✅ | — | `vae_test/vae_mnist.pt` | `reconstruction/samples/interpolation.png` |
| 2 卷积 VAE | ✅ | — | `vae_test/conv/` | `conv/`, `my_infer_conv/` |
| 3 latent=2 流形 | ✅ | 212936 | `vae_test/latent2/` | `latent2/` |
| 4 CVAE 条件生成 | ✅ | 212949 (rc=0) | `vae_test/cvae/` | `my_infer_cvae/cond_grid.png` |
| 5 Latent Diffusion | ✅ | 213306 (rc=0) | `vae_test/latdiff/latdiff_mnist.pt` | `latdiff/samples_{diffusion,randn}.png` |
| 6 条件 LatDiff (指定数字) | ✅ | 213885 (rc=0) | `vae_test/latdiff_cond/latdiff_cond_mnist.pt` | `latdiff_cond/cond_grid.png` |
| 下一步 迷你文字生图 (T2I) | ⬜ 未开始 | — | — | — |

- ckpt 根目录 `/public/home/liuhuan/workspace_xd/ckpt/`, 效果图根目录 `output/images/`
- **版本管理**: 全部源码/配置/脚本/文档已提交并推送 `origin/main` (github.com/Action2134/vae_test), 工作区干净
- **无在跑任务**: vae_test 相关 SLURM job 均已结束
- **LatDiff 三件套分工**: `diffusion_latent.py` 造零件 (模型/算法) → `train_latdiff.py` 用零件训练 → `infer_latdiff.py` 拿训好的零件出图

---

## 实验1: 基础 VAE (全连接)

### 目的
理解 VAE 的核心机制: 编码器输出**分布**而非点, KL 项把 latent 空间压成标准正态, 使得随机采样也能生成合理图片。

### 模型结构 (`src/vae.py`)
```
encoder: 784 → Linear(512) → ReLU → 两个头: mu(16), logvar(16)
重参数化: z = mu + std × randn        (训练时; 推理时直接 return mu)
decoder: 16 → Linear(512) → ReLU → Linear(512) → ReLU → Linear(784) → Sigmoid
```

### 训练 (`src/train.py` + `configs/train_vae.yaml`)
- loss = **BCE 重建项 + kl_beta × KL 散度项**, kl_beta=1.0
- epochs=10, batch=128, lr=1e-3, seed=42
- 产物: `ckpt/vae_test/vae_mnist.pt`

### 结果与验证 (`output/images/`)
| 图 | 验证点 |
|----|--------|
| `reconstruction.png` | 上排原图/下排重建, 结构对得上 → encoder-decoder 通路有效 |
| `samples.png` | 纯 randn 采 z 出图, 是"数字的样子" → KL 压正态成功, latent 无空洞 |
| `interpolation.png` | 两图 latent 连线插值, 中间帧平滑过渡 → latent 空间连续 |
| `samples_untrained/`, `untrained_demo/` | 未训练模型出图是噪片 → 对照证明"生成能力是训出来的" |

### 关键结论
- **std 是模型的自信度**: 数字签名维度 std≈0.1, 留白维度 std≈0.98
- 推理时 `reparameterize` 走 `return mu` 分支取"雾心", 生成是**一次前向、零循环**
- latent 只有 16 维, 图的信息被压缩了 49 倍 (784→16)

---

## 实验2: 卷积 VAE

### 目的
换骨干网络: 全连接对"平移不变的笔画结构"不敏感, 卷积抓局部特征, 对比生成质量。

### 配置 (`src/vae_conv.py` + `configs/train_vae_conv.yaml`)
- type=conv, hidden_dim=256 (卷积版中为中间全连接层宽度), latent_dim=16
- **epochs=50**: 10 epoch 没训透、笔画发软; 卷积版收敛慢, 加大训练量
- 其余超参与全连接版对齐, 保证公平对比
- 产物: `ckpt/vae_test/conv/`, 图: `output/images/conv/`, `my_infer_conv/`

### 结论
卷积版生成图笔画更锐利; 但本项目 latent 只有 16 维, 骨干差异不是主线, 后续实验回到全连接版。

---

## 实验3: latent_dim=2 流形可视化

### 目的
不为重建清晰, 而是把 latent 压到 **2 维**, 让每张图变成平面上的一个点, 直接"看见" latent 空间的结构。

### 配置 (`configs/train_vae_latent2.yaml` + `src/latent_map.py`)
- latent_dim=2, epochs=20 (2 维瓶颈信息量小, 多训几轮让点云充分展开)
- 产物: `ckpt/vae_test/latent2/`, 图: `output/images/latent2/`

### 结论
点云图上**同数字聚成簇、相邻簇之间是过渡笔迹** —— 直观展示了"latent 空间有语义结构、且被 KL 压得连续无空洞", 这就是 randn 采样能出合理图的原因的可视化证据。

---

## 实验4: CVAE (条件 VAE, 指定数字生成)

### 目的
从"随机出图"升级到"**指定数字出图**" —— 条件注入的最小形态, 直接衔接 T2I (把 one-hot 换成文本向量就是文字生图)。

### 模型结构 (`src/vae_cvae.py`)
one-hot 条件在模型内部生成 (`F.one_hot`), **两处注入**:
```
encoder 入口: Linear(784+10 → 512)   编码时告知"这张图是几"
decoder 入口: Linear(16+10 → 512)    生成时命令"给我画几"
```

### 训练 (`configs/train_vae_cvae.yaml`, SLURM job 212949, COMPLETED rc=0)
- 超参与全连接版对齐只加条件输入; epochs=20, batch=128, lr=1e-3
- loss 仍是 BCE+KL, **没有显式分类项**: 但训练时条件与重建目标配对
  (喂 y=3 必须重建出 3), 画错数字=重建失败=重罚, "数字对错"是重建目标的副产品
- 产物: `ckpt/vae_test/cvae/vae_mnist.pt` (5.2M)

### 推理 (`src/my_infer.py --mode cond`)
- **只走 decoder, encoder 完全不被调用**; y 的来源从真实标签变成用户指定
- 支持 `--digit 8` 单数字生成、`--seed` 换笔迹

### 结果验证 (`output/images/my_infer_cvae/cond_grid.png`, 10×10 网格)
| 验证点 | 现象 | 结论 |
|--------|------|------|
| 每行同数字 | 行=条件 y | 条件注入生效 |
| 同列笔迹相似 | 列=同一个 z | z 管风格, y 管内容, 两者解耦 |

---

## 实验5: Latent Diffusion —— 把 VAE 的生成引擎换成扩散模型

### 目的
前面所有实验的生成方式都是 `z = randn(16)` **一步到位**。本实验改成:
**纯噪声 → diffusion 逐步去噪 → z → 冻结的 VAE decoder → 图**。
改完就是 **Stable Diffusion 的迷你骨架**。

```
老办法:   z ~ N(0,1) 一把抓          ─► VAE decoder ─► 图
新办法:   纯噪声 ─► diffusion 逐步去噪 ─► z ─► VAE decoder ─► 图
                      └── 本次新增 ──┘     └── 复用实验1的 VAE, 冻结不动 ──┘
```

### 为什么这么设计 (而不是 diffusion 直接出图)
| 问题 | 答案 |
|------|------|
| diffusion 不能直接在像素空间做吗? | 能, 经典 DDPM 就是这么干的。但真图 512×512×3=78万维, 1000 步去噪每步都算 78 万维, 贵到不现实。先压进 latent 再扩散, 维度暴降 (SD 缩 48 倍, 我们缩 49 倍: 784→16) |
| VAE 都训好了, diffusion 起啥作用? | **分工**: VAE decoder 管"像素细节"(边缘、笔画粗细, 类比印刷厂), diffusion 管"语义内容"(图里有什么, 类比作家) |

### 与真实 Stable Diffusion 的零件对照
| 我们的迷你版 | Stable Diffusion |
|-------------|------------------|
| 16 维 latent | 4×64×64 latent |
| MLP 去噪器 + 时间嵌入 | UNet + 时间嵌入 |
| 100 步线性调度 | 1000 步调度 |
| （还没加） | 文本条件 ← 下一步 |

流派归属: 本实现是最经典的 **DDPM** (epsilon-prediction)。DiT 不是并列的另一种东西,
它就是扩散——只是把去噪网络换成 Transformer; Flow Matching 则是另一条范式
(预测速度场、走直线轨迹、步数更省)。

### 方法
**前向扩散** (加噪, 训练时出题): beta 线性 1e-4→0.02 共 100 步, 靠 alpha_bar 一步跳到第 t 步:

\[ z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon \]

**训练目标** (只有一个): 给污染到第 t 步的 z_t, 把混进去的噪声猜出来, MSE:

\[ L = \mathbb{E}_{z_0, t, \epsilon}\, \big\| \epsilon_\theta(z_t, t) - \epsilon \big\|^2 \]

**反向去噪** (生成): 从纯噪声出发循环 100 步, 预测噪声→减掉→补一点随机噪声 (最后一步不补):

\[ z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(z_t - \text{coef}_t \cdot \epsilon_\theta(z_t, t)\big) + \sqrt{\beta_t}\,\epsilon \]

**两阶段流程**:
```
阶段一 (实验1产物, 冻结不动): VAE 784 ↔ 16   ckpt/vae_test/vae_mnist.pt
阶段二 (本实验):
    ① 6万张图 --(冻结 encoder, 取雾心 mu)--> 6万个 16维点
    ② 在这些点上训去噪器 (0.046M 参数)
    ③ 生成: 纯噪声 --去噪100步--> z --(冻结 decoder)--> 图
```
编码取 `mu` 而非重参数采样 —— 扩散要学确定性的数据分布, 不要额外采样噪声。

### 代码结构 (新增 4 个文件)
| 文件 | 职责 |
|------|------|
| `src/diffusion_latent.py` | 核心: TimeEmbedding (正弦编码+MLP) / LatentDenoiser (3层MLP, SiLU) / GaussianDiffusion (调度系数预计算 + q_sample / training_loss / sample) |
| `src/train_latdiff.py` | 加载冻结 VAE → 全量编码取 mu → 训去噪器 → 存 ckpt |
| `src/infer_latdiff.py` | 采样出图 + 对照组 (diffusion vs randn 直接采), 打印两路 z 统计 |
| `configs/train_latdiff.yaml` | timesteps=100, hidden=128, time_dim=64, epochs=30, batch=256, lr=2e-4 |
| `scripts/run_train_latdiff.sh` | SLURM: 1卡30分钟, 训练成功后自动推理出图 |

关键设计点:
- **去噪器极小** (0.046M): latent 才 16 维, 带时间条件的 MLP 足够, 登录节点 CPU 可冒烟
- **时间步怎么进网络**: t → 正弦位置编码 → 2层MLP → 64维, 与 z_t 拼接。同一网络要处理 100 种污染程度, 必须知道自己在第几步
- **调度系数全注册为 buffer**: 随模型自动搬 CPU/GPU, 不进梯度

### 运行
```bash
cd /public/home/liuhuan/workspace_xd/vae_test
sbatch scripts/run_train_latdiff.sh     # 日志: logs/vae_latdiff_<jobid>.out
```
冒烟测试已在登录节点通过 (CPU, 20 步小扩散, 完整链路 噪声→去噪→z→图 走通)。

### 结果 (job 213306, COMPLETED rc=0, 训练仅 46 秒)
- [x] 训练 30 epoch 完成, 最终 loss=0.5207 (0.89 起步 → 0.52 收敛, 符合 ε-prediction 预期)
- [x] `output/images/latdiff/samples_diffusion.png` — 去噪路径出图
- [x] `output/images/latdiff/samples_randn.png` — 对照组 (randn 直接采)
- [x] z 统计对比 (真实编码: 均值=-0.001, std=0.823):

| 采样路径 | 均值 | std | 与真实分布的距离 |
|---------|------|------|----------------|
| **diffusion 去噪** | -0.030 | **0.821** | 几乎贴合真实 std=0.823 |
| randn 直接采 | -0.015 | 1.006 | 偏胖 ~22% |

**结论**: diffusion 路赢。肉眼对照: diffusion 图碎字更少、笔画更实; randn 图
有若干发虚/坍塌的样本。原因: KL 把 latent 压得"接近"标准正态但不是完美标准正态
(真实 std=0.823), randn 从 N(0,1) 采样天生偏胖, 部分样本落在稀疏区出碎字;
diffusion 学到的是真实分布本身, 连尺度一起学对了。

**插曲**: job 内第 2 步出图曾崩溃 (GPU 张量直接 `.numpy()` 报 TypeError,
冒烟测试纯 CPU 未暴露), 已在 `infer_latdiff.py` 的 `save_grid` 加 `.cpu()` 修复后补跑。

本实验的真正价值是**把 SD 的骨架零件全部走通一遍**: 冻结 VAE + latent 内扩散 +
对照验证, 全部闭环。

---

## 实验6: 条件 Latent Diffusion (class-conditional, 指定数字生成)

### 目的
实验5的去噪器是**无条件**的 (`forward(z_t, t)`), 生成什么数字纯靠噪声随机。
本实验给它加**数字条件 y**, 从"随机出数字"升级到"**指定数字出数字**" ——
把实验4 CVAE 的 one-hot 条件思想搬到扩散去噪器上, 也是通往 T2I 的第一级台阶。

### 方法: one-hot 条件注入
- 条件编码: 数字 k → one-hot 10 维向量 (`F.one_hot`), 消除整数输入的虚假数值邻近性
- 注入位置: 去噪器入口 concat `[z_t(16) | temb(64) | onehot(y)(10)] = 90维 → Linear(90→128)`
- 数学本质: one-hot 过第一层 Linear ≡ 从权重后 10 列"查出"第 k 列加进去,
  即每个数字一列专属指令向量 (等价于一次 embedding 查表)
- 训练: y = 真实标签 (**事实**), 教会网络"条件 k → 去噪出 k"
- 推理: y = 用户指定 (**命令**), `sample(ys)` 每步拼 one-hot(ys), 把去噪轨迹导向数字 k

### 代码结构 (新增 5 个文件, 原无条件版保留作对比)
| 文件 | 职责 |
|------|------|
| `src/diffusion_latent_cond.py` | CondLatentDenoiser (入口拼 one-hot, 90维) + CondGaussianDiffusion (继承实验5, 只重写 training_loss/sample 带 y) |
| `src/train_latdiff_cond.py` | encode_dataset 保留标签 (`for x, y`), 训练喂 (z0, y0) |
| `src/infer_latdiff_cond.py` | `--digit N` 单数字 / 默认 10×10 网格 (每行一个数字) |
| `configs/train_latdiff_cond.yaml` | num_classes=10, 输出到 _cond 目录 |
| `scripts/run_train_latdiff_cond.sh` | SLURM: 训练成功后自动出网格图 |

关键设计:
- **继承复用**: CondGaussianDiffusion 继承 GaussianDiffusion, 调度数学/q_sample 原样复用, 只重写带 y 的两个方法
- **参数量 sanity check**: 0.0455M → 0.047M, 多出的正是 one-hot 入口 10×128=1280 参数, 可验证维度对齐

### 运行
```bash
cd /public/home/liuhuan/workspace_xd/vae_test
sbatch scripts/run_train_latdiff_cond.sh     # 日志: logs/vae_latdiff_cond_<jobid>.out
```
冒烟: 登录节点 CPU 1 epoch 验证全链路 (标签保留/参数量/出图); 欠训图每行混杂属预期。

### 结果 (job 213885, COMPLETED rc=0)
- [x] 训练 30 epoch, loss 0.89 → 0.47 收敛
- [x] `output/images/latdiff_cond/cond_grid.png` — 10×10 网格, **每行主导数字 = 指定标签** (条件生效)
- [x] z 统计 std=0.759 (接近真实编码 0.823)
- 诚实说明: 每行仍有约 10-20% 杂格 (个别非该行数字)。one-hot concat 是最朴素的注入方式,
  条件信号有时压不过扩散噪声; 但对比冒烟 (1 epoch) 的完全混杂, 30 epoch 已明显学会条件。

### 结论: 实验4 的条件思想 × 实验5 的生成引擎
```
实验4 CVAE:    one-hot 注入 decoder, randn 一步出图
实验5 LatDiff: 无条件扩散去噪
实验6 (本):    one-hot 注入扩散去噪器 = 条件思想 × 扩散引擎 => 指定数字生成
下一步 T2I:    把 one-hot 换成文本向量 (+cross-attention) => 文字生图
```

---

## 下一步: 迷你文字生图

在实验6 (条件扩散) 基础上, 把 one-hot 数字条件升级为**文本向量**:

```
实验6 CondLatDiff:  y=数字标签 → one-hot(10维) → 注入去噪器
下一步 T2I:        “a handwritten 8” → 字符级编码 → 文本向量 → 注入去噪器 (+cross-attention)
```

得到完整链路: 文本 + 纯噪声 → 条件扩散逐步去噪 → z → VAE decoder → 图。

## 附录: 环境备忘

- Python: `/public/home/liuhuan/anaconda/anaconda3/envs/py310_qwen_pip/bin/python` (SLURM 下 `conda activate` 会失效, 必须用绝对路径)
- torch 2.1.2 + cu118
- ckpt 目录约定: `/public/home/liuhuan/workspace_xd/ckpt/vae_test/` 只放 `.pt`, 效果图放项目 `output/images/`
- MNIST 数据: `/public/home/liuhuan/workspace_xd/data` (多项目共享)
