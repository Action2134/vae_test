#!/bin/bash
#SBATCH --job-name=vae_cvae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaV100-SXM2-32GB:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/vae_cvae_%j.out
#SBATCH --error=logs/vae_cvae_%j.err

# 注意: SLURM 下 conda activate 会静默失效, 必须用 Python 绝对路径
# logs/ 目录必须提前存在 (SLURM 在脚本执行前就打开日志文件)
ENV_BIN=/public/home/liuhuan/anaconda/anaconda3/envs/py310_qwen_pip/bin/python
cd /public/home/liuhuan/workspace_xd/vae_test
mkdir -p logs

# GPU 自检: 失败时日志里能直接看到原因
echo "=== GPU 自检 ==="
$ENV_BIN -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())" || { echo "GPU 自检失败, 退出"; exit 1; }

echo "=== 第 1 步: 训练条件 VAE (GPU) ==="
echo "job start: $(date)"
$ENV_BIN src/train.py --config configs/train_vae_cvae.yaml
TRAIN_RC=$?
echo "job end: $(date), rc=$TRAIN_RC"

if [ $TRAIN_RC -ne 0 ]; then
    echo "训练失败, 跳过推理"
    exit $TRAIN_RC
fi

echo "=== 第 2 步: 条件生成网格 (10行数字 x 10列随机z) ==="
$ENV_BIN src/my_infer.py --mode cond \
    --ckpt /public/home/liuhuan/workspace_xd/ckpt/vae_test/cvae/vae_mnist.pt \
    --out_dir /public/home/liuhuan/workspace_xd/vae_test/output/images/my_infer_cvae
echo "cond infer rc=$?"
