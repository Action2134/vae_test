#!/bin/bash
#SBATCH --job-name=vae_latdiff_cond
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaV100-SXM2-32GB:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/vae_latdiff_cond_%j.out
#SBATCH --error=logs/vae_latdiff_cond_%j.err

# 注意: SLURM 下 conda activate 会静默失效, 必须用 Python 绝对路径
ENV_BIN=/public/home/liuhuan/anaconda/anaconda3/envs/py310_qwen_pip/bin/python
cd /public/home/liuhuan/workspace_xd/vae_test
mkdir -p logs

# GPU 自检: 失败时日志里能直接看到原因
echo "=== GPU 自检 ==="
$ENV_BIN -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())" || { echo "GPU 自检失败, 退出"; exit 1; }

echo "=== 第 1 步: 训练条件 latent 扩散 (冻结 VAE + 训条件去噪器) ==="
echo "job start: $(date)"
$ENV_BIN src/train_latdiff_cond.py --config configs/train_latdiff_cond.yaml
TRAIN_RC=$?
echo "job end: $(date), rc=$TRAIN_RC"

# 训练失败就不出图了
if [ $TRAIN_RC -ne 0 ]; then
    echo "训练失败, 跳过推理出图"
    exit $TRAIN_RC
fi

echo "=== 第 2 步: 指定数字生成出图 (10x10 网格, 每行一个数字) ==="
$ENV_BIN src/infer_latdiff_cond.py
echo "infer rc=$?"
