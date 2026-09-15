#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaV100-SXM2-32GB:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/train_vae_%j.out
#SBATCH --error=logs/train_vae_%j.err

# 注意: SLURM 下 conda activate 会静默失效, 必须用 Python 绝对路径
ENV_BIN=/public/home/liuhuan/anaconda/anaconda3/envs/py310_qwen_pip/bin/python
cd /public/home/liuhuan/workspace_xd/vae_test
mkdir -p logs output

echo "job start: $(date)"
$ENV_BIN src/train.py --config configs/train_vae.yaml
echo "job end: $(date)"
