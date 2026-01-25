#!/bin/bash
#SBATCH --account=xdong
#SBATCH --partition=gpu_standard
#SBATCH --nodes=5
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=tgcn_T14_L3
#SBATCH --output=logs/tgcn_%j.out
#SBATCH --error=logs/tgcn_%j.err

set -euo pipefail

# ========== Environment ==========
source /home/u20/omidzandi/micromamba/etc/profile.d/mamba.sh
micromamba activate pyG

cd /xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval
mkdir -p logs

# Prevent CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# NCCL / debugging (optional but useful)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "========== JOB STARTED: $(date) =========="
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# -------- Training --------
srun /home/u20/omidzandi/micromamba/envs/pyG/bin/python \
     /xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/training_task.py \
     --config /xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/configs/tgcn_train.toml

echo "=========== JOB ENDED: $(date) ==========="