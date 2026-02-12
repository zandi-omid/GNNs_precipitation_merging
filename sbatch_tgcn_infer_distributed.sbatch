#!/bin/bash
#SBATCH --account=xdong
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --job-name=tgcn_infer_t30
#SBATCH --output=logs/tgcn_infer_%j.out
#SBATCH --error=logs/tgcn_infer_%j.err
#SBATCH --exclusive

export SLURM_NO_AUDIT=1
export SLURM_DISABLE_FATTR=1

echo "========== JOB STARTED: $(date) =========="
echo "Node list: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "----------------------------------------------------------"

source /home/u20/omidzandi/micromamba/etc/profile.d/mamba.sh
micromamba activate pyG   # <-- use the env that has pytorch-lightning + torch-geometric

cd /xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval

# Prevent CPU oversubscription (important when many ranks run on same node)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0

nvidia-smi || true
echo "----------------------------------------------------------"

# Each SLURM task = one rank = one GPU
# Our script uses SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID
srun --export=ALL \
  python inference_distributed.py \
    --config configs/tgcn_infer.toml

echo "=========== JOB ENDED: $(date) ==========="