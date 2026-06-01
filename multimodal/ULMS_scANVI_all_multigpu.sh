#!/bin/bash
#SBATCH --job-name=scANVI_all_multigpu
#SBATCH --partition=gpu_normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --account=delitto

echo "The current Slurm job name is $SLURM_JOB_NAME"
echo "Job started at $(date)"

nvidia-smi
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Environment variables for multi-GPU
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN

eval "$(conda shell.bash hook)"
conda activate /home/jpagolia/miniforge3/envs/scvi-env
echo "The current conda environment is $CONDA_PREFIX"

python -u ULMS_scANVI_all_multigpu.py

conda deactivate
echo "Job finished at: $(date)"
