#!/bin/bash
#SBATCH --job-name=scPoli_all
#SBATCH --partition=gpu_normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --account=delitto

echo "The current Slurm job name is $SLURM_JOB_NAME"
echo "Job started at $(date)"
eval "$(conda shell.bash hook)"
conda activate /labs/delitto/james/.envs/arch
echo "The current conda environment is $CONDA_PREFIX"

python -u ULMS_scPoli_all.py

conda deactivate
echo "Job finished at: $(date)"
