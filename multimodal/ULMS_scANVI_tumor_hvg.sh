#!/bin/bash
#SBATCH --job-name=scANVI_tumor_hvg
#SBATCH --partition=gpu_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --account=delitto

echo "The current Slurm job name is $SLURM_JOB_NAME"
echo "Job started at $(date)"

# Prevent oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate /home/jpagolia/miniforge3/envs/scvi-env
echo "The current conda environment is $CONDA_PREFIX"

python -u ULMS_scANVI_tumor_hvg.py

conda deactivate
echo "Job finished at: $(date)"
