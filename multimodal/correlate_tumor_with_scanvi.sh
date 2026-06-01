#!/bin/bash
#SBATCH --job-name=corr
#SBATCH --partition=batch
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --account=delitto

echo "The current Slurm job name is $SLURM_JOB_NAME"
echo "Job started at $(date)"

eval "$(conda shell.bash hook)"
conda activate /home/jpagolia/miniforge3/envs/scvi-env
echo "The current conda environment is $CONDA_PREFIX"

python -u correlate_tumor_with_scanvi.py

conda deactivate
echo "Job finished at: $(date)"
