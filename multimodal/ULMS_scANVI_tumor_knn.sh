#!/bin/bash
#SBATCH --job-name=knn
#SBATCH --partition=batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=delitto

echo "The current Slurm job name is $SLURM_JOB_NAME"
echo "Job started at $(date)"

eval "$(conda shell.bash hook)"
conda activate /home/jpagolia/miniforge3/envs/scvi-env
echo "The current conda environment is $CONDA_PREFIX"

python -u ULMS_scANVI_tumor_knn.py

conda deactivate
echo "Job finished at: $(date)"
