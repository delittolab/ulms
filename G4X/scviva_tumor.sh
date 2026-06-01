#!/bin/bash
#SBATCH --job-name=tumor
#SBATCH --partition=batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=delitto

echo "Start"
eval "$(conda shell.bash hook)"
conda activate /home/jpagolia/miniforge3/envs/scvi-env

python -u scviva_tumor.py

conda deactivate
echo "Completion"
