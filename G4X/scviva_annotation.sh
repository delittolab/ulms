#!/bin/bash
#SBATCH --job-name=ann_scviva
#SBATCH --partition=batch
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"
eval "$(conda shell.bash hook)"
conda activate /home/jpagolia/miniforge3/envs/scvi-env

python -u scviva_annotation.py

conda deactivate
echo "Completion"
