#!/bin/bash
#SBATCH --job-name=scviva
#SBATCH --partition=gpu_admin
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_test

echo "Start"
eval "$(conda shell.bash hook)"
conda activate /home/jpagolia/miniforge3/envs/scvi-env

python -u scviva.py

conda deactivate
echo "Completion"
