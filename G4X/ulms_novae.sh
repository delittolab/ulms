#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --partition=gpu_normal
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --account=delitto

echo "Start"
eval "$(conda shell.bash hook)"
conda activate /oak/stanford/groups/longaker/james/.envs/novae

python -u ulms_novae.py

conda deactivate
echo "Completion"
