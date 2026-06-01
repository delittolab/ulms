#!/bin/sh
#SBATCH --job-name=scib
#SBATCH --partition=gpu_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --account=delitto

echo "Start"
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate /oak/stanford/groups/longaker/james/.envs/scib
python -u ulms_scib_part2.py
conda deactivate
echo "Completion"
