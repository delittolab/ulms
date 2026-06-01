#!/bin/bash
#SBATCH --job-name=scvi_low
#SBATCH --partition=gpu_normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --account=delitto

echo "Start"
nvidia-smi
source /home/jpagolia/agolia_virtual_env/bin/activate
python -u ulms_scvi_low_quality.py
deactivate
echo "Completion"
