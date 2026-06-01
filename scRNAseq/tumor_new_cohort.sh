#!/bin/sh
#SBATCH --job-name=new_tumor
#SBATCH --partition=gpu_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"
nvidia-smi
source /home/jpagolia/agolia_virtual_env/bin/activate
python -u tumor_new_cohort.py
deactivate
echo "Completion"
