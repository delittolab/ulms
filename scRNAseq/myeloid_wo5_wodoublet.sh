#!/bin/sh
#SBATCH --job-name=m_wo5_wod
#SBATCH --partition=gpu_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --account=delitto

echo "$SLURM_JOB_NAME"
echo "Start at $(date)"
nvidia-smi
source /home/jpagolia/agolia_virtual_env/bin/activate
echo "$VIRTUAL_ENV"
python -u myeloid_wo5_wodoublet.py
deactivate
echo "Completion at $(date)"
