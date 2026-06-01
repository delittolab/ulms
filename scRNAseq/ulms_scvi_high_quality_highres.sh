#!/bin/bash
#SBATCH --job-name=highres
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"
source /home/jpagolia/agolia_virtual_env/bin/activate
python -u ulms_scvi_high_quality_highres.py
deactivate
echo "Completion"
