#!/bin/sh
#SBATCH --job-name=qc_ulms
#SBATCH --partition=batch
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"

source /home/jpagolia/agolia_virtual_env/bin/activate

python -u ulms_pp.py

deactivate

echo "Completion"
