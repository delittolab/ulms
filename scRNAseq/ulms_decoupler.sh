#!/bin/bash
#SBATCH --job-name=dc
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"

eval "$(conda shell.bash hook)"

conda activate /labs/delitto/james/.envs/decoupler

python -u ulms_decoupler.py

conda deactivate

echo "Completion"
