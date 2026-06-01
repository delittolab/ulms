#!/bin/bash
#SBATCH --job-name=infercnvpy
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"

eval "$(conda shell.bash hook)"

conda activate /labs/delitto/james/.envs/jpa_infercnv

python -u infercnv.py

conda deactivate

echo "Completion"
