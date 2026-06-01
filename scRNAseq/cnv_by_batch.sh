#!/bin/bash
#SBATCH --job-name=cnv
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"

eval "$(conda shell.bash hook)"

conda activate /labs/delitto/james/.envs/jpa_infercnv

python -u cnv_by_batch.py

conda deactivate

echo "Completion"
