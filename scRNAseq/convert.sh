#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --partition=batch
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"

module load R/4.3.3

Rscript convert.R

echo "Completion"
