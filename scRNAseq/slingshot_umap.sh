#!/bin/bash
#SBATCH --job-name=ss_umap
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --account=delitto

echo "Start"

module load R/4.3.3

Rscript slingshot_umap.R

echo "Completion"
