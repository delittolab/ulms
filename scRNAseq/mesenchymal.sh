#!/bin/sh
#SBATCH --job-name=mesenchymal
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --account=delitto

# Removed GPU request from this file to cluster at higher resolutions.

echo "Start"
source /home/jpagolia/agolia_virtual_env/bin/activate
python -u mesenchymal.py
deactivate
echo "Completion"
