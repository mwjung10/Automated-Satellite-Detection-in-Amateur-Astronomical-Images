#!/bin/bash
#
#SBATCH -J TreningYOLOFalseNegPos
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output="yolo_log_flase_%j.out" #(%j = task ID)
#SBATCH --error="yolo_err_false_%j.err"

echo "Task started: $(date)"
cd $SLURM_SUBMIT_DIR

# Load conda config
source ~/.bashrc
conda activate PDyplomowa

echo "Environment started. Starts python script..."

python -u false_neg_pos.py

echo "Task finished: $(date)"
