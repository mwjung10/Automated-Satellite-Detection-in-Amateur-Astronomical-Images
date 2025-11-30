#!/bin/bash
#
#SBATCH -J TreningRTDERT 
#SBATCH -p plgrid-gpu-a100  
#SBATCH --gres=gpu:1   
#SBATCH -N 1  
#SBATCH -n 8  
#SBATCH --mem=32G  
#SBATCH -t 03:00:00 
#SBATCH --output="rtdert_log_%j.out" #(%j = task ID)
#SBATCH --error="rtdert_err_%j.err" 

echo "Task started: $(date)"
cd $SLURM_SUBMIT_DIR

# Load conda config
source ~/.bashrc
conda activate PDyplomowa

echo "Environment started. Starts python script..."

python -u Models_training/rtdert_train.py

echo "Task finished: $(date)"
