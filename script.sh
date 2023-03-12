#!/usr/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH -t 1-00:00:00
#SBATCH -p debug 
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=mind-1-23
#SBATCH --output=/user_data/isaacl/stdoutx1.out

cd /user_data/isaacl/tang-vit-tuning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangvit
python tang-vit-run.py
