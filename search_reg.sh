#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-7]
#SBATCH -J ensem_search
#SBATCH -o logs/ensem_search.%J.out
#SBATCH -e logs/ensem_search.%J.err
#SBATCH --time=30:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python train.py --cfg configs/exp${SLURM_ARRAY_TASK_ID}.yml --gpus 0 --output adv_col_${SLURM_ARRAY_TASK_ID}