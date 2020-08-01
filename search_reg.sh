#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-16]
#SBATCH -J ison_search_reg
#SBATCH -o logs/ison_search_reg.%J.out
#SBATCH -e logs/ison_search_reg.%J.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python train.py --cfg configs/exp${SLURM_ARRAY_TASK_ID}.yml --gpus 0 --output isoS_exp${SLURM_ARRAY_TASK_ID} --size small