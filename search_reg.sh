#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[6-16]
#SBATCH -J XS_iso
#SBATCH -o logs/XS_iso.%J.out
#SBATCH -e logs/XS_iso.%J.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python train.py --cfg configs/exp${SLURM_ARRAY_TASK_ID}.yml --gpus 0 --output XSISONet_small_${SLURM_ARRAY_TASK_ID} --size small