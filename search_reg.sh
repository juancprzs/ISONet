#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=1
#SBATCH -J ensem
#SBATCH -o logs/ensem.%J.out
#SBATCH -e logs/ensem.%J.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python train.py --cfg configs/CIF10-ISO18.yaml --gpus 0 --output ensemble_debug