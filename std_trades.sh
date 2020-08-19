#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-6]
#SBATCH -J std_trades
#SBATCH -o logs/std_trades.%J.out
#SBATCH -e logs/std_trades.%J.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python train.py --cfg trades_configs/trades_exp${SLURM_ARRAY_TASK_ID}.yml --gpus 0 --output trades_${SLURM_ARRAY_TASK_ID} --size small