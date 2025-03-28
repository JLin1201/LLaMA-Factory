#!/bin/sh
#SBATCH --job-name=
#SBATCH -p 
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --account=
#SBATCH -o 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32



# Run the training command with FORCE_TORCHRUN
FORCE_TORCHRUN=1 llamafactory-cli train LLaMA-Factory/examples/train_full/Qwen2.5_full_pretrain.yaml