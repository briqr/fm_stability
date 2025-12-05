#!/bin/bash
#SBATCH --account= account           
#SBATCH --nodes=1                 
#SBATCH --gres=gpu:4 
#SBATCH --job-name=exp_name
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=48              
#SBATCH -o logs/expname%j.log
#SBATCH --time=12:00:00          
#SBATCH --partition=partition

source env_name/activate.sh
dataset="celebhq"
pruning_method="unpruned"
pruning_ratio="0"
num_classes="1"
batch_size="128"
inverse="0" #whether the method or its inverse

srun torchrun --nnodes=1 --nproc_per_node=4 train.py --dataset $dataset --pruning_method $pruning_method --pruning_ratio $pruning_ratio --inverse $inverse --global-batch-size $batch_size --num-classes $num_classes 
