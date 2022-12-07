#!/bin/bash
#YBATCH -r am_1
#SBATCH -N 1
#SBATCH -o /home/lfsm/code/result/logs/vit_b%j.out
#SBATCH --time=72:00:00
#SBATCH -J v-sm
#SBATCH --error /home/lfsm/code/result/logs/vit_b%j.err
source activate
conda deactivate
conda activate pytorch
python /home/lfsm/code/ASDL_train/train_model_with_asdl.py  --model vit_base_patch16_224_in21k \
        --dataset CIFAR100 \
        --batch-size 100 \
        --img-size 224 \
        --optimizer sgd
