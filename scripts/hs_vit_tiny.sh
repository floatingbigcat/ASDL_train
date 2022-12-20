#!/bin/bash
#YBATCH -r a6000_1
#SBATCH -N 1
#SBATCH -o /home/lfsm/code/result/logs/vit_t%j.out
#SBATCH --time=72:00:00
#SBATCH -J vit_t
#SBATCH --error /home/lfsm/code/result/logs/vit_t%j.err
source activate
conda deactivate
conda activate pytorch
python /home/lfsm/code/ASDL_train/train_model_with_asdl.py  \
        --model vit_tiny_patch16_224 \
        --dataset CIFAR10 \
        --img-size 224

