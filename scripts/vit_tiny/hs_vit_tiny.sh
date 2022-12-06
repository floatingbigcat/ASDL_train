#!/bin/bash
#YBATCH -r dgx-a100_1
#SBATCH -N 1
#SBATCH -o /home/lfsm/code/result/logs/vit_t%j.out
#SBATCH --time=72:00:00
#SBATCH -J v-sm
#SBATCH --error /home/lfsm/code/result/logs/vit_t%j.err
source activate
conda deactivate
conda activate pytorch
cp -r /home/lfsm/sundatasets/CIFAR10/cifar-10-batches-py $HINADORI_LOCAL_SCRATCH

python /home/lfsm/code/train/train_model_with_asdl.py
        --data-path $HINADORI_LOCAL_SCRATCH

