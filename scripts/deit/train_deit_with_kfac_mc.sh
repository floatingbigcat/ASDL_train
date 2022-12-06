#!/bin/bash
#YBATCH -r a100_1
#SBATCH -N 1
#SBATCH -o /home/lfsm/code/exp_result/logs/deit_tinykfacmc.%j.out
#SBATCH --time=72:00:00
#SBATCH -J deit_kmc
#SBATCH --error /home/lfsm/code/exp_result/logs/deit_tinykfacmc%j.err
source activate
conda deactivate
conda activate torch
cp -r /home/lfsm/sundatasets/CIFAR10/cifar-10-batches-py $HINADORI_LOCAL_SCRATCH
python3 /home/lfsm/code/sun_asdfghjkl/train_model_with_asdf.py --model deit_tiny_patch16_224 \
      --optimizer kfac_mc \
      --epochs 50 \
      --batch-size 512 \
      --lr 0.003 \
      --ema-decay -1 \
      --clip-grad-norm 10 \
      --log-path /home/lfsm/code/exp_result/deit_tiny/kfac_mc_lr0.003_clip10 \
      --data-path $HINADORI_LOCAL_SCRATCH \
