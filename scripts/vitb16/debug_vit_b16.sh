#!/bin/bash
source activate
conda deactivate
conda activate pytorch

python3 /home/lfsm/code/ASDL_train/debug_model_with_asdl.py --model resnet18 \
      --optimizer sgd \
      --img-size 384 \
      --lr 1e-3 \
      --epochs 40 \
      --batch-size 100 \
