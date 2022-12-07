#!/bin/bash
source activate
conda deactivate
conda activate pytorch

python3 /home/lfsm/code/ASDL_train/debug_model_with_asdl.py --model vit_tiny_patch16_224 \
      --optimizer shampoo \
      --img-size 224 \
      --lr 1e-3 \
      --epochs 40 \
      --batch-size 512 \
