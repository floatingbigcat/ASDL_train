source activate
conda deactivate
conda activate pytorch
cp -r /home/lfsm/sundatasets/CIFAR10/cifar-10-batches-py $HINADORI_LOCAL_SCRATCH

python /home/lfsm/code/train/debug_model_with_asdl.py  --model vit_tiny_patch16_224 \
        --epochs 20 \
        --optimizer sgd \
        --batch-size 512 \
        --data-path $HINADORI_LOCAL_SCRATCH