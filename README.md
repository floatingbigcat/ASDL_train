# ASDL_train
Train models with asdl/dev-grad-maker.

- Models: supported from timm(model library). vit-b16, vit-tiny.
- Dataset: CIFAR10, CIFAR100

TIMM:
https://timm.fast.ai/

ASDL(Automatic Second-order Differentiation Library):
https://github.com/kazukiosawa/asdl/tree/dev-grad-maker

## Usage
  1. add path of asdl library in utils/utils.py/line2
  2. change hyper search range in train_model_with_asdl.py
 
