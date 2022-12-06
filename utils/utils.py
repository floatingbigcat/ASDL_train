import sys
sys.path.append('/home/lfsm/code/asdfghjkl')

import torch
import torch.nn as nn
import asdfghjkl as asdl
from asdfghjkl import FISHER_EMP,FISHER_MC


def chooseOptGM(model, args):
    """
    choose optimizer and gradmaker according to args
    """
    parameters = model.parameters()
    opt_name = args.optimizer.lower()
    if opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
            )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    if opt_name == "kfac_mc":
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_MC,
                                            damping=args.damping,
                                            curvature_upd_interval=args.cov_update_freq,
                                            preconditioner_upd_interval=args.inv_update_freq,
                                            ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config, swift=False)
    elif opt_name == "kfac_emp":
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_EMP,
                                            damping=args.damping,
                                            curvature_upd_interval=args.cov_update_freq,
                                            preconditioner_upd_interval=args.inv_update_freq,
                                            ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config, swift=False)
    elif opt_name == "shampoo":
        config = asdl.ShampooGradientConfig(damping=args.damping,
                                        curvature_upd_interval=args.cov_update_freq,
                                        preconditioner_upd_interval=args.inv_update_freq)
        grad_maker = asdl.ShampooGradientMaker(model, config)
    elif opt_name == "psgd":
        config = asdl.PsgdGradientConfig(curvature_upd_interval=args.cov_update_freq,
                                        # ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
                                        preconditioner_upd_interval=args.inv_update_freq)
        grad_maker = asdl.KronPsgdGradientMaker(model,config)
    elif opt_name == "smw_ngd":
        config = asdl.SmwEmpNaturalGradientConfig(data_size=args.batch_size,
                                                  ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
                                                  damping=args.damping)
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
    elif opt_name == "kbfgs":
        config = asdl.KronBfgsGradientConfig(
                                             data_size=args.batch_size,
                                             ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
                                            damping=args.damping)
        grad_maker = asdl.KronBfgsGradientMaker(model, config)
    elif opt_name == "seng":
        ### LayerNorm not support
        config = asdl.SengGradientConfig(data_size=args.batch_size,
                                         damping=args.damping)
        grad_maker = asdl.SengGradientMaker(model, config)
    else:
        grad_maker = asdl.GradientMaker(model)        
    return optimizer,grad_maker

class Metric(object):
    """
    metric for classification
    """
    def __init__(self, device):
        self._n = torch.tensor([0.0]).to(device)
        self._losses = torch.tensor([0.0]).to(device)
        self._corrects = torch.tensor([0.0]).to(device)

    def update(self, n, loss, outputs, targets):
        with torch.inference_mode():
            self._n += n
            self._losses += loss * n 
            _, preds = torch.max(outputs, dim=1)
            if targets.dim() != 1: # use mix up
                _, targets = torch.max(targets, dim=1)
            self._corrects += torch.sum(preds == targets)
    @property
    def loss(self):
        return (self._losses / self._n).item()
    @property
    def accuracy(self):
        return (self._corrects / self._n).item()
