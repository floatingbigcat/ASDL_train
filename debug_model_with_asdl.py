import torch
import timm
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from utils.dataset import CIFAR10
from utils.utils import chooseOptGM,Metric
# import wandb


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ASDL')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--data-path',
                        default="/home/lfsm/sundatasets/CIFAR100",
                        type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument("--model", default="vit_base_patch16_224", type=str, choices=["resnet18","vit_tiny_patch16_224", "vit_base_patch16_224"])
    parser.add_argument('--img-size', default=384, type=int)
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "rmsprop", "adamw", "kfac_mc", "kfac_emp","shampoo","psgd"])
    parser.add_argument("--dataset", default="CIFAR100", type=str, choices=['CIFAR10','CIFAR100'])
    parser.add_argument('--label-smoothing', default=0.1, type=float)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup-epochs', type=float, default=0)
    parser.add_argument('--warmup-factor', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cov-update-freq', type=int, default=10)
    parser.add_argument('--inv-update-freq', type=int, default=100)
    parser.add_argument('--ema-decay', type=float, default=-1)
    parser.add_argument('--damping', type=float, default=0.001)
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--clip-grad-norm', type = float,  default=10)
    parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
    return parser.parse_args()

def train(epoch, datasets, model, optimizer, grad_maker, args):
    model.train()
    metric = Metric(args.device)
    num_steps_per_epoch = len(datasets.train_loader)
    from tqdm import tqdm
    for i, (inputs, targets) in tqdm(enumerate(datasets.train_loader)):
        # prepare
        inputs,targets = inputs.to(args.device),targets.to(args.device)
        optimizer.zero_grad()
        # grad_maker 
        dummy_y = grad_maker.setup_model_call(model, inputs)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, targets,label_smoothing=args.label_smoothing)
        # forward and backward
        outputs, loss = grad_maker.forward_and_backward()
        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step() 
        # record info
        metric.update(inputs.shape[0], loss, outputs, targets)
        if i % args.log_interval == 0:
            metrics = {"train/loss": float(loss),
                       "train/iteration": (epoch-1)*num_steps_per_epoch+i ,
                       "train/epoch": epoch, 
                       "train/lr": optimizer.param_groups[0]['lr']}
            # wandb.log(metrics)            
            print(f'Train Epoch{epoch} Iter{(epoch-1)*num_steps_per_epoch+i} Loss{loss:.4f}')

def val(epoch, datasets, model, criterion, args):
    model.eval()
    metric = Metric(args.device)
    num_steps_per_epoch = len(datasets.val_loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(datasets.val_loader):
            inputs,targets = inputs.to(args.device),targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metric.update(inputs.shape[0], loss, outputs, targets)
    metrics = {"val/loss": metric.loss,
                "val/accuracy":metric.accuracy,
                "val/iteration": (epoch-1)*num_steps_per_epoch+i-1 ,
                "val/epoch": epoch}
    # wandb.log(metrics)    
    print(f'Epoch {epoch} Val/loss {metric.loss:.4f} Val/Acc {metric.accuracy:.4f}')

def test(epoch, datasets, model, criterion, args):
    model.eval()
    metric = Metric(args.device)
    num_steps_per_epoch = len(datasets.test_loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(datasets.test_loader):
            inputs,targets = inputs.to(args.device),targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metric.update(inputs.shape[0], loss, outputs, targets)
    metrics = {"test/loss": metric.loss,
                "test/accuracy":metric.accuracy,
                "test/iteration": (epoch-1)*num_steps_per_epoch+i-1 ,
                "test/epoch": epoch}
    # wandb.log(metrics)    
    print(f'Epoch {epoch} test/loss {metric.loss:.4f} test/Acc {metric.accuracy:.4f}')

if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    # ========== Logger ==========
    config = vars(args).copy()
    datasets = CIFAR10(args)
    criterion = nn.CrossEntropyLoss()
    num_classes=len(datasets.train_dataset.classes)
    # ========== MODEL ==========
    if args.model in ["vit_tiny_patch16_224", "vit_base_patch16_224"]:
        # vit model needs img_size args for ensure input size
        model = timm.create_model(args.model, pretrained=args.pretrained, img_size=args.img_size,num_classes = num_classes) ### Add num_classes!
    else:
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes = num_classes) ### Add num_classes!
    model.to(args.device)
    # ========== OPTIMIZER ==========
    optimizer,grad_maker = chooseOptGM(model,args)
    # ========== LEARNING RATE SCHEDULER ==========
    if args.warmup_epochs > 0:
        lr_scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, args.warmup_factor, total_iters=args.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)],
            milestones=[args.warmup_epochs]
        )
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ========== TRAINING ==========
    # training 
    total_train_time = 0
    for epoch in range(1,args.epochs+1):
        train(epoch, datasets, model, optimizer, grad_maker, args)
        ### as CIFAR10 don't have val datasets, so only test is used
        # val(epoch, datasets, model, criterion, args)
        test(epoch, datasets, model, criterion, args)
        lr_scheduler.step()
