import torch
from torch import nn
import torch.nn.functional as F
from utils.dataset import getDataset
import utils.utils as utils
import wandb

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ASDL')
    parser.add_argument('--data-path',
                        default="/home/lfsm/sundataset/CIFAR100",
                        type=str)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--auto_aug', default='rand-m9-mstd0.5-inc1', type=str,choices=['rand-m9-mstd0.5-inc1'])
    parser.add_argument('--img-size', default=224, type=int)
    parser.add_argument("--model", default="vit_tiny_patch16_224", type=str, choices=["resnet18","vit_tiny_patch16_224", "vit_base_patch16_224_in21k"])
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "rmsprop", "adamw","kfac_mc","kfac_emp","shampoo","psgd","kbfgs"])
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

def train(epoch, dataset, model, optimizer, grad_maker, args):
    model.train()
    metric = utils.getMetric(args)
    num_steps_per_epoch = len(dataset.train_loader)
    for i, (inputs, targets) in enumerate(dataset.train_loader):
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
            wandb.log(metrics)            
            print(f'Train Epoch{epoch} Iter{(epoch-1)*num_steps_per_epoch+i} Train Loss{loss:.4f}')

def val(epoch, dataset, model, criterion, args):
    model.eval()
    metric = utils.Metric(args)
    num_steps_per_epoch = len(dataset.val_loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataset.val_loader):
            inputs,targets = inputs.to(args.device),targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metric.update(inputs.shape[0], loss, outputs, targets)
    metrics = {"val/loss": metric.loss,
                "val/accuracy":metric.accuracy,
                "val/iteration": (epoch-1)*num_steps_per_epoch+i-1 ,
                "val/epoch": epoch}
    wandb.log(metrics)    
    print(f'Epoch {epoch} Val/loss {metric.loss:.4f} Val/Acc {metric.accuracy:.4f}')

def test(epoch, dataset, model, criterion, args):
    model.eval()
    metric = utils.getMetric(args)
    num_steps_per_epoch = len(dataset.test_loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataset.test_loader):
            inputs,targets = inputs.to(args.device),targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metric.update(inputs.shape[0], loss, outputs, targets)
    metrics = {"test/loss": metric.loss,
                "test/accuracy":metric.accuracy,
                "test/iteration": (epoch-1)*num_steps_per_epoch+i-1 ,
                "test/epoch": epoch}
    wandb.log(metrics)    
    print(f'Epoch {epoch} test/loss {metric.loss:.4f} test/Acc {metric.accuracy:.4f}')

def train_pipeline():
    # ========== Setting ==========
    # get config and init wandb
    config = vars(parse_args()).copy()
    wandb.init(config=config)
    args = wandb.config
    print(args)
    # ensure compute speed
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        ### enable TensorFloat-32 tensor cores to be used in matrix multiplications
        torch.backends.cudnn.benchmark = True
        ### causes cuDNN to benchmark multiple convolution algorithms and select the fastest.    
    torch.manual_seed(args.seed)
    dataset = getDataset(args)
    criterion = nn.CrossEntropyLoss()
    num_classes=len(dataset.train_dataset.classes)
    model = utils.getModel(args,num_classes)
    optimizer,grad_maker = utils.getOptGM(model,args)
    lr_scheduler = utils.getLrscheduler(args,optimizer)
    # ========== TRAINING ==========
    for epoch in range(1,args.epochs+1):
        train(epoch, dataset, model, optimizer, grad_maker, args)
        ### as CIFAR10 from torchvision.datasets don't have val dataset, only test is used
        # val(epoch, dataset, model, criterion, args)
        test(epoch, dataset, model, criterion, args)
        lr_scheduler.step()
    wandb.finish()

def hyper_search():
    # Wandb for grid search
    sweep_config = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'optimizer':{
                'values':["sgd", "rmsprop", "adamw", "kfac_mc", "kfac_emp","shampoo","psgd"]
            },
            'lr':{
                'values':[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]  
            },
            'batch_size':{
                'values':[32, 64, 128, 256, 512]
            }
        }
    }
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="vit_with_asdl")
    # Start sweep job.
    wandb.agent(sweep_id, function=train_pipeline)

if __name__ == "__main__":
    hyper_search()
    # train_pipeline()