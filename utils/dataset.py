import torchvision
from torch.utils.data import RandomSampler,SequentialSampler, DataLoader
from torchvision import transforms
import os
from timm.data.transforms_factory import create_transform

# minidataset /home/lfsm/minidataset/miniImageNet # debug pwd /home/lfsm/code/debug_trash
class Dataset(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = True
        self.train_sampler = RandomSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                     batch_size=self.batch_size,
                                     sampler=self.train_sampler,
                                     num_workers=args.num_workers,
                                     pin_memory=True)
        self.eval_sampler = SequentialSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                    batch_size=self.batch_size,
                                    sampler=self.eval_sampler,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                    batch_size=self.batch_size,
                                    sampler=self.eval_sampler,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
        
class IMAGENET(Dataset):
    def __init__(self, args):
        self.num_classes = 1000
        ImageNetTransforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=ImageNetTransforms)
        self.val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'val'),
            transform=ImageNetTransforms)
        self.test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'test'),
            transform=ImageNetTransforms)        
        super().__init__(args)


class CIFAR10(Dataset):
    def __init__(self, args):      
        TrainCIFAR10Transforms = create_transform(
            args.img_size,
            is_training=True,
            auto_augment=args.auto_aug,
            interpolation='bilinear',
            re_prob=0.0,
        )        
        ValCIFAR10Transforms = create_transform(
            args.img_size,
            is_training=False,
            interpolation='bilinear',
            crop_pct=1
        ) 

        # no aug:
        # transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
    
        # transform_test = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
    
        self.train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, transform=TrainCIFAR10Transforms, download=True)
        self.val_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=ValCIFAR10Transforms)        
        self.test_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=ValCIFAR10Transforms)        
        super().__init__(args)
        
class CIFAR100(Dataset):
    def __init__(self, args):      
        TrainCIFAR100Transforms = create_transform(
            args.img_size,
            is_training=True,
            auto_augment=args.auto_aug,
            interpolation='bilinear',
            re_prob=0.0,
        )        
        ValCIFAR100Transforms = create_transform(
            args.img_size,
            is_training=False,
            interpolation='bilinear',
            crop_pct=1
        ) 
        self.train_dataset = torchvision.datasets.CIFAR100(root=args.data_path, transform=TrainCIFAR100Transforms, download=True)
        self.val_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, transform=ValCIFAR100Transforms)        
        self.test_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, transform=ValCIFAR100Transforms)        
        super().__init__(args)
        
def getDataset(args):
    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(args)
    else:
        dataset = CIFAR100(args)
    return dataset