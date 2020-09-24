import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import (Compose, RandomCrop, RandomHorizontalFlip, 
    ToTensor, RandomResizedCrop, Resize, CenterCrop)
from isonet.utils.config import C

def construct_dataset():
    transform = {
        'cifar_train': Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        'cifar_test': Compose([
            ToTensor(),
        ]),
        'ilsvrc2012_train': Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        'ilsvrc2012_test': Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
        ])
    }

    if C.DATASET.NAME == 'CIFAR10':
        train_set = CIFAR10(root=C.DATASET.ROOT, train=True, 
            transform=transform['cifar_train'], download=True)
        # set aside 5k images for validation
        train_set, val_set = random_split(train_set, [45_000, 5_000])
        test_set = CIFAR10(root=C.DATASET.ROOT, train=False, 
            transform=transform['cifar_test'], download=True)
    elif C.DATASET.NAME == 'CIFAR100':
        train_set = CIFAR100(root=C.DATASET.ROOT, train=True, 
            transform=transform['cifar_train'], download=True)
        # set aside 5k images for validation
        train_set, val_set = random_split(train_set, [45_000, 5_000])
        test_set = CIFAR100(root=C.DATASET.ROOT, train=False, 
            transform=transform['cifar_test'], download=True)
    elif C.DATASET.NAME == 'ILSVRC2012': # haven't checked this cases works
        train_dir = os.path.join(C.DATASET.ROOT, 'ILSVRC2012', 'train')
        val_dir = os.path.join(C.DATASET.ROOT, 'ILSVRC2012', 'val')
        train_set = ImageFolder(train_dir, transform['ilsvrc2012_train'])
        val_set = ImageFolder(val_dir, transform['ilsvrc2012_test'])
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        train_set, batch_size=C.SOLVER.TRAIN_BATCH_SIZE, shuffle=True, 
        num_workers=C.DATASET.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=C.SOLVER.TRAIN_BATCH_SIZE, shuffle=True, 
        num_workers=C.DATASET.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=C.SOLVER.TEST_BATCH_SIZE, shuffle=False, 
        num_workers=C.DATASET.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader
