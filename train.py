import os
import argparse
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from isonet.utils.config import C

import isonet.utils.dataset as du
import isonet.utils.optim as ou
import isonet.utils.logger as lu

from isonet.models import *
from isonet.models.resnet import ResNet18
from torchvision.models import mobilenet_v2
from isonet.trainer import Trainer

import numpy as np
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--output', default='default', type=str)
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--probs', action='store_true')
    parser.add_argument('--mobilenet', action='store_true')
    args = parser.parse_args()
    return args


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        # # standardization params (hardcoded for CIFAR)
        # mean
        self.mean = [0.4914, 0.4822, 0.4465]
        self.mean = torch.tensor(self.mean).view(1, 3, 1, 1).to('cuda')
        # std
        self.std = [0.2023, 0.1994, 0.2010]
        self.std = torch.tensor(self.std).view(1, 3, 1, 1).to('cuda')

    def forward(self, x):
        x = (x - self.mean) / self.std # normalize
        return self.model(x)

def main():
    args = arg_parse()
    # ---- reproducibility ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    # disable imagenet dataset jpeg warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    # ---- setup GPUs ----
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    assert torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    cudnn.benchmark = True
    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    C.SOLVER.TRAIN_BATCH_SIZE *= num_gpus
    C.SOLVER.TEST_BATCH_SIZE *= num_gpus
    C.SOLVER.BASE_LR *= num_gpus
    C.freeze()
    # ---- setup logger and output ----
    output_dir = os.path.join(C.OUTPUT_DIR, C.DATASET.NAME, args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger = lu.construct_logger('isonet', output_dir)
    logger.info('Using {} GPUs'.format(num_gpus))
    logger.info(C.dump())
    # ---- setup dataset ----
    train_loader, val_loader = du.construct_dataset()

    # net1
    net1 = mobilenet_v2(num_classes=10) if args.mobilenet else ResNet18()
    net1.to(torch.device('cuda'))
    net1 = torch.nn.DataParallel(
        net1, device_ids=list(range(args.gpus.count(',') + 1))
    )
    net1 = ModelWrapper(net1)
    # net1
    net2 = mobilenet_v2(num_classes=10) if args.mobilenet else ResNet18()
    net2.to(torch.device('cuda'))
    net2 = torch.nn.DataParallel(
        net2, device_ids=list(range(args.gpus.count(',') + 1))
    )
    net2 = ModelWrapper(net2)

    optim = ou.construct_optim(net1, net2, num_gpus)

    trainer = Trainer(
        torch.device('cuda'),
        train_loader,
        val_loader,
        net1,
        net2,
        optim,
        logger,
        output_dir,
        probs=args.probs
    )

    if args.resume:
        cp = torch.load(args.resume)
        trainer.model.load_state_dict(cp['net'])
        trainer.optim.load_state_dict(cp['optim'])
        trainer.epochs = cp['epoch']
        trainer.train_acc = cp['train_accuracy']
        trainer.val_acc = cp['test_accuracy']

    trainer.train()


if __name__ == '__main__':
    main()
