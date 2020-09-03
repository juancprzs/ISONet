import time
import torch
import torch.nn as nn
from isonet.utils.misc import tprint, pprint_without_newline
from isonet.utils.config import C
from isonet.utils.lips_utils import resnet18_lipschitz, isonet18_lipschitz
from isonet.utils.trades_utils import trades_loss
# Import models
from isonet.models.isonet import ISONet
from isonet.models.resnet import ResNet
# For adversary
from autoattack import AutoAttack
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim, logger, 
            output_dir, trades=False):
        # misc
        self.device = device
        self.output_dir = output_dir
        # data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # nn setting
        self.model = model
        self.optim = optim
        # lr setting
        self.criterion = nn.CrossEntropyLoss()
        # training loop settings
        self.epochs = 1
        # loss settings
        self.train_acc, self.val_acc = [], []
        self.best_valid_acc = 0
        self.best_rob_acc = 0
        self.ce_loss, self.ortho_loss = 0, 0
        # others
        self.ave_time = 0
        self.logger = logger
        self.trades = trades # conduct TRADES adversarial training


    def train(self):
        while self.epochs <= C.SOLVER.MAX_EPOCHS:
            self.adjust_learning_rate()
            self.train_epoch()
            self.val()
            self.epochs += 1

    def train_epoch(self):
        self.model.train()
        self.ce_loss = 0
        self.ortho_loss = 0
        self.ave_time = 0
        correct = 0
        total = 0
        epoch_t = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            iter_t = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optim.zero_grad()
            batch_size = inputs.shape[0]

            if self.trades:
                # default parameters taken from
                # https://github.com/yaodongyu/TRADES/blob/master/train_trades_cifar10.py#L30
                outputs, loss = self.loss(outputs=None, targets=targets, 
                    inputs=inputs)
            else:
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
            loss.backward()
            self.optim.step()

            _, predicted = outputs.max(1)
            total += batch_size

            correct += predicted.eq(targets).sum().item()

            self.ave_time += time.time() - iter_t
            tprint(f'train Epoch: {self.epochs} | {batch_idx + 1} / {len(self.train_loader)} | '
                   f'Acc: {100. * correct / total:.3f} | CE: {self.ce_loss / (batch_idx + 1):.3f} | '
                   f'O: {self.ortho_loss / (batch_idx + 1):.3f} | time: {self.ave_time / (batch_idx + 1):.3f}s')

        info_str = f'train Epoch: {self.epochs} | Acc: {100. * correct / total:.3f} | ' \
                   f'CE: {self.ce_loss / (batch_idx + 1):.3f} | ' \
                   f'time: {time.time() - epoch_t:.2f}s |'
        self.logger.info(info_str)
        pprint_without_newline(info_str)
        self.train_acc.append(100. * correct / total)

    def val(self):
        self.model.eval()
        self.ce_loss = 0
        self.ortho_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        
        # self.snapshot(None) # save ALL snapshots
        acc = correct / total
        self.best_valid_acc = max(self.best_valid_acc, 100. * acc)
        # Lipschitz constant and robust accuracy
        lip_with_pool, lip_no_pool = 0.0, 0.0 # self.get_lipschitz_const()
        rob_acc, nat_acc = self.get_rob_acc(cheap=True)
        assert nat_acc == acc
        if acc > 0.8: # larger than 80% acc
            if 100. * rob_acc > self.best_rob_acc:
                self.snapshot('best')
                self.best_rob_acc = 100. * rob_acc
        else:
            if 100. * acc > self.best_valid_acc:
                self.snapshot('best')
                self.best_valid_acc = 100. * acc

        info_str = f'valid | Acc: {100. * correct / total:.3f} | ' \
                   f'CE: {self.ce_loss / len(self.val_loader):.3f} | ' \
                   f'O: {self.ortho_loss / len(self.val_loader):.3f} | ' \
                   f'best: {self.best_valid_acc:.3f} | ' \
                   f'best_r: {self.best_rob_acc:.3f} | ' \
                   f'L w. pool: {lip_with_pool:5.3E} | ' \
                   f'L no pool: {lip_no_pool:5.3E} | ' \
                   f'Rob. acc: {100. * rob_acc:.3f}' 
        print(info_str)
        self.logger.info(info_str)
        self.val_acc.append(100. * correct / total)

    def get_lipschitz_const(self):
        lip_with_pool = isonet18_lipschitz(self.model.model.module, [3, 32, 32], with_pool=True)
        lip_no_pool = isonet18_lipschitz(self.model.model.module, [3, 32, 32], with_pool=False)
        return lip_with_pool, lip_no_pool

    def get_rob_acc(self, cheap=False):
        adversary = AutoAttack(self.model.forward, norm='Linf', 
            eps=8./255., verbose=False)
        if cheap: # compare to https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py#L230
            adversary.attacks_to_run = ['apgd-ce', 'square']
            adversary.apgd.n_iter = 20
            adversary.square.n_queries = 500
        else:
            print(f'Running EXPENSIVE adversarial attack')
        # run actual attack
        correct, adv_correct, total = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader)
            for idx, (X, y) in enumerate(pbar):
                X, y = X.to(self.device), y.to(self.device)
                total += y.size(0)
                # normal eval
                _, predicted = self.model(X).max(1)
                correct += predicted.eq(y).sum().item()
                # adversarial eval
                x_adv = adversary.run_standard_evaluation(X, y, bs=y.size(0))
                _, adv_predicted = self.model(x_adv).max(1)
                adv_correct += adv_predicted.eq(y).sum().item()
                if idx % 10 == 0:
                    rob_acc, nat_acc = 100.*adv_correct/total, 100.*correct/total
                    info = f'Rob acc: {rob_acc:.3f} | Nat acc: {nat_acc:.3f}'
                    pbar.set_description(info)

        rob_acc, nat_acc = adv_correct / total, correct / total
        return rob_acc, nat_acc

    def loss(self, outputs, targets, inputs=None):
        if self.trades and (outputs is None):
            assert inputs is not None
            outputs, loss = trades_loss(
                self.model, inputs, targets, self.optim, step_size=0.007, 
                epsilon=0.031, perturb_steps=10, beta=6.0, distance='l_inf'
            )
        else:
            loss = self.criterion(outputs, targets)
        self.ce_loss += loss.item()

        if C.ISON.ORTHO_COEFF > 0:
            o_loss = self.model.model.module.ortho()
            self.ortho_loss += o_loss.item()
            loss += o_loss * C.ISON.ORTHO_COEFF
        
        return (outputs, loss) if self.trades else loss

    def adjust_learning_rate(self):
        # if do linear warmup
        if C.SOLVER.WARMUP and self.epochs < C.SOLVER.WARMUP_EPOCH:
            lr = C.SOLVER.BASE_LR * self.epochs / C.SOLVER.WARMUP_EPOCH
        else:
            # normal (step) scheduling
            lr = C.SOLVER.BASE_LR
            for m_epoch in C.SOLVER.LR_MILESTONES:
                if self.epochs > m_epoch:
                    lr *= C.SOLVER.LR_GAMMA

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
            if 'scaling' in param_group:
                param_group['lr'] *= param_group['scaling']

    def snapshot(self, name=None):
        state = {
            'net': self.model.model.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.epochs,
            'train_accuracy': self.train_acc,
            'test_accuracy': self.val_acc
        }
        if name is None:
            torch.save(state, f'{self.output_dir}/{self.epochs}.pt')
        else:
            torch.save(state, f'{self.output_dir}/{name}.pt')
