import time
import torch
import torch.nn as nn
from isonet.utils.misc import tprint, pprint_without_newline
from isonet.utils.config import C
from isonet.utils.lips_utils import resnet18_lipschitz, isonet18_lipschitz
# Import models
from isonet.models.isonet import ISONet
from isonet.models.resnet import ResNet
# For adversary
from autoattack import AutoAttack
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import pdb


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim, logger, output_dir):
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
        self.ce_loss, self.ortho_loss = 0, 0
        # others
        self.ave_time = 0
        self.logger = logger
        # For the adversary. NOTE: mean and std are HARDCODED for CIFAR
        self.adv_val_set = CIFAR10(root=C.DATASET.ROOT, train=False, 
            transform=Compose([ToTensor(),])) # leave between 0 and 1
        self.adv_val_loader = DataLoader(self.adv_val_set, 
            batch_size=self.val_loader.batch_size, num_workers=1)
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)


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

        if 100. * correct / total > self.best_valid_acc:
            self.snapshot('best')
        self.snapshot('latest')
        self.best_valid_acc = max(self.best_valid_acc, 100. * correct / total)
        # Lipschitz constant and robust accuracy
        lips_const = self.get_lipschitz_const()
        cheap = self.epochs < C.SOLVER.MAX_EPOCHS # cheap attack for all epochs but the last
        rob_acc, nat_acc = self.get_rob_acc(cheap=cheap)
        assert nat_acc == correct / total
        info_str = f'valid | Acc: {100. * correct / total:.3f} | ' \
                   f'CE: {self.ce_loss / len(self.val_loader):.3f} | ' \
                   f'O: {self.ortho_loss / len(self.val_loader):.3f} | ' \
                   f'best: {self.best_valid_acc:.3f} | ' \
                   f'Lipschitz: {lips_const:5.3E} | ' \
                   f'Rob. acc: {100. * rob_acc:.3f}' 
        print(info_str)
        self.logger.info(info_str)
        self.val_acc.append(100. * correct / total)

    def get_lipschitz_const(self):
        pdb.set_trace()
        if isinstance(self.model.module, ISONet):
            fun = isonet18_lipschitz
        elif isinstance(self.model.module, ResNet18):
            fun = resnet18_lipschitz
        else:
            raise ValueError('"model" should be either an ISONet or a ResNet18')
            
        return fun(self.model, [3, 32, 32])

    def get_rob_acc(self, cheap=False):
        class ModelWrapper(nn.Module):
            def __init__(self, model, mean, std):
                super(ModelWrapper, self).__init__()
                self.model, self.mean, self.std = model, mean, std
            def forward(self, x):
                return self.model((x - self.mean) / self.std)

        model_wrapper = ModelWrapper(self.model, self.mean, self.std)
        adversary = AutoAttack(model_wrapper.forward, norm='Linf', 
            eps=8./255., plus=False, verbose=False)
        if cheap: # run cheap version of the attack
            print('Running CHEAP adversarial attack')
            adversary.cheap()
        else:
            print('Running EXPENSIVE adversarial attack')
        # run actual attack
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.adv_val_loader:
                X, y = X.to(self.device), y.to(self.device)
                total += y.size(0)
                # normal eval
                _, predicted = self.model(X).max(1)
                correct += predicted.eq(y).sum().item()
                # adversarial eval
                x_adv = adversary.run_standard_evaluation(X, y, bs=y.size(0))
                _, adv_predicted = self.model(x_adv).max(1)
                adv_correct += adv_predicted.eq(y).sum().item()

        rob_acc = adv_correct / total
        nat_acc = correct / total
        return rob_acc, nat_acc

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        self.ce_loss += loss.item()

        if C.ISON.ORTHO_COEFF > 0:
            o_loss = self.model.module.ortho()
            self.ortho_loss += o_loss.item()
            loss += o_loss * C.ISON.ORTHO_COEFF
        return loss

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
            'net': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.epochs,
            'train_accuracy': self.train_acc,
            'test_accuracy': self.val_acc
        }
        if name is None:
            torch.save(state, f'{self.output_dir}/{self.epochs}.pt')
        else:
            torch.save(state, f'{self.output_dir}/{name}.pt')
