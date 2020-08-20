import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from isonet.utils.misc import tprint, pprint_without_newline
from isonet.utils.config import C
# Import models
from isonet.models.resnet import ResNet
# For adversary
from autoattack import AutoAttack
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model1, model2, optim, 
            logger, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        # data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # nn setting
        self.model1 = model1
        self.model2 = model2
        self.optim = optim
        # lr setting
        self.criterion = nn.CrossEntropyLoss()
        # training loop settings
        self.epochs = 1
        # loss settings
        self.train_acc, self.val_acc = [], []
        self.best_valid_acc = 0
        self.ce_loss, self.ce_loss1, self.ce_loss2 = 0, 0, 0
        self.disagreement = 0
        # others
        self.ave_time = 0
        self.logger = logger
        self.kl_crit = nn.KLDivLoss(reduction='batchmean')


    def train(self):
        while self.epochs <= C.SOLVER.MAX_EPOCHS:
            self.adjust_learning_rate()
            self.train_epoch()
            self.val()
            self.epochs += 1

    def train_epoch(self):
        self.model1.train()
        self.model2.train()
        self.ce_loss, self.ce_loss1, self.ce_loss2, self.disagreement = 0, 0, 0, 0
        self.ave_time = 0
        correct, correct1, correct2 = 0, 0, 0
        total = 0
        epoch_t = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            iter_t = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optim.zero_grad()
            batch_size = inputs.shape[0]

            outputs1, outputs2 = self.model1(inputs), self.model2(inputs)
            loss = self.loss(outputs1, outputs2, targets)
            loss.backward()
            self.optim.step()

            _, predicted = ((outputs1 + outputs2) / 2.).max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            total += batch_size

            correct += predicted.eq(targets).sum().item()
            correct1 += predicted1.eq(targets).sum().item()
            correct2 += predicted2.eq(targets).sum().item()

            self.ave_time += time.time() - iter_t
            tprint(f'train Epoch: {self.epochs} | {batch_idx + 1} / {len(self.train_loader)} | '
                   f'Acc: {100. * correct / total:.3f} | CE: {self.ce_loss / (batch_idx + 1):.3f} | '
                   f'Acc1: {100. * correct1 / total:.3f} | CE1: {self.ce_loss1 / (batch_idx + 1):.3f} | '
                   f'Acc2: {100. * correct2 / total:.3f} | CE2: {self.ce_loss2 / (batch_idx + 1):.3f} | '
                   f'Disag: {self.disagreement / (batch_idx + 1):.3f} | '
                   f'time: {self.ave_time / (batch_idx + 1):.3f}s')

        info_str = f'train Epoch: {self.epochs} | ' \
                   f'Acc: {100. * correct / total:.3f} | CE: {self.ce_loss / (batch_idx + 1):.3f} | ' \
                   f'Acc1: {100. * correct1 / total:.3f} | CE1: {self.ce_loss1 / (batch_idx + 1):.3f} | ' \
                   f'Acc2: {100. * correct2 / total:.3f} | CE2: {self.ce_loss2 / (batch_idx + 1):.3f} | ' \
                   f'Disag: {self.disagreement / (batch_idx + 1):.3f} | ' \
                   f'time: {time.time() - epoch_t:.2f}s |'
        self.logger.info(info_str)
        pprint_without_newline(info_str)
        self.train_acc.append(100. * correct / total)

    def val(self):
        self.model1.eval()
        self.model2.eval()
        self.ce_loss, self.ce_loss1, self.ce_loss2, self.disagreement = 0, 0, 0, 0
        correct, correct1, correct2 = 0, 0, 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs1, outputs2 = self.model1(inputs), self.model2(inputs)
                loss = self.loss(outputs1, outputs2, targets)

                _, predicted = ((outputs1 + outputs2) / 2.).max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                total += targets.size(0)
                
                correct += predicted.eq(targets).sum().item()
                correct1 += predicted1.eq(targets).sum().item()
                correct2 += predicted2.eq(targets).sum().item()

        if 100. * correct / total > self.best_valid_acc:
            self.snapshot('best')
        self.snapshot('latest')
        self.snapshot(None) # save ALL snapshots
        self.best_valid_acc = max(self.best_valid_acc, 100. * correct / total)
        cheap = self.epochs < C.SOLVER.MAX_EPOCHS # cheap attack for all epochs but the last
        # attack ensemble
        model_forward = lambda x: (self.model1(x) + self.model2(x)) / 2.
        rob_acc, nat_acc = self.get_rob_acc(model_forward, cheap=cheap)
        assert nat_acc == correct / total
        # attack model 1
        rob_acc1, nat_acc1 = self.get_rob_acc(self.model1.forward, cheap=cheap)
        assert nat_acc1 == correct1 / total
        # attack model 2
        rob_acc2, nat_acc2 = self.get_rob_acc(self.model2.forward, cheap=cheap)
        assert nat_acc2 == correct2 / total

        info_str = f'valid | ' \
                   f'Acc: {100. * correct / total:.3f} | CE: {self.ce_loss / len(self.val_loader):.3f} | ' \
                   f'Acc1: {100. * correct1 / total:.3f} | CE: {self.ce_loss1 / len(self.val_loader):.3f} | ' \
                   f'Acc2: {100. * correct2 / total:.3f} | CE: {self.ce_loss2 / len(self.val_loader):.3f} | ' \
                   f'Disag: {self.disagreement / len(self.val_loader):.3f} | ' \
                   f'best: {self.best_valid_acc:.3f} | ' \
                   f'Rob. acc: {100. * rob_acc:.3f} |' \
                   f'Rob. acc1: {100. * rob_acc1:.3f} |' \
                   f'Rob. acc2: {100. * rob_acc2:.3f} |'
        print(info_str)
        self.logger.info(info_str)
        self.val_acc.append(100. * correct / total)

    def get_rob_acc(self, model_forward, cheap=False):
        adversary = AutoAttack(model_forward, norm='Linf', 
            eps=8./255., plus=False, verbose=False)
        if cheap:
            print(f'Running CHEAP adversarial attack')
            adversary = AutoAttack(
                model_forward, norm='Linf', eps=8./255., plus=False, 
                verbose=False, attacks_to_run=['apgd-ce']
            )
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
                _, predicted = model_forward(X).max(1)
                correct += predicted.eq(y).sum().item()
                # adversarial eval
                if cheap: # run cheap version of the attack
                    adversary.cheap()
                x_adv = adversary.run_standard_evaluation(X, y, bs=y.size(0))
                _, adv_predicted = model_forward(x_adv).max(1)
                adv_correct += adv_predicted.eq(y).sum().item()
                if idx % 10 == 0:
                    rob_acc, nat_acc = 100.*adv_correct/total, 100.*correct/total
                    info = f'Rob acc: {rob_acc:.3f} | Nat acc: {nat_acc:.3f}'
                    pbar.set_description(info)

        rob_acc, nat_acc = adv_correct / total, correct / total
        return rob_acc, nat_acc

    def loss(self, outputs1, outputs2, targets):
        # xent of first model
        loss1 = self.criterion(outputs1, targets)
        self.ce_loss1 += loss1.item()
        # xent of second model
        loss2 = self.criterion(outputs2, targets)
        self.ce_loss2 += loss2.item()
        # xent of mixture
        loss_mix = self.criterion((outputs1 + outputs2) / 2., targets)
        self.ce_loss += loss_mix.item()
        # # disagreement term
        # between 1 and 2
        inter_loss_12 = self.kl_crit(F.log_softmax(outputs1, dim=1),
                                     F.softmax(outputs2, dim=1))
        # between 2 and 1
        inter_loss_21 = self.kl_crit(F.log_softmax(outputs2, dim=1),
                                     F.softmax(outputs1, dim=1))
        disagreement = inter_loss_12 + inter_loss_21
        self.disagreement += disagreement.item()

        return loss1 + loss2 - C.SOLVER.DISAG_COEFF * disagreement

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
            'net1': self.model1.model.state_dict(),
            'net2': self.model2.model.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.epochs,
            'train_accuracy': self.train_acc,
            'test_accuracy': self.val_acc
        }
        if name is None:
            torch.save(state, f'{self.output_dir}/{self.epochs}.pt')
        else:
            torch.save(state, f'{self.output_dir}/{name}.pt')
