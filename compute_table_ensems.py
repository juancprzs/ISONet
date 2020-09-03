import itertools
import pandas as pd
import os.path as osp
from tqdm import tqdm
from isonet.models import *
from train import ModelWrapper
import isonet.utils.dataset as du
from autoattack import AutoAttack
from isonet.utils.config import C
from isonet.models.resnet import ResNet18

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 2./255.
C.merge_from_file('configs/CIF10-ISO18.yaml')
FILENAME = osp.join('.', 'ensembles_results_final_advcoll.csv')
DF_COLS = ['exp', 'run', 'acc1', 'acc2', 'acc3', 'disag_all', 'disag_wrng', \
    'disag_wrng2', 'rob_acc1', 'rob_acc2', 'rob_acc3']

def compute_disag(outputs1, outputs2, targets, d_type):
    if d_type == 'all': # only compute disagreement b/w logits of WRONG classes
        kl_crit = nn.KLDivLoss(reduction='batchmean')
        # between 1 and 2
        inter_loss_12 = kl_crit(F.log_softmax(outputs1, dim=1),
                                F.softmax(outputs2, dim=1))
        # between 2 and 1
        inter_loss_21 = kl_crit(F.log_softmax(outputs2, dim=1),
                                F.softmax(outputs1, dim=1))
    elif d_type == 'wrng':
        kl_crit = nn.KLDivLoss(reduction='batchmean')
        n_insts, n_classes = outputs1.size(0), outputs1.size(1)
        wrng_msk = torch.arange(n_classes).unsqueeze(0).expand(n_insts, -1)
        wrng_msk = wrng_msk.to('cuda') != targets.unsqueeze(1)
        outputs1 = outputs1[wrng_msk].view(n_insts, -1)
        outputs2 = outputs2[wrng_msk].view(n_insts, -1)
        # between 1 and 2
        inter_loss_12 = kl_crit(F.log_softmax(outputs1, dim=1),
                                F.softmax(outputs2, dim=1))
        # between 2 and 1
        inter_loss_21 = kl_crit(F.log_softmax(outputs2, dim=1),
                                F.softmax(outputs1, dim=1))
    elif d_type == 'wrng2':
        kl_crit = nn.KLDivLoss(reduction='none')
        # between 1 and 2
        inter_loss_12 = kl_crit(F.log_softmax(outputs1, dim=1),
                                F.softmax(outputs2, dim=1))
        # between 2 and 1
        inter_loss_21 = kl_crit(F.log_softmax(outputs2, dim=1),
                                F.softmax(outputs1, dim=1))

        n_insts, n_classes = outputs1.size(0), outputs1.size(1)
        wrng_msk = torch.arange(n_classes).unsqueeze(0).expand(n_insts, -1)
        wrng_msk = wrng_msk.to('cuda') != targets.unsqueeze(1)
        inter_loss_12 = inter_loss_12[wrng_msk].view(n_insts, -1).sum(dim=1).mean()
        inter_loss_21 = inter_loss_21[wrng_msk].view(n_insts, -1).sum(dim=1).mean()
    
    disag = inter_loss_12 + inter_loss_21
    return disag.item()


def eval_models(model1, model2, loader):
    model1.eval()
    model2.eval()
    disag1, disag2, disag3 = 0., 0., 0.
    correct1, correct2, correct3, total = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            # counts
            curr_insts = targets.size(0)
            total += curr_insts
            # compute accs
            # model1
            outputs1 = model1(inputs)
            correct1 += outputs1.max(1)[1].eq(targets).sum().item()
            # model2
            outputs2 = model2(inputs)
            correct2 += outputs2.max(1)[1].eq(targets).sum().item()
            # ensemble
            outputs3 = (outputs1 + outputs2) / 2.
            correct3 += outputs3.max(1)[1].eq(targets).sum().item()

            # compute disags
            disag1 += curr_insts * compute_disag(outputs1, outputs2, targets, d_type='all')
            disag2 += curr_insts * compute_disag(outputs1, outputs2, targets, d_type='wrng')
            disag3 += curr_insts * compute_disag(outputs1, outputs2, targets, d_type='wrng2')


    acc1, acc2, acc3 = correct1/total, correct2/total, correct3/total
    disag1, disag2, disag3 = disag1/total, disag2/total, disag3/total
    return acc1, acc2, acc3, disag1, disag2, disag3


def compute_robustness(model_forward, loader):
    # adversary
    adversary = AutoAttack(model_forward, norm='Linf', eps=EPS, plus=False, 
        verbose=False)
    adversary.cheap()
    total, correct, adv_correct = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader)
        for idx, (X, y) in enumerate(pbar):
            X, y = X.to('cuda'), y.to('cuda')
            total += y.size(0)
            # normal eval
            _, predicted = model_forward(X).max(1)
            correct += predicted.eq(y).sum().item()
            # adversarial eval
            x_adv = adversary.run_standard_evaluation(X, y, bs=y.size(0))
            _, adv_predicted = model_forward(x_adv).max(1)
            adv_correct += adv_predicted.eq(y).sum().item()
            if idx % 10 == 0:
                rob_acc, nat_acc = 100.*adv_correct/total, 100.*correct/total
                info = f'Rob acc: {rob_acc:.3f} | Nat acc: {nat_acc:.3f}'
                pbar.set_description(info)

        rob_acc, nat_acc = adv_correct / total, correct / total
        return rob_acc, nat_acc

# net1
net1 = ModelWrapper(torch.nn.DataParallel(ResNet18().to('cuda')))
# net1
net2 = ModelWrapper(torch.nn.DataParallel(ResNet18().to('cuda')))

_, val_loader = du.construct_dataset()

exps = range(2, 7)
runs = range(1, 4)
product = list(itertools.product(exps, runs))
pbar = tqdm(product)
df = { col : [] for col in DF_COLS }
rob_accs = {}

for exp, run in pbar:
    model_path = f'outputs/cls/CIFAR10/ResNet_{exp}_coll_probs_run{run}/best.pt'
    ckpt = torch.load(model_path)
    # net1
    net1.model.load_state_dict(ckpt['net1'])
    # net2
    net2.model.load_state_dict(ckpt['net2'])
    # evaluate (and compute disagreements)
    acc1, acc2, acc3, disag1, disag2, disag3 = eval_models(net1, net2, val_loader)
    # print description
    desc = f'Exp {exp}, run {run}. M1 {acc1:3.4f}, M2 {acc2:3.4f}, ENS ({acc3:3.4f}) | '\
        f'Disags: {disag1:3.4f} (1), {disag2:3.4f} (2), {disag3:3.4f} (3)'
    pbar.set_description(desc)
    # # compute robustness
    # net1
    rob_acc1, nat_acc1 = compute_robustness(lambda x: net1(x), val_loader)
    assert acc1 == nat_acc1
    pbar.set_description(f'M1. Acc: {100.*nat_acc1:2.3f} - Rob. Acc: {100.*rob_acc1:2.3f}')
    # net2
    rob_acc2, nat_acc2 = compute_robustness(lambda x: net2(x), val_loader)
    assert acc2 == nat_acc2
    pbar.set_description(f'M2. Acc: {100.*nat_acc2:2.3f} - Rob. Acc: {100.*rob_acc2:2.3f}')
    # ensemble
    ensem_forward = lambda x: (net1(x) + net2(x)) / 2.
    rob_acc3, nat_acc3 = compute_robustness(ensem_forward, val_loader)
    pbar.set_description(f'ENS. Acc: {100.*nat_acc3:2.3f} - Rob. Acc: {100.*rob_acc3:2.3f}')
    assert nat_acc3 == acc3
    # # add to dict
    # inds
    df['exp'].append(exp)
    df['run'].append(run)
    # accs
    df['acc1'].append(acc1)
    df['acc2'].append(acc2)
    df['acc3'].append(acc3)
    # disags
    df['disag_all'].append(disag1)
    df['disag_wrng'].append(disag2)
    df['disag_wrng2'].append(disag3)
    # robust accs
    df['rob_acc1'].append(rob_acc1)
    df['rob_acc2'].append(rob_acc2)
    df['rob_acc3'].append(rob_acc3)
    # update csv
    pd.DataFrame.from_dict(df).to_csv(FILENAME, index=False)

df = pd.DataFrame.from_dict(df)
df.to_csv(FILENAME, index=False)
print(f'Evaluation finished. Saved to file "{FILENAME}"', df)
