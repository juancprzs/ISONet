import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot results for ISONet')
parser.add_argument('--exp', required=True, help='pattern for experiment names', 
    type=str)
args = parser.parse_args()

# run like
# for f in adv_col_{1..5}; do echo $f; python plot_results.py --exp $f; echo ""; done

pattern = f'outputs/cls/CIFAR10/{args.exp}/*.txt'
ff = sorted(glob(pattern))[0]

print('This file will be read: ', ff)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

with open(ff) as fo:
    lines = fo.readlines()
# remove whitespace characers
lines = [x.strip() for x in lines]
# get regularization coefficient
coeff = None
for x in lines:
    if 'DISAG_COEFF:' in x:
        coeff = float(x.split(':')[1].strip())
        break
# only take validation info
lines = [x for x in lines if 'INFO: valid' in x]
# a line looks like this:
# 2020-08-21 15:33:34,336 isonet INFO: valid | Acc: 93.760 | CE: 0.268 | Acc1: 92.480 | CE: 0.332 | Acc2: 92.910 | CE: 0.309 | Disag: 0.435 | best: 93.760 | Rob. acc: 0.000 | Rob. acc1: 0.000 | Rob. acc2: 0.000 |
# accuracies
accs_ens = [float(x.split('|')[1].split(':')[1].strip()) for x in lines]
accs_m1 = [float(x.split('|')[3].split(':')[1].strip()) for x in lines]
accs_m2 = [float(x.split('|')[5].split(':')[1].strip()) for x in lines]
# xent loss
xent_ens = [float(x.split('|')[2].split(':')[1].strip()) for x in lines]
xent_m1 = [float(x.split('|')[4].split(':')[1].strip()) for x in lines]
xent_m2 = [float(x.split('|')[6].split(':')[1].strip()) for x in lines]
# disagreement
disagg = [float(x.split('|')[7].split(':')[1].strip()) for x in lines]
# robust accs
rob_acc_ens = [float(x.split('|')[9].split(':')[1].strip()) for x in lines]
rob_acc_m1 = [float(x.split('|')[10].split(':')[1].strip()) for x in lines]
rob_acc_m2 = [float(x.split('|')[11].split(':')[1].strip()) for x in lines]

# plot stuff
# accs
axes[0,0].plot(accs_ens, label='ENS')
axes[0,0].plot(accs_m1, label='M1')
axes[0,0].plot(accs_m2, label='M2')
axes[0,0].set_ylabel('Accuracies')
axes[0,0].set_xlabel('Epochs')
axes[0,0].grid()
axes[0,0].legend()
# xent losses
axes[0,1].plot(xent_ens, label='ENS')
axes[0,1].plot(xent_m1, label='M1')
axes[0,1].plot(xent_m2, label='M2')
axes[0,1].set_ylabel('XENT Loss')
axes[0,1].set_xlabel('Epochs')
axes[0,1].grid()
axes[0,1].legend()
# disagreement
axes[1,0].plot(disagg)
axes[1,0].set_ylabel('Disagreement')
axes[1,0].set_xlabel('Epochs')
axes[1,0].grid()
# robust accuracy
axes[1,1].plot(rob_acc_ens, label='ENS')
axes[1,1].plot(rob_acc_m1, label='M1')
axes[1,1].plot(rob_acc_m2, label='M2')
axes[1,1].set_ylabel('Robust acc (eps=2/255)')
axes[1,1].set_xlabel('Epochs')
axes[1,1].grid()
axes[1,1].legend()

# save figure
fig.suptitle(f'coeff={coeff:2.1E}')
plt.tight_layout()
fig_name = f'outputs/cls/CIFAR10/{args.exp}/results_fig.png'
print(f'Saving fig for exp "{args.exp}" at {fig_name}', end=' ', flush=True)
plt.savefig(fig_name, dpi=200)
print('done.')