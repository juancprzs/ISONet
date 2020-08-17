import argparse
import numpy as np
from glob import glob
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot results for ISONet')
parser.add_argument('--exp', default='exp', help='experiment name')
args = parser.parse_args()

pattern = f'outputs/cls/CIFAR10/{args.exp}/*.txt'
f = sorted(glob(pattern))[0]

print(f'File "{f}" will be read')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
with open(f) as ff:
    lines = ff.readlines()
# remove whitespace characers
lines = [x.strip() for x in lines]
# get regularization coefficient
coeff = None
for x in lines:
    if 'ORTHO_COEFF:' in x:
        coeff = float(x.split(':')[1].strip())
        break
# only take validation info
lines = [x for x in lines if 'INFO: valid' in x]
# a line looks like this:
# 2020-08-01 03:24:15,472 isonet INFO: valid | Acc: 87.000 | CE: 0.398 | O: 468.498 | best: 87.000 | Lipschitz: 2.637E+08 | Rob. acc: 0.000
# accuracies
accs = np.array([float(x.split('|')[1].split(':')[1].strip()) for x in lines])
# xent loss
xent = np.array([float(x.split('|')[2].split(':')[1].strip()) for x in lines])
# orthogonality loss
orth = np.array([float(x.split('|')[3].split(':')[1].strip()) for x in lines])
# Lipschitz constants
lips = np.array([float(x.split('|')[5].split(':')[1].strip()) for x in lines])
if 'nostd_' in args.exp:
    print('Not using standardization. Lipschitz const will not be scaled')
else:
    stds = (0.2023, 0.1994, 0.2010)
    scale = min(stds)
    print(f'Using standardization. Lipschitz const is multiplied by 1/{scale}')
    lips = lips / scale
# plot stuff
axes[0,0].plot(accs, label=f'coeff={coeff:2.1E}')
axes[0,1].plot(xent)
axes[1,0].plot(orth)
axes[1,1].plot(lips)

# first axes
axes[0,0].set_xlabel('epochs')
axes[0,0].set_title('Accuracies')
axes[0,0].grid()
axes[0,0].legend(ncol=2)

# second axes
axes[0,1].set_xlabel('epochs')
axes[0,1].set_title('Cross-entropy loss')
axes[0,1].grid()

# third axes
axes[1,0].set_yscale('log')
axes[1,0].set_xlabel('epochs')
axes[1,0].set_title('Orthogonality loss')
axes[1,0].grid()

# fourth axes
axes[1,1].set_yscale('log')
axes[1,1].set_xlabel('epochs')
axes[1,1].set_title('Lipschitz constant')
axes[1,1].grid()

# save figure
plt.tight_layout()
basename = osp.basename(f)
fig_name = f.replace(basename, 'results_fig.png')
print(f'Saving fig for exp "{args.exp}" at {fig_name}', end=' ', flush=True)
plt.savefig(fig_name, dpi=200)
print('done.')