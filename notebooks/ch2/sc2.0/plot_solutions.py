#!/usr/bin/env python
"""
sc2.0 -- plot each trained method's predicted u(t, x) against the analytical
toothsaw solution, one panel per training time level.
Writes figs/sc20_solutions.png (+ .pdf). Run manually: python plot_solutions.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (
    DEVICE, X0, XF, TIMES, toothsaw, TimeSlice, CHECKPOINTS_DIR, FIGS_DIR,
)
from pinns_framework import load_pinns

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size'       : 12,
    'axes.titlesize'  : 12,
    'axes.labelsize'  : 11,
    'legend.fontsize' : 8,
})

METHODS = {
    'TD-PINNs (MLP)': ('td_mlp',      'tab:blue'),
    'TD-PINNs (KAN)': ('td_kan',      'tab:cyan'),
    'Vanilla (MLP)':  ('vanilla_mlp', 'tab:orange'),
    'Vanilla (KAN)':  ('vanilla_kan', 'tab:red'),
}


def load_slices(name):
    nets = load_pinns(f'{CHECKPOINTS_DIR}/{name}.pt', device=DEVICE)
    nets = [n.to(DEVICE) for n in nets]
    if len(nets) == 1:
        return [TimeSlice(nets[0], t).to(DEVICE) for t in TIMES]
    return nets


N_X = 400
x_grid = np.linspace(X0, XF, N_X)
xs = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1).to(DEVICE)

preds = {}
for label, (name, color) in METHODS.items():
    slices = load_slices(name)
    row = []
    for net in slices:
        net.eval()
        with torch.no_grad():
            row.append(net(xs).cpu().numpy().ravel())
    preds[label] = row

n_t = len(TIMES)
ncols = 3
nrows = int(np.ceil(n_t / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), sharey=True)
axes = np.atleast_1d(axes).ravel()

for k, t in enumerate(TIMES):
    ax = axes[k]
    u_ex = toothsaw(t, x_grid)
    ax.plot(x_grid, u_ex, 'k-', lw=2.4, label='analytical', zorder=5)
    for label, (name, color) in METHODS.items():
        ax.plot(x_grid, preds[label][k], '--', lw=1.5, color=color, label=label)
    ax.set_title(f't = {t:.3f}')
    ax.set_xlabel('x')
    if k % ncols == 0:
        ax.set_ylabel('u(t, x)')
    ax.grid(alpha=0.3)

for k in range(n_t, len(axes)):
    axes[k].axis('off')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Burgers sawtooth (sc2.0) -- predicted vs analytical solution', fontsize=15)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(f'{FIGS_DIR}/sc20_solutions.png', dpi=140, bbox_inches='tight')
fig.savefig(f'{FIGS_DIR}/sc20_solutions.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc20_solutions.png')
print('saved ->', f'{FIGS_DIR}/sc20_solutions.pdf')
plt.close(fig)
