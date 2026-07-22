#!/usr/bin/env python
"""
sc2.1 -- extra plots for visual inspection:
  1. figs/sc21_toothsaw.png   -- analytic sawtooth solution (heatmap + snapshots)
  2. figs/sc21_spectral.png   -- PNG copy of the spectral-error figure (for viewing)
Run manually: python plot_extra.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (
    DEVICE, X0, XF, NT, TIMES, T_END, NU, toothsaw, TimeSlice,
    CHECKPOINTS_DIR, FIGS_DIR,
)
from pinns_framework import load_pinns

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size'       : 12,
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 12,
    'legend.fontsize' : 9,
})

# ── 1. toothsaw analytic solution ───────────────────────────────────────────
NX, NT_FINE = 400, 200
x_grid = np.linspace(X0, XF, NX)
t_grid = np.linspace(0.0, T_END, NT_FINE)
U = np.array([toothsaw(t, x_grid) for t in t_grid])  # (NT_FINE, NX)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
im = ax.pcolormesh(x_grid, t_grid, U, shading='auto', cmap='viridis')
fig.colorbar(im, ax=ax, label='u(t, x)')
ax.set_xlabel('x'); ax.set_ylabel('t')
ax.set_title(f'Burgers sawtooth analytic solution ($\\nu$={NU})')

ax = axes[1]
cmap = plt.get_cmap('plasma')
for i, t in enumerate(TIMES):
    ax.plot(x_grid, toothsaw(t, x_grid), '-', lw=1.8,
            color=cmap(i / max(len(TIMES) - 1, 1)), label=f't={t:.3f}')
ax.set_xlabel('x'); ax.set_ylabel('u(t, x)')
ax.set_title('Snapshots at training time levels')
ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle('Burgers sawtooth -- analytic (Hopf-Cole) solution', fontsize=14)
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/sc21_toothsaw.png', dpi=140, bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc21_toothsaw.png')
fig.savefig(f'{FIGS_DIR}/sc21_toothsaw.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc21_toothsaw.pdf')
plt.close(fig)

# ── 2. PNG copies of the existing comparison figures (for inline viewing) ──
import subprocess
for name in ('sc21_errors', 'sc21_conservation', 'sc21_spectral'):
    src = f'{FIGS_DIR}/{name}.pdf'
    dst = f'{FIGS_DIR}/{name}.png'
    try:
        subprocess.run(['pdftoppm', '-png', '-r', '140', '-singlefile', src, dst[:-4]],
                        check=True)
        print('saved ->', dst)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f'could not convert {src} -> png ({e}); pdftoppm not available?')
