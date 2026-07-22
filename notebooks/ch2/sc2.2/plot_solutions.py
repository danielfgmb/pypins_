#!/usr/bin/env python
"""
sc2.2 -- predicted vs analytical (Hopf-Lax) solution plots, one figure per
nu, one panel per time step (t=0, 0.15, 0.30, 0.45, 0.60), all 6 methods
overlaid on the black analytical curve. Requires all 30 results/*.DONE +
checkpoints/*.pt (from train_one.py). PNG+PDF export.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import DEVICE, X0, XF, TIMES, N_EVAL, exact_solution, TimeSlice, CHECKPOINTS_DIR, FIGS_DIR, RESULTS_DIR
from train_one import METHODS, nu_tag
from pinns_framework import load_pinns

STYLE = {
    'td_strong':                    ('TD-PINNs (strong)',           'o', 'tab:blue'),
    'td_weak_projection':           ('TD-PINNs (weak_projection)',  '^', 'tab:cyan'),
    'td_agnostic':                  ('TD-PINNs (agnostic)',         'v', 'tab:green'),
    'vanilla_strong':               ('Vanilla (strong)',             's', 'tab:orange'),
    'vanilla_weak_projection':      ('Vanilla (weak_projection)',    'D', 'tab:red'),
    'vanilla_agnostic':             ('Vanilla (agnostic)',           'P', 'tab:purple'),
    'vanilla_agnostic_weak_time':   ('Vanilla (agnostic weak-t)',    'X', 'tab:brown'),
}

from common import NU_LIST


def check_done():
    missing = [f'{m}_nu{nu_tag(nu)}' for m in METHODS for nu in NU_LIST
               if not os.path.exists(f'{RESULTS_DIR}/{m}_nu{nu_tag(nu)}.DONE')]
    if missing:
        print(f'Missing {len(missing)} DONE markers, aborting.')
        sys.exit(1)


def load_slices(name):
    nets = load_pinns(f'{CHECKPOINTS_DIR}/{name}.pt', device=DEVICE)
    nets = [n.to(DEVICE) for n in nets]
    if len(nets) == 1:
        return [TimeSlice(nets[0], t).to(DEVICE) for t in TIMES]
    return nets


def main():
    check_done()
    xg_np = np.linspace(X0, XF, N_EVAL)
    xg = torch.tensor(xg_np, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    for nu in NU_LIST:
        tag = nu_tag(nu)
        fig, axes = plt.subplots(1, len(TIMES), figsize=(4.2 * len(TIMES), 4.2), sharey=True)
        for k, t in enumerate(TIMES):
            ax = axes[k]
            u_ex = exact_solution(t, xg_np)
            ax.plot(xg_np, u_ex, 'k-', lw=2.4, label='analytical' if k == 0 else None)
            for method in METHODS:
                leg, marker, color = STYLE[method]
                name = f'{method}_nu{tag}'
                slices = load_slices(name)
                net = slices[k].eval()
                with torch.no_grad():
                    up = net(xg).cpu().numpy().ravel()
                ax.plot(xg_np, up, '--', lw=1.1, color=color,
                        label=leg if k == 0 else None)
            ax.set_title(f't = {t:.2f}')
            ax.set_xlabel('x')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('u')
        axes[0].legend(fontsize=7, loc='upper right')
        fig.suptitle(rf'sc2.2 -- predicted vs Hopf-Lax analytical, $\nu={nu}$', fontsize=14)
        plt.tight_layout()
        fig.savefig(f'{FIGS_DIR}/sc22_solutions_nu{tag}.pdf', bbox_inches='tight')
        fig.savefig(f'{FIGS_DIR}/sc22_solutions_nu{tag}.png', dpi=140, bbox_inches='tight')
        print('saved ->', f'{FIGS_DIR}/sc22_solutions_nu{tag}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
