#!/usr/bin/env python
"""sc2.4 -- compare strong / weak_projection / agnostic / agnostic_weak_time
on the viscous Burgers traveling shock. Error-vs-t table+plot and predicted-
vs-exact solution snapshots (checking specifically whether strong-form looks
noisy/unstable near the shock front despite a possibly-low aggregate error)."""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import DEVICE, X0, XF, NU, S_SHOCK, exact_solution, PARAMS_VAN, RESULTS_DIR, CHECKPOINTS_DIR, FIGS_DIR
from pinns_framework import PINArchitecture, load_pinns

METHODS = ['strong', 'weak_projection', 'agnostic', 'agnostic_weak_time']
STYLE = {'strong': ('tab:red', 'o'), 'weak_projection': ('tab:blue', 's'),
         'agnostic': ('tab:green', '^'), 'agnostic_weak_time': ('tab:brown', 'D')}

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'cm'})


def load_net(name):
    nets = load_pinns(f'{CHECKPOINTS_DIR}/{name}.pt', device=DEVICE)
    net = nets[0].to(DEVICE)
    net.eval()
    return net


def main():
    rows = []
    for m in METHODS:
        df = pd.read_csv(f'{RESULTS_DIR}/{m}_metrics.csv')
        for _, r in df.iterrows():
            rows.append({'method': m, 't': r['t'], 'L1': r['L1'], 'L2': r['L2'], 'Linf': r['Linf']})
    edf = pd.DataFrame(rows)
    edf.to_csv(f'{RESULTS_DIR}/combined_metrics.csv', index=False)
    print(edf.pivot(index='t', columns='method', values='L2').to_string())

    summ = {m: pd.read_json(f'{RESULTS_DIR}/{m}_summary.json', typ='series') for m in METHODS}
    print('\n=== summary (mean_L2, final_L2) ===')
    for m in METHODS:
        print(f'{m:22s} mean_L2={summ[m]["mean_L2"]:.4e}  final_L2={summ[m]["final_L2"]:.4e}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    piv = edf.pivot(index='t', columns='method', values='L2')
    for m in METHODS:
        color, marker = STYLE[m]
        ax.semilogy(piv.index, piv[m], '-', marker=marker, ms=7, color=color, label=m)
    ax.set_xlabel('t'); ax.set_ylabel('relative L2 error vs exact')
    ax.set_title('Error vs time'); ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    xb = np.arange(len(METHODS))
    vals = [summ[m]['mean_L2'] for m in METHODS]
    colors = [STYLE[m][0] for m in METHODS]
    ax.bar(xb, vals, color=colors)
    ax.set_xticks(xb); ax.set_xticklabels(METHODS, rotation=20, ha='right')
    ax.set_yscale('log'); ax.set_ylabel('mean relative L2 error')
    ax.set_title('Mean error, all t')
    ax.grid(True, which='both', axis='y', alpha=0.3)

    fig.suptitle(rf'sc2.4 -- Burgers traveling shock, $\nu={NU}$ (2000 epochs, contention-limited)', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc24_error_vs_t.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc24_error_vs_t.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc24_error_vs_t.png')

    # predicted vs exact, zoomed on the shock front, at t=0.5 and t=1.0
    times = [0.0, 0.5, 1.0]
    fig2, axes2 = plt.subplots(1, len(times), figsize=(5.2 * len(times), 4.6))
    x_full = np.linspace(X0, XF, 2000)
    xg = torch.tensor(x_full, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    nets = {m: load_net(m) for m in METHODS}
    for ax, t in zip(axes2, times):
        ue = exact_solution(t, x_full)
        ax.plot(x_full, ue, 'k-', lw=2.2, label='exact')
        tcol = torch.full((xg.shape[0], 1), float(t), device=DEVICE)
        for m in METHODS:
            color, marker = STYLE[m]
            with torch.no_grad():
                up = nets[m](torch.cat([tcol, xg], dim=1)).cpu().numpy().ravel()
            ax.plot(x_full, up, '--', lw=1.4, color=color, label=m)
        shock_x = S_SHOCK * t
        ax.set_xlim(shock_x - 2, shock_x + 2)
        ax.set_title(f't = {t}  (shock at x={shock_x:.1f})')
        ax.set_xlabel('x'); ax.grid(alpha=0.3)
        if t == times[0]:
            ax.legend(fontsize=8)
    axes2[0].set_ylabel('u')
    fig2.suptitle(rf'sc2.4 -- predicted vs exact near the shock, $\nu={NU}$', fontsize=13)
    plt.tight_layout()
    fig2.savefig(f'{FIGS_DIR}/sc24_solutions_near_shock.png', dpi=140, bbox_inches='tight')
    fig2.savefig(f'{FIGS_DIR}/sc24_solutions_near_shock.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc24_solutions_near_shock.png')


if __name__ == '__main__':
    main()
