#!/usr/bin/env python
"""sc2.3 -- error vs k (strong vs weak_projection) plus predicted-vs-exact
overlay at the highest k. Requires all results/{form}_k{k}.DONE markers."""
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (
    DEVICE, X0, XF, LAYERS, N_EVAL, K_LIST, FORMS,
    exact_solution, name_for, RESULTS_DIR, CHECKPOINTS_DIR, FIGS_DIR,
)
from pinns_framework import PINArchitecture

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 12,
})


def check_done():
    missing = [name_for(f, k) for f in FORMS for k in K_LIST
               if not os.path.exists(f'{RESULTS_DIR}/{name_for(f, k)}.DONE')]
    if missing:
        print('Missing DONE markers:', missing); sys.exit(1)


def load_net(name):
    net = PINArchitecture('MLP', {'layers': LAYERS, 'activation': 'tanh'}).to(DEVICE)
    net.load_state_dict(torch.load(f'{CHECKPOINTS_DIR}/{name}.pt', map_location=DEVICE))
    net.eval()
    return net


def main():
    check_done()
    rows = []
    for form in FORMS:
        for k in K_LIST:
            s = pd.read_json(f'{RESULTS_DIR}/{name_for(form, k)}_summary.json', typ='series')
            rows.append({'form': form, 'k': k, 'rel_L2': s['rel_L2'], 'L1': s['L1'], 'Linf': s['Linf']})
    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/summary_all.csv', index=False)
    pd.set_option('display.float_format', '{:.4e}'.format)
    print(df.pivot(index='k', columns='form', values='rel_L2').to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    piv = df.pivot(index='k', columns='form', values='rel_L2')
    ax.semilogy(piv.index, piv['strong'], 'o-', color='tab:red', label='strong')
    ax.semilogy(piv.index, piv['weak_projection'], 's-', color='tab:blue', label='weak_projection')
    ax.semilogy(piv.index, piv['agnostic'], '^-', color='tab:green', label='agnostic (IBP, u_x only)')
    ax.set_xlabel('k (frequency of exact solution)')
    ax.set_ylabel('relative L2 error')
    ax.set_title(r"1D Poisson, $u=\sin(k\pi x)$: strong vs weak")
    ax.grid(True, which='both', alpha=0.3); ax.legend()

    ax = axes[1]
    k_show = K_LIST[-1]
    xg_np = np.linspace(X0, XF, N_EVAL)
    xg = torch.tensor(xg_np, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    ue = exact_solution(k_show, xg_np)
    with torch.no_grad():
        up_s = load_net(name_for('strong', k_show))(xg).cpu().numpy().ravel()
        up_w = load_net(name_for('weak_projection', k_show))(xg).cpu().numpy().ravel()
        up_a = load_net(name_for('agnostic', k_show))(xg).cpu().numpy().ravel()
    ax.plot(xg_np, ue, 'k-', lw=2, label='exact')
    ax.plot(xg_np, up_s, '--', color='tab:red', lw=1.5, label='strong')
    ax.plot(xg_np, up_w, '--', color='tab:blue', lw=1.5, label='weak_projection')
    ax.plot(xg_np, up_a, '--', color='tab:green', lw=1.5, label='agnostic (IBP)')
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.set_title(f'Predicted vs exact, k={k_show}')
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle('sc2.3 -- 1D Poisson high-frequency benchmark (VPINN, arXiv:1912.00873)', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc23_error_vs_k.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc23_error_vs_k.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc23_error_vs_k.png')

    # solution overlay for every k
    fig2, axes2 = plt.subplots(1, len(K_LIST), figsize=(3.4 * len(K_LIST), 3.6), sharey=False)
    for ax, k in zip(axes2, K_LIST):
        ue_k = exact_solution(k, xg_np)
        with torch.no_grad():
            ups = load_net(name_for('strong', k))(xg).cpu().numpy().ravel()
            upw = load_net(name_for('weak_projection', k))(xg).cpu().numpy().ravel()
            upa = load_net(name_for('agnostic', k))(xg).cpu().numpy().ravel()
        ax.plot(xg_np, ue_k, 'k-', lw=2, label='exact')
        ax.plot(xg_np, ups, '--', color='tab:red', lw=1.3, label='strong')
        ax.plot(xg_np, upw, '--', color='tab:blue', lw=1.3, label='weak_projection')
        ax.plot(xg_np, upa, '--', color='tab:green', lw=1.3, label='agnostic (IBP)')
        ax.set_title(f'k={k}'); ax.set_xlabel('x'); ax.grid(alpha=0.3)
        if k == K_LIST[0]:
            ax.legend(fontsize=7)
    axes2[0].set_ylabel('u')
    fig2.suptitle('sc2.3 -- predicted vs exact, all k', fontsize=13)
    plt.tight_layout()
    fig2.savefig(f'{FIGS_DIR}/sc23_solutions_all_k.png', dpi=140, bbox_inches='tight')
    fig2.savefig(f'{FIGS_DIR}/sc23_solutions_all_k.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc23_solutions_all_k.png')


if __name__ == '__main__':
    main()
