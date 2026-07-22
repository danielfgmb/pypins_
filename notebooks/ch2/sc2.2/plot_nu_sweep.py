#!/usr/bin/env python
"""
sc2.2 -- once all 30 (method, nu) runs are DONE, plot final-time (t=T_END,
post-shock) relative-L2 error vs nu, one line per method (6 lines), nu on a
log x-axis. Shows how strong vs weak/agnostic formulations degrade as
nu -> 0 (shock regime, reduced regularity of the entropy solution).

Run manually once every results/<method>_nu<tag>.DONE exists:
    python plot_nu_sweep.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import NU_LIST, RESULTS_DIR, FIGS_DIR, T_END, T_C
from train_one import METHODS, nu_tag

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size'       : 12,
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 12,
    'legend.fontsize' : 9,
})

STYLE = {
    'td_strong':                    ('TD-PINNs (strong)',           'o', 'tab:blue'),
    'td_weak_projection':           ('TD-PINNs (weak_projection)',  '^', 'tab:cyan'),
    'td_agnostic':                  ('TD-PINNs (agnostic)',         'v', 'tab:green'),
    'vanilla_strong':               ('Vanilla (strong)',             's', 'tab:orange'),
    'vanilla_weak_projection':      ('Vanilla (weak_projection)',    'D', 'tab:red'),
    'vanilla_agnostic':             ('Vanilla (agnostic)',           'P', 'tab:purple'),
    'vanilla_agnostic_weak_time':   ('Vanilla (agnostic weak-t)',    'X', 'tab:brown'),
}


def check_done():
    missing = [f'{m}_nu{nu_tag(nu)}' for m in METHODS for nu in NU_LIST
               if not os.path.exists(f'{RESULTS_DIR}/{m}_nu{nu_tag(nu)}.DONE')]
    if missing:
        print(f'Not all runs finished yet -- missing {len(missing)}/{len(METHODS) * len(NU_LIST)} DONE markers:')
        for m in missing:
            print('  ', m)
        sys.exit(1)


def main():
    check_done()

    rows = []
    for method in METHODS:
        for nu in NU_LIST:
            name = f'{method}_nu{nu_tag(nu)}'
            s = pd.read_json(f'{RESULTS_DIR}/{name}_summary.json', typ='series')
            rows.append({'method': method, 'nu': nu,
                         'final_L1': s['final_L1'], 'final_L2': s['final_L2'],
                         'final_Linf': s['final_Linf'], 'mean_L2': s['mean_L2'],
                         'wall_s': s['wall_s']})
    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/nu_sweep_summary.csv', index=False)
    pd.set_option('display.float_format', '{:.4e}'.format)
    print('=== sc2.2 nu-sweep -- final-time (t=T_END) error summary ===')
    print(df.pivot(index='nu', columns='method', values='final_L2').to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, col, label in zip(
        axes, ['final_L1', 'final_L2', 'final_Linf'], [r'$L_1$', r'$L_2$', r'$L_\infty$']
    ):
        for method in METHODS:
            leg, marker, color = STYLE[method]
            sub = df[df.method == method].sort_values('nu')
            ax.loglog(sub['nu'], sub[col], '-', marker=marker, ms=7, lw=1.6,
                      color=color, label=leg)
        ax.set_xlabel(r'$\nu$')
        ax.set_ylabel(f'{label} error @ t={T_END} vs Hopf-Lax')
        ax.invert_xaxis()
        ax.set_title(f'{label} error at final time vs viscosity')
        ax.grid(True, which='both', alpha=0.3)
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle(rf'Burgers, Gaussian IC — error vs $\nu$ near shock '
                 rf'($t_c \approx {T_C:.3f}$, eval $t={T_END}$)', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc22_nu_sweep.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc22_nu_sweep.png', dpi=140, bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc22_nu_sweep.pdf')
    plt.close(fig)

    # single-panel L2 version (the plot the user asked for: nu on x, error on y, 6 lines)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method in METHODS:
        leg, marker, color = STYLE[method]
        sub = df[df.method == method].sort_values('nu')
        ax.loglog(sub['nu'], sub['final_L2'], '-', marker=marker, ms=8, lw=1.8,
                  color=color, label=leg)
    ax.invert_xaxis()
    ax.set_xlabel(r'$\nu$ (viscosity)')
    ax.set_ylabel(rf'relative $L_2$ error at $t={T_END}$ (post-shock) vs Hopf-Lax')
    ax.set_title('Strong vs weak/agnostic formulations near the shock')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc22_nu_sweep_L2.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc22_nu_sweep_L2.png', dpi=140, bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc22_nu_sweep_L2.pdf')
    plt.close(fig)

    print('\nAll comparison outputs written under results/ and figs/.')


if __name__ == '__main__':
    main()
