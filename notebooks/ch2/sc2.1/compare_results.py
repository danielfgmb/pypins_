#!/usr/bin/env python
"""
sc2.1 -- compare the four completed runs (td_weak_projection, td_agnostic,
vanilla_weak_projection, vanilla_agnostic): error norms, conservation,
spectral error per frequency band. Reads results/*_metrics.csv and
checkpoints/*.pt written by the four train_*.py scripts; writes combined
CSVs and PDFs into results/ and figs/.

Run manually once all four DONE markers exist, e.g.:
    python compare_results.py
or let launch_all.sh's waiter window trigger it automatically.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (
    DEVICE, X0, XF, NT, TIMES, X_DIM, N_EVAL, toothsaw, TimeSlice,
    RESULTS_DIR, CHECKPOINTS_DIR, FIGS_DIR,
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

METHODS = {
    'TD-PINNs (strong)':          ('td_strong',                    'v', 'tab:green'),
    'TD-PINNs (weak_projection)': ('td_weak_projection',           'o', 'tab:blue'),
    'TD-PINNs (agnostic)':        ('td_agnostic',                  '^', 'tab:cyan'),
    'Vanilla (strong)':           ('vanilla_strong',                'P', 'tab:purple'),
    'Vanilla (weak_projection)':  ('vanilla_weak_projection',       's', 'tab:orange'),
    'Vanilla (agnostic)':         ('vanilla_agnostic',              'D', 'tab:red'),
    'Vanilla (agnostic weak-t)':  ('vanilla_agnostic_weak_time',    'X', 'tab:brown'),
}


def check_done():
    missing = [name for _, (name, _, _) in METHODS.items()
               if not os.path.exists(f'{RESULTS_DIR}/{name}.DONE')]
    if missing:
        print('Not all runs are finished yet -- missing DONE markers for:', missing)
        print('Re-run this script once every results/<name>.DONE file exists.')
        sys.exit(1)


def load_slices(name):
    """Load a checkpoint and return a list of per-time spatial nets, same
    convention as the notebook (TD: one net per step; Vanilla: one net
    wrapped per time via TimeSlice)."""
    nets = load_pinns(f'{CHECKPOINTS_DIR}/{name}.pt', device=DEVICE)
    nets = [n.to(DEVICE) for n in nets]  # load_pinns' device= only sets torch.load's
    # map_location; load_state_dict copies into a freshly-constructed (CPU) module, so
    # the net itself stays on CPU unless we move it explicitly.
    if len(nets) == 1:  # Vanilla checkpoint: single space-time net
        return [TimeSlice(nets[0], t).to(DEVICE) for t in TIMES]
    return nets  # TD checkpoint: already one net per time level


def main():
    check_done()

    # ── load per-method metrics ─────────────────────────────────────────
    results = {}
    for label, (name, marker, color) in METHODS.items():
        df = pd.read_csv(f'{RESULTS_DIR}/{name}_metrics.csv', index_col=0)
        summary = pd.read_json(f'{RESULTS_DIR}/{name}_summary.json', typ='series')
        results[label] = {'name': name, 'df': df, 'marker': marker, 'color': color,
                           'epochs': summary['epochs'], 'wall': summary['wall_s']}

    # ── combined summary CSV ────────────────────────────────────────────
    rows = []
    for label, r in results.items():
        d = r['df']
        rows.append({'method': label, 'epochs': r['epochs'],
                     'mean_L1': d['L1'].mean(), 'mean_L2': d['L2'].mean(), 'mean_Linf': d['Linf'].mean(),
                     'final_L1': d['L1'].iloc[-1], 'final_L2': d['L2'].iloc[-1], 'final_Linf': d['Linf'].iloc[-1],
                     'wall_s': r['wall']})
    summary_df = pd.DataFrame(rows).set_index('method')
    summary_df.to_csv(f'{RESULTS_DIR}/summary_all.csv')
    pd.set_option('display.float_format', '{:.4e}'.format)
    print('=== weak_projection vs agnostic, TD-PINNs vs Vanilla -- error summary ===')
    print(summary_df.to_string())

    # ── combined per-time error/conservation CSV ────────────────────────
    combined = []
    for label, r in results.items():
        d = r['df'].copy()
        d.insert(0, 'method', label)
        d.insert(1, 't', d.index)
        combined.append(d.reset_index(drop=True))
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(f'{RESULTS_DIR}/combined_per_time_metrics.csv', index=False)

    # ── error norms plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (label_ax, col) in zip(axes, [(r'$L_1$', 'L1'), (r'$L_2$', 'L2'), (r'$L_\infty$', 'Linf')]):
        for label, r in results.items():
            ax.semilogy(TIMES, r['df'][col].values, '-', marker=r['marker'], ms=6, lw=1.5,
                        color=r['color'], label=label)
        ax.set_xlabel('t'); ax.set_ylabel(f'{label_ax} error vs exact')
        ax.set_title(f'{label_ax} error per time level')
        ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle('Burgers sawtooth (sc2.1) -- weak_projection vs agnostic, TD-PINNs vs Vanilla', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc21_errors.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc21_errors.pdf')
    plt.close(fig)

    # ── conservation plot ─────────────────────────────────────────────────
    df_ref = next(iter(results.values()))['df']  # exact_mass / exact_energy are method-independent
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(TIMES, df_ref['exact_mass'].values, 'k-', lw=2.4, label='analytical')
    for label, r in results.items():
        ax.plot(TIMES, r['df']['mass'].values, '-', marker=r['marker'], ms=6, lw=1.5,
                color=r['color'], label=label)
    ax.set_xlabel('t'); ax.set_ylabel(r'mass  $\int u\,dx$')
    ax.set_title('Mass conservation (flat = conserved)')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(TIMES, df_ref['exact_energy'].values, 'k-', lw=2.4, label='analytical')
    for label, r in results.items():
        ax.plot(TIMES, r['df']['energy'].values, '-', marker=r['marker'], ms=6, lw=1.5,
                color=r['color'], label=label)
    ax.set_xlabel('t'); ax.set_ylabel(r'energy  $\int u^2\,dx$')
    ax.set_title(r'Energy dissipation ($dE/dt \leq 0$, decreasing)')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc21_conservation.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc21_conservation.pdf')
    plt.close(fig)

    # ── spectral test (needs the trained checkpoints) ──────────────────
    N_SPEC = 512
    x_spec = np.linspace(X0, XF, N_SPEC, endpoint=False)
    xs     = torch.tensor(x_spec, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    kf     = NT
    t_spec = TIMES[kf]

    u_ex = toothsaw(t_spec, x_spec)
    U_ex = np.fft.rfft(u_ex) / N_SPEC
    kk   = np.arange(len(U_ex))

    spec = {}
    for label, (name, marker, color) in METHODS.items():
        slices = load_slices(name)
        net = slices[kf].eval()
        with torch.no_grad():
            u = net(xs).cpu().numpy().ravel()
        spec[label] = np.fft.rfft(u) / N_SPEC

    BANDS = [(0, 0, 'DC (k=0)'), (1, 4, 'low (1-4)'), (5, 15, 'mid (5-15)'), (16, kk[-1], 'high (16+)')]
    band_labels = [b[2] for b in BANDS]
    band_err = {label: [np.abs(spec[label] - U_ex)[lo:hi + 1].sum() for lo, hi, _ in BANDS]
                for label in results}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.semilogy(kk[1:], np.abs(U_ex)[1:], 'k-', lw=2.0, label='analytical')
    for label, r in results.items():
        ax.semilogy(kk[1:], np.abs(spec[label])[1:], '-', color=r['color'], lw=1.3, label=label)
    ax.set_xlabel('wavenumber k'); ax.set_ylabel(r'$|\hat u_k|$')
    ax.set_title(f'Spectrum at t = {t_spec:.3f}'); ax.set_xlim(0, 40)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    for label, r in results.items():
        ax.semilogy(kk[1:], np.abs(spec[label] - U_ex)[1:], '-', color=r['color'], lw=1.3, label=label)
    ax.set_xlabel('wavenumber k'); ax.set_ylabel(r'$|\hat u^{pred}_k - \hat u^{exact}_k|$')
    ax.set_title('Per-mode error'); ax.set_xlim(0, 40)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7)

    ax = axes[2]
    xb = np.arange(len(BANDS)); w = 0.8 / len(results)
    for i, (label, r) in enumerate(results.items()):
        ax.bar(xb + (i - (len(results) - 1) / 2) * w, band_err[label], width=w, color=r['color'], label=label)
    ax.set_yscale('log')
    ax.set_xticks(xb); ax.set_xticklabels(band_labels, rotation=15)
    ax.set_ylabel('summed spectral error'); ax.set_title('Error per frequency band')
    ax.grid(True, which='both', axis='y', alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(f'Spectral error analysis (t = {t_spec:.3f})', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc21_spectral.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc21_spectral.pdf')
    plt.close(fig)

    band_df = pd.DataFrame(band_err, index=band_labels)
    band_df.to_csv(f'{RESULTS_DIR}/spectral_band_error.csv')
    print('\n=== summed |spectral error| per band (final time) ===')
    print(band_df.to_string())

    print('\nAll comparison outputs written under results/ and figs/.')


if __name__ == '__main__':
    main()
