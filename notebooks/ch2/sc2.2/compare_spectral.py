#!/usr/bin/env python
"""
sc2.2 -- spectral error analysis at t=T_END (post-shock), one 3-panel figure
per nu (spectrum / per-mode error / per-band error bar chart, same layout as
sc2.0/sc2.1's compare_results.py), plus a consolidated high-band-error vs nu
line plot (6 methods) since that's the direct spectral signature of "how
well is the shock's high-frequency content captured as nu -> 0". Requires
all 30 results/*.DONE + checkpoints/*.pt.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import DEVICE, X0, XF, T_END, NU_LIST, exact_solution, CHECKPOINTS_DIR, FIGS_DIR, RESULTS_DIR
from plot_solutions import load_slices, check_done, STYLE
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

N_SPEC = 512
BANDS = None  # set once kk is known


def spectrum_at(nu, tag):
    x_spec = np.linspace(X0, XF, N_SPEC, endpoint=False)
    xs = torch.tensor(x_spec, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    u_ex = exact_solution(T_END, x_spec)
    U_ex = np.fft.rfft(u_ex) / N_SPEC
    kk = np.arange(len(U_ex))

    spec = {}
    for method in METHODS:
        name = f'{method}_nu{tag}'
        slices = load_slices(name)
        net = slices[-1].eval()   # last time = T_END
        with torch.no_grad():
            u = net(xs).cpu().numpy().ravel()
        spec[method] = np.fft.rfft(u) / N_SPEC
    return kk, U_ex, spec


def main():
    check_done()
    global BANDS
    band_rows = []

    for nu in NU_LIST:
        tag = nu_tag(nu)
        kk, U_ex, spec = spectrum_at(nu, tag)
        BANDS = [(0, 0, 'DC (k=0)'), (1, 4, 'low (1-4)'), (5, 15, 'mid (5-15)'), (16, kk[-1], 'high (16+)')]
        band_labels = [b[2] for b in BANDS]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

        ax = axes[0]
        ax.semilogy(kk[1:], np.abs(U_ex)[1:], 'k-', lw=2.0, label='analytical')
        for method in METHODS:
            leg, marker, color = STYLE[method]
            ax.semilogy(kk[1:], np.abs(spec[method])[1:], '-', color=color, lw=1.3, label=leg)
        ax.set_xlabel('wavenumber k'); ax.set_ylabel(r'$|\hat u_k|$')
        ax.set_title(f'Spectrum at t = {T_END}'); ax.set_xlim(0, 60)
        ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7)

        ax = axes[1]
        for method in METHODS:
            leg, marker, color = STYLE[method]
            ax.semilogy(kk[1:], np.abs(spec[method] - U_ex)[1:], '-', color=color, lw=1.3, label=leg)
        ax.set_xlabel('wavenumber k'); ax.set_ylabel(r'$|\hat u^{pred}_k - \hat u^{exact}_k|$')
        ax.set_title('Per-mode error'); ax.set_xlim(0, 60)
        ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7)

        ax = axes[2]
        band_err = {method: [np.abs(spec[method] - U_ex)[lo:hi + 1].sum() for lo, hi, _ in BANDS]
                    for method in METHODS}
        xb = np.arange(len(BANDS)); w = 0.8 / len(METHODS)
        for i, method in enumerate(METHODS):
            leg, marker, color = STYLE[method]
            ax.bar(xb + (i - (len(METHODS) - 1) / 2) * w, band_err[method], width=w, color=color, label=leg)
        ax.set_yscale('log')
        ax.set_xticks(xb); ax.set_xticklabels(band_labels, rotation=15)
        ax.set_ylabel('summed spectral error'); ax.set_title('Error per frequency band')
        ax.grid(True, which='both', axis='y', alpha=0.3); ax.legend(fontsize=7)

        fig.suptitle(rf'sc2.2 spectral error, $\nu={nu}$ (t={T_END}, post-shock)', fontsize=14)
        plt.tight_layout()
        fig.savefig(f'{FIGS_DIR}/sc22_spectral_nu{tag}.pdf', bbox_inches='tight')
        fig.savefig(f'{FIGS_DIR}/sc22_spectral_nu{tag}.png', dpi=140, bbox_inches='tight')
        print('saved ->', f'{FIGS_DIR}/sc22_spectral_nu{tag}.pdf')
        plt.close(fig)

        for method in METHODS:
            for (lo, hi, blabel), err in zip(BANDS, band_err[method]):
                band_rows.append({'nu': nu, 'method': method, 'band': blabel, 'error': err})

    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(f'{RESULTS_DIR}/spectral_band_error_vs_nu.csv', index=False)

    # ── consolidated: high-band error vs nu, 6 lines (the shock's HF content) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, band_name in zip(axes, ['mid (5-15)', 'high (16+)']):
        sub_all = band_df[band_df.band == band_name]
        for method in METHODS:
            leg, marker, color = STYLE[method]
            sub = sub_all[sub_all.method == method].sort_values('nu')
            ax.loglog(sub['nu'], sub['error'], '-', marker=marker, ms=7, lw=1.6, color=color, label=leg)
        ax.invert_xaxis()
        ax.set_xlabel(r'$\nu$'); ax.set_ylabel('summed spectral error')
        ax.set_title(f'Band: {band_name}')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('sc2.2 -- spectral error vs nu, mid/high-frequency bands (shock content)', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc22_spectral_vs_nu.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc22_spectral_vs_nu.png', dpi=140, bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc22_spectral_vs_nu.pdf')
    plt.close(fig)

    print('\nAll sc2.2 spectral outputs written under results/ and figs/.')


if __name__ == '__main__':
    main()
