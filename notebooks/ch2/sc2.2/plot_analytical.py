#!/usr/bin/env python
"""Quick look at the Hopf-Lax analytical solution: how steep is the shock by
t=T_END? Snapshots at several times spanning t=0 (Gaussian IC) through
shock formation (t_c) to the evaluation time T_END; a zoom near the shock;
a tight zoom on the exact viscous (Cole-Hopf) boundary layer at small nu;
and the same nu comparison plotted on the normal (full-domain, not zoomed)
x-axis, showing why the tight zoom was needed in the first place. No
training involved."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import exact_solution, viscous_solution, X0, XF, T_END, T_C, FIGS_DIR

x = np.linspace(X0, XF, 2000)
snap_times = sorted(set([0.0, T_C * 0.5, T_C, T_C * 1.3, T_END * 0.66, T_END]))
NUS_TO_SHOW = [(0.001, 'tab:red'), (0.005, 'tab:orange'), (0.01, 'tab:blue')]

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

# ── panel 1: multi-time snapshots, full domain ──────────────────────────────
ax = axes[0, 0]
cmap = plt.cm.viridis
for i, t in enumerate(snap_times):
    u = exact_solution(t, x)
    lbl = f't={t:.3f}' + ('  (shock forms)' if abs(t - T_C) < 1e-9 else '')
    ax.plot(x, u, color=cmap(i / max(1, len(snap_times) - 1)), lw=2, label=lbl)
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.set_title('Hopf-Lax entropy solution, Gaussian IC')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── panel 2: zoom near shock at T_END ───────────────────────────────────────
ax = axes[0, 1]
u_final = exact_solution(T_END, x)
ax.plot(x, u_final, 'k-', lw=2)
grad = np.gradient(u_final, x)
i_steep = np.argmin(grad)
xc = x[i_steep]
ax.axvline(xc, color='red', ls='--', lw=1, alpha=0.6, label=f'steepest slope x={xc:.2f}')
ax.set_xlim(max(X0, xc - 1.2), min(XF, xc + 1.2))
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.set_title(f'Zoom near shock at t={T_END} (post shock, t_c={T_C:.3f})')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── panel 3: tight zoom -- exact viscous boundary layer at small nu ────────
ax = axes[1, 0]
xz = np.linspace(xc - 0.015, xc + 0.015, 4000)
u_entropy_z = exact_solution(T_END, xz)
ax.plot(xz, u_entropy_z, 'k-', lw=2.5, label=r'$\nu=0$ (entropy / Hopf-Lax)')
for nu, color in NUS_TO_SHOW:
    ax.plot(xz, viscous_solution(T_END, xz, nu), '--', lw=1.8, color=color,
            label=rf'$\nu={nu}$ (exact Cole-Hopf)')
ax.axvline(xc, color='gray', ls=':', lw=1, alpha=0.6)
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.set_title(r'Boundary-layer width, tight zoom ($\times$50 vs panel 2)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── panel 4: same nu comparison, normal (full-domain, not zoomed) x-axis ───
ax = axes[1, 1]
u_entropy_full = exact_solution(T_END, x)
ax.plot(x, u_entropy_full, 'k-', lw=2.5, label=r'$\nu=0$ (entropy / Hopf-Lax)')
for nu, color in NUS_TO_SHOW:
    ax.plot(x, viscous_solution(T_END, x, nu), '--', lw=1.6, color=color,
            label=rf'$\nu={nu}$ (exact Cole-Hopf)')
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.set_title(f'Same comparison, normal x-axis (not zoomed) at t={T_END}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle('sc2.2 -- Burgers, Gaussian IC: how close to a shock is t=T_END?', fontsize=13)
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/sc22_analytical_shock.png', dpi=140, bbox_inches='tight')
fig.savefig(f'{FIGS_DIR}/sc22_analytical_shock.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc22_analytical_shock.png')
print(f'max |du/dx| at t={T_END}: {np.abs(grad).max():.2f}  (steepest x={xc:.3f})')
