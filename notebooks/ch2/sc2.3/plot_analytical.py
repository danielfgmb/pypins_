#!/usr/bin/env python
"""sc2.3 -- preliminary look at the analytical solutions before training:
u_exact(x) = sin(k*pi*x) for each k in K_LIST. No training involved."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import exact_solution, X0, XF, K_LIST, FIGS_DIR

x = np.linspace(X0, XF, 1000)

fig, axes = plt.subplots(1, len(K_LIST), figsize=(3.2 * len(K_LIST), 3.4), sharey=True)
for ax, k in zip(axes, K_LIST):
    u = exact_solution(k, x)
    ax.plot(x, u, 'k-', lw=1.8)
    ax.set_title(f'k = {k}')
    ax.set_xlabel('x')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)
axes[0].set_ylabel('u')
fig.suptitle(r'sc2.3 analytical solutions $u(x) = \sin(k\pi x)$, $x \in [-1,1]$', fontsize=13)
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/sc23_analytical.png', dpi=140, bbox_inches='tight')
fig.savefig(f'{FIGS_DIR}/sc23_analytical.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc23_analytical.png')

# overlay, single panel
fig2, ax = plt.subplots(figsize=(7, 4.5))
cmap = plt.cm.viridis
for i, k in enumerate(K_LIST):
    ax.plot(x, exact_solution(k, x), color=cmap(i / max(1, len(K_LIST) - 1)), lw=1.6, label=f'k={k}')
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.set_title(r'$u(x)=\sin(k\pi x)$ for all k (overlay)')
ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
fig2.savefig(f'{FIGS_DIR}/sc23_analytical_overlay.png', dpi=140, bbox_inches='tight')
fig2.savefig(f'{FIGS_DIR}/sc23_analytical_overlay.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc23_analytical_overlay.png')
