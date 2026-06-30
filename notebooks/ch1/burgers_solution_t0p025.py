# ─────────────────────────────────────────────────────────────────────────────
# Solution comparison at t = 0.025  —  TD-PINNs vs Vanilla vs analytical
#
# Drop-in cell for `burgers_tdpinns_vs_vanilla_sawtooth.ipynb`.  Run it AFTER the
# notebook's training/eval cells (it reuses the in-memory `results`, so there is
# no need to retrain).  In the notebook either paste this as a new cell or use:
#       %run burgers_solution_t0p025.py
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Reuse the notebook's globals (fall back gracefully if a name is missing)
FIGDIR  = globals().get('FIGDIR', 'figs')
METHODS = globals().get('METHODS', {'TD-PINNs': ('o', 'tab:blue'),
                                    'Vanilla': ('s', 'tab:orange')})
N_EVAL  = globals().get('N_EVAL', 1000)

T_TARGET = 0.025
kf       = int(np.argmin([abs(t - T_TARGET) for t in times]))   # nearest time level
t_show   = times[kf]

x_eval = torch.linspace(X0, XF, N_EVAL).unsqueeze(1).to(DEVICE)
xev_np = x_eval.cpu().numpy().ravel()
u_ex   = toothsaw(t_show, xev_np)

# Evaluate every method once at this time level
preds = {}
for m, r in results.items():
    net = r['slices'][kf].eval()
    with torch.no_grad():
        u = net(x_eval).cpu().numpy().ravel()
    preds[m] = u

fig, (ax, axe) = plt.subplots(1, 2, figsize=(13, 4.5))

# (left) solution overlay
ax.plot(xev_np, u_ex, 'k-', lw=2.4, label='analytical')
for m, u in preds.items():
    mk, c = METHODS[m]
    ax.plot(xev_np, u, '--', color=c, lw=1.6, label=f'{m}  (Linf={np.abs(u - u_ex).max():.2e})')
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.set_title(f'Solution at t = {t_show:.3f}')
ax.grid(alpha=0.3); ax.legend()

# (right) pointwise error
for m, u in preds.items():
    mk, c = METHODS[m]
    axe.plot(xev_np, u - u_ex, '-', color=c, lw=1.4, label=m)
axe.axhline(0, color='gray', lw=0.6)
axe.set_xlabel('x'); axe.set_ylabel(r'$u_{pred} - u_{exact}$')
axe.set_title(f'Pointwise error at t = {t_show:.3f}')
axe.grid(alpha=0.3); axe.legend()

plt.tight_layout()
os.makedirs(FIGDIR, exist_ok=True)
savepath = os.path.join(FIGDIR, 'burgers_solution_t0p025.pdf')
fig.savefig(savepath, bbox_inches='tight')
print('saved ->', savepath)
plt.show()
