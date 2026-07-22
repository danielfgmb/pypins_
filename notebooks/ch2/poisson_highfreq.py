#!/usr/bin/env python
"""
Quick, cheap literature-grounded case where strong-form PINN is expected to
struggle relative to the weak (Petrov-Galerkin) form: 1D Poisson with a
high-frequency manufactured solution, the canonical benchmark from Kharazmi,
Zhang & Karniadakis, "Variational Physics-Informed Neural Networks For
Solving Partial Differential Equations" (VPINN, arXiv:1912.00873) and its
hp-VPINN follow-up (arXiv:2003.05385).

    -u''(x) = f(x),  x in [-1, 1],   u(-1) = u(1) = 0
    u_exact(x) = sin(k*pi*x)   =>   f(x) = k^2 * pi^2 * sin(k*pi*x)

Why strong form is expected to fail as k grows: the strong-form residual
needs u_xx pointwise via autodiff through a *fixed-size* MLP -- fitting a
high-frequency, unit-amplitude target forces large curvature into a small
number of parameters, and the second derivative amplifies whatever
representation error remains (same spectral-bias mechanism as Wang, Teng &
Perdikaris, "Understanding and mitigating gradient pathologies in PINNs").
The weak form only ever needs u_x (one integration by parts moves the second
derivative onto the *known, exact* test functions v_k), so it never
differentiates the network's own approximation error twice.

Static problem (no time-stepping), so this bypasses PINTrainer entirely --
just PINArchitecture + PINLossFunction + a bare Adam loop. Small net, few
epochs, k in {1, 2, 4, 8, 16}: seconds per run, not another long sweep.
"""
import sys, time
sys.path.insert(0, '/home/df.gomezb/Tesis')

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pinns_framework import PINArchitecture, PINSpatialDimension, PINLossFunction

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X0, XF = -1.0, 1.0
N_COLLOC = 200
N_EVAL = 500
EPOCHS = 8000          # still small/fast, but 3000 wasn't enough for either
                        # form to resolve k=16 at all (pure underfitting, not
                        # the intended second-derivative-noise failure mode)
LR = 2e-3
LAYERS = (1, 20, 20, 1)
K_LIST = [1, 2, 4, 8, 16]
N_TEST = 20

X_DIM = PINSpatialDimension('x', domain=(X0, XF), n_colloc=N_COLLOC)
x_bc = torch.tensor([[X0], [XF]], dtype=torch.float32, device=DEVICE)
u_bc = torch.zeros(2, 1, device=DEVICE)


def residual(u, x, k):
    u_x  = torch.autograd.grad(u.sum(),   x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    f = (k * np.pi) ** 2 * torch.sin(k * np.pi * x)
    return -u_xx - f


def train_one(form, k, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = PINArchitecture('MLP', {'layers': LAYERS, 'activation': 'tanh'}).to(DEVICE)
    # domain_volume=(XF-X0): weak_projection's raw Monte Carlo mean estimates
    # E[R*v_k] = (1/|Omega|)*integral, not the integral itself -- same fix as
    # sc2.1/sc2.2's PINLossFunction usage this session.
    loss_fn = PINLossFunction([X_DIM], residual, form=form, n_test=N_TEST,
                               domain_volume=(XF - X0))
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        opt.zero_grad()
        x_col = X_DIM.sample(N_COLLOC, device=DEVICE).requires_grad_(True)
        u_col = net(x_col)
        pde = loss_fn(u=u_col, x=x_col, k=k)
        bc = torch.nn.functional.mse_loss(net(x_bc), u_bc)
        loss = pde + 10.0 * bc
        loss.backward()
        opt.step()

    xg = torch.linspace(X0, XF, N_EVAL).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        up = net(xg).cpu().numpy().ravel()
    ue = np.sin(k * np.pi * xg.cpu().numpy().ravel())
    rel_l2 = np.sqrt(np.mean((up - ue) ** 2)) / (np.sqrt(np.mean(ue ** 2)) + 1e-12)
    return rel_l2, up, ue, xg.cpu().numpy().ravel()


def main():
    rows = []
    curves = {}
    t0 = time.perf_counter()
    for k in K_LIST:
        for form in ['strong', 'weak_projection']:
            err, up, ue, xg = train_one(form, k)
            rows.append({'k': k, 'form': form, 'rel_L2': err})
            curves[(k, form)] = (xg, up, ue)
            print(f'k={k:2d}  {form:16s}  rel_L2={err:.4e}')
    print(f'total wall time: {time.perf_counter() - t0:.1f}s')

    df = pd.DataFrame(rows)
    df.to_csv('/home/df.gomezb/Tesis/notebooks/ch2/poisson_highfreq_results.csv', index=False)
    print(df.pivot(index='k', columns='form', values='rel_L2').to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    piv = df.pivot(index='k', columns='form', values='rel_L2')
    ax.semilogy(piv.index, piv['strong'], 'o-', color='tab:red', label='strong')
    ax.semilogy(piv.index, piv['weak_projection'], 's-', color='tab:blue', label='weak_projection')
    ax.set_xlabel('k (frequency of exact solution)')
    ax.set_ylabel('relative L2 error')
    ax.set_title(r"1D Poisson, $u=\sin(k\pi x)$: strong vs weak")
    ax.grid(True, which='both', alpha=0.3); ax.legend()

    ax = axes[1]
    k_show = K_LIST[-1]
    xg, up_s, ue = curves[(k_show, 'strong')]
    _,  up_w, _  = curves[(k_show, 'weak_projection')]
    ax.plot(xg, ue, 'k-', lw=2, label='exact')
    ax.plot(xg, up_s, '--', color='tab:red', lw=1.5, label='strong')
    ax.plot(xg, up_w, '--', color='tab:blue', lw=1.5, label='weak_projection')
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.set_title(f'Predicted vs exact, k={k_show}')
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle('1D Poisson high-frequency benchmark (VPINN, arXiv:1912.00873)', fontsize=13)
    plt.tight_layout()
    fig.savefig('/home/df.gomezb/Tesis/notebooks/ch2/poisson_highfreq.png', dpi=140, bbox_inches='tight')
    fig.savefig('/home/df.gomezb/Tesis/notebooks/ch2/poisson_highfreq.pdf', bbox_inches='tight')
    print('saved -> poisson_highfreq.png')


if __name__ == '__main__':
    main()
