#!/usr/bin/env python
"""
sc2.3 -- generic single-run trainer: one (form, k) combination.
    python train_one.py --form {strong,weak_projection,agnostic} --k <int>

Static problem (no time-stepping), so this bypasses PINTrainer entirely --
just PINArchitecture + PINLossFunction + a bare Adam loop. Writes
results/{form}_k{k}_summary.json, results/{form}_k{k}_loss_history.csv
(epoch, pde_loss, bc_loss, rel_L2 -- the real error vs the exact solution,
sampled every ERROR_EVERY epochs, see plot_loss_curves.py) and
checkpoints/{form}_k{k}.pt, then touches results/{form}_k{k}.DONE.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from common import (
    DEVICE, X0, XF, X_DIM, LAYERS, N_COLLOC, N_EVAL, N_TEST, EPOCHS, LR, BC_WEIGHT,
    ERROR_EVERY, residual, poisson_agnostic_loss, exact_solution, name_for, RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import PINArchitecture, PINLossFunction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--form', required=True, choices=['strong', 'weak_projection', 'agnostic'])
    ap.add_argument('--k', required=True, type=int)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--checkpoint-every', type=int, default=None)
    args = ap.parse_args()

    form, k = args.form, args.k
    epochs = args.epochs if args.epochs is not None else EPOCHS
    NAME = name_for(form, k)
    ckpt_dir = f'{CHECKPOINTS_DIR}/{NAME}_ckpts'
    if args.checkpoint_every:
        import os
        os.makedirs(ckpt_dir, exist_ok=True)
    print(f'[{NAME}] device={DEVICE} form={form} k={k} epochs={epochs} checkpoint_every={args.checkpoint_every}')

    torch.manual_seed(0); np.random.seed(0)
    net = PINArchitecture('MLP', {'layers': LAYERS, 'activation': 'tanh'}).to(DEVICE)
    if form == 'agnostic':
        loss_fn = PINLossFunction([X_DIM], poisson_agnostic_loss, form='agnostic')
    else:
        # domain_volume=(XF-X0): weak_projection's raw Monte Carlo mean
        # estimates E[R*v_k] = (1/|Omega|)*integral, not the integral itself
        # -- same domain-volume fix as sc2.1/sc2.2's PINLossFunction usage.
        loss_fn = PINLossFunction([X_DIM], residual, form=form, n_test=N_TEST,
                                   domain_volume=(XF - X0))
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    x_bc = torch.tensor([[X0], [XF]], dtype=torch.float32, device=DEVICE)
    u_bc = torch.zeros(2, 1, device=DEVICE)

    xg = torch.linspace(X0, XF, N_EVAL).unsqueeze(1).to(DEVICE)
    ue_np = exact_solution(k, xg.cpu().numpy().ravel())
    ue_norm = np.sqrt(np.mean(ue_np ** 2)) + 1e-12

    def real_error():
        was_training = net.training
        net.eval()
        with torch.no_grad():
            up = net(xg).cpu().numpy().ravel()
        if was_training:
            net.train()
        return float(np.sqrt(np.mean((up - ue_np) ** 2)) / ue_norm)

    loss_history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad()
        x_col = X_DIM.sample(N_COLLOC, device=DEVICE).requires_grad_(True)
        u_col = net(x_col)
        pde = loss_fn(u=u_col, x=x_col, k=k)
        bc = F.mse_loss(net(x_bc), u_bc)
        loss = pde + BC_WEIGHT * bc
        loss.backward()
        opt.step()
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            torch.save(net.state_dict(), f'{ckpt_dir}/{NAME}_epoch{epoch}.pt')
        if epoch % ERROR_EVERY == 0 or epoch == epochs - 1:
            err = real_error()
            loss_history.append({'epoch': epoch, 'pde_loss': pde.item(),
                                  'bc_loss': bc.item(), 'rel_L2': err})
            print(f'  epoch {epoch:6d} | pde {pde.item():.4e} | bc {bc.item():.4e} | real_rel_L2 {err:.4e}')
    wall = time.perf_counter() - t0

    with torch.no_grad():
        up = net(xg).cpu().numpy().ravel()
    diff = up - ue_np
    rel_l2 = float(np.sqrt(np.mean(diff ** 2)) / ue_norm)
    l1 = float(np.mean(np.abs(diff)))
    linf = float(np.max(np.abs(diff)))
    print(f'[{NAME}] done | wall {wall:.1f}s | rel_L2 {rel_l2:.4e}')

    torch.save(net.state_dict(), f'{CHECKPOINTS_DIR}/{NAME}.pt')
    pd.DataFrame(loss_history).to_csv(f'{RESULTS_DIR}/{NAME}_loss_history.csv', index=False)
    with open(f'{RESULTS_DIR}/{NAME}_summary.json', 'w') as f:
        json.dump({'form': form, 'k': k, 'epochs': epochs, 'wall_s': wall,
                    'rel_L2': rel_l2, 'L1': l1, 'Linf': linf}, f, indent=2)
    open(f'{RESULTS_DIR}/{NAME}.DONE', 'w').close()
    print(f'[{NAME}] DONE marker written')


if __name__ == '__main__':
    main()
