#!/usr/bin/env python
"""
sc2.4 -- generic single-run trainer, Vanilla PINNs only, one method:
    python train_one.py --method {strong,weak_projection,agnostic,agnostic_weak_time}

Writes results/{method}_{metrics,history,loss_history,summary}.* and
checkpoints/{method}.pt, then touches results/{method}.DONE.
"""
import argparse, time, json
import numpy as np
import torch
import pandas as pd

from common import (
    DEVICE, PARAMS_VAN, T_END, NU, LR, MAX_LR, MIN_LR,
    IC_WEIGHT, BC_WEIGHT, PDE_WEIGHT, N_TEST, N_EVAL, EPOCHS, ERROR_EVERY,
    X_IC, U_IC, X_BC, U_BC, X_DIM, DOMAIN_VOLUME, SPACETIME_VOLUME,
    vanilla_residual, vanilla_agnostic_loss, vanilla_agnostic_weak_time_loss,
    real_error_vanilla, exact_solution, exact_fn, RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import (
    PINArchitecture, PINSpatialDimension, PINLossFunction, PINTrainer,
    compute_metrics, save_pinns, collocation_uniform,
)

METHODS = ('strong', 'weak_projection', 'agnostic', 'agnostic_weak_time')
_UNREACHABLE_TOL = -1e18


def build(method):
    if method == 'strong':
        loss_fn = PINLossFunction([X_DIM], vanilla_residual, form='strong')
    elif method == 'weak_projection':
        loss_fn = PINLossFunction([X_DIM], vanilla_residual, form='weak_projection',
                                   n_test=N_TEST, domain_volume=SPACETIME_VOLUME)
    elif method == 'agnostic':
        loss_fn = PINLossFunction([X_DIM], vanilla_agnostic_loss, form='agnostic')
    elif method == 'agnostic_weak_time':
        loss_fn = PINLossFunction([X_DIM], vanilla_agnostic_weak_time_loss, form='agnostic')
    else:
        raise ValueError(f'unknown method {method!r}')
    return loss_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--checkpoint-every', type=int, default=None)
    args = ap.parse_args()

    method = args.method
    epochs = args.epochs if args.epochs is not None else EPOCHS
    NAME = method
    print(f'[{NAME}] device={DEVICE} epochs={epochs} nu={NU} checkpoint_every={args.checkpoint_every}')
    torch.manual_seed(0); np.random.seed(0)

    loss_fn = build(method)
    net = PINArchitecture('MLP', PARAMS_VAN)
    kwargs = dict(
        net=net, scheme='VANILLA', loss_fn=loss_fn, spatial_dims=[X_DIM],
        epochs=epochs, lr=LR, max_lr=MAX_LR, min_lr=MIN_LR, tolerance=_UNREACHABLE_TOL,
        ic_weight=IC_WEIGHT, bc_weight=BC_WEIGHT, pde_weight=PDE_WEIGHT,
        collocation_fn=collocation_uniform, device=DEVICE, verbose=True, log_every=1000,
        track_best=True, error_fn=real_error_vanilla, error_every=ERROR_EVERY,
    )
    if args.checkpoint_every is not None:
        kwargs['checkpoint_every'] = args.checkpoint_every
        kwargs['checkpoint_dir'] = f'{CHECKPOINTS_DIR}/{NAME}_ckpts'
        kwargs['checkpoint_prefix'] = NAME
    trainer = PINTrainer(**kwargs)

    t0 = time.perf_counter()
    net_trained = trainer.fit(x_ic=X_IC, u_ic=U_IC, x_bc=X_BC, u_bc=U_BC, T=T_END, nu=NU)[0]
    wall = time.perf_counter() - t0
    epochs_done = trainer.step_history[-1]['epochs']
    print(f'[{NAME}] done | epochs={epochs_done} | wall {wall:.1f}s')

    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    xg = torch.linspace(-5, 5, N_EVAL).unsqueeze(1).to(DEVICE)
    rows = []
    net_trained.eval()
    with torch.no_grad():
        for t in times:
            tcol = torch.full((xg.shape[0], 1), float(t), device=DEVICE)
            up = net_trained(torch.cat([tcol, xg], dim=1)).cpu().numpy().ravel()
            ue = exact_solution(t, xg.cpu().numpy().ravel())
            diff = up - ue
            l2 = float(np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(ue**2)) + 1e-12))
            l1 = float(np.mean(np.abs(diff)))
            linf = float(np.max(np.abs(diff)))
            rows.append({'t': t, 'L1': l1, 'L2': l2, 'Linf': linf})
    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/{NAME}_metrics.csv', index=False)

    pd.DataFrame(trainer.step_history).to_csv(f'{RESULTS_DIR}/{NAME}_history.csv', index=False)
    pd.DataFrame(trainer.loss_history).to_csv(f'{RESULTS_DIR}/{NAME}_loss_history.csv', index=False)
    save_pinns([net_trained], trainer, 'MLP', PARAMS_VAN, f'{CHECKPOINTS_DIR}/{NAME}.pt')

    with open(f'{RESULTS_DIR}/{NAME}_summary.json', 'w') as f:
        json.dump({'method': method, 'epochs': epochs_done, 'wall_s': wall,
                    'mean_L2': float(df['L2'].mean()), 'final_L2': float(df['L2'].iloc[-1]),
                    'mean_Linf': float(df['Linf'].mean()), 'final_Linf': float(df['Linf'].iloc[-1])},
                  f, indent=2)
    open(f'{RESULTS_DIR}/{NAME}.DONE', 'w').close()
    print(f'[{NAME}] DONE marker written')


if __name__ == '__main__':
    main()
