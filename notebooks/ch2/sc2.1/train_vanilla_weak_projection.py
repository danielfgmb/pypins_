#!/usr/bin/env python
"""
sc2.1 -- Vanilla PINNs, form='weak_projection'. Standalone script (run under tmux):
    python train_vanilla_weak_projection.py
Writes results/vanilla_weak_projection_{metrics,history,summary}.* and
checkpoints/vanilla_weak_projection.pt, then touches
results/vanilla_weak_projection.DONE.

NOTE: COLLOCATION_FN = collocation_last_residual (see common.py), but
PINTrainer._fit_vanilla never populates the residual-history buffers that
strategy needs -- for this run it silently behaves as plain uniform sampling
throughout. This is a framework limitation, not a bug in this script.

NOTE: domain_volume=SPACETIME_VOLUME (=|Omega|*T), not DOMAIN_VOLUME, even
though weak_projection's test functions only depend on x here (PINLossFunction
only ever builds V from the 'x' kwarg) -- t is still independently random per
collocation point, so it gets silently marginalized out by the same Monte
Carlo mean, same as for vanilla_agnostic_loss. See NOTA_analisis_agnostic.md.
"""
import time, json
import numpy as np
import torch
import pandas as pd

from common import (
    DEVICE, PARAMS_VAN, T_END, TIMES, LR, MAX_LR, MIN_LR,
    IC_WEIGHT, BC_WEIGHT, PDE_WEIGHT, N_TEST, N_EVAL, VAN_EPOCHS, VAN_TOL, ERROR_EVERY,
    COLLOCATION_FN, X_IC, U_IC, X_BC, U_BC, X_DIM, NU, SPACETIME_VOLUME,
    vanilla_residual, real_error_vanilla, exact_fn, add_L1, TimeSlice, RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import (
    PINArchitecture, PINLossFunction, PINTrainer, compute_metrics, save_pinns,
)

NAME = 'vanilla_weak_projection'


def main():
    print(f'[{NAME}] device={DEVICE} epochs={VAN_EPOCHS} tol={VAN_TOL}')
    print(f'[{NAME}] WARNING: collocation_last_residual has no effect for Vanilla '
          f'(PINTrainer never populates residual history in _fit_vanilla) -- '
          f'this run uses plain uniform sampling in practice, see common.py.')
    torch.manual_seed(0)
    np.random.seed(0)

    loss_fn = PINLossFunction([X_DIM], vanilla_residual, form='weak_projection', n_test=N_TEST,
                               domain_volume=SPACETIME_VOLUME)
    net = PINArchitecture('MLP', PARAMS_VAN)
    trainer = PINTrainer(
        net=net, scheme='VANILLA', loss_fn=loss_fn, spatial_dims=[X_DIM],
        epochs=VAN_EPOCHS, lr=LR, max_lr=MAX_LR, min_lr=MIN_LR, tolerance=VAN_TOL,
        ic_weight=IC_WEIGHT, bc_weight=BC_WEIGHT, pde_weight=PDE_WEIGHT,
        collocation_fn=COLLOCATION_FN, device=DEVICE, verbose=True, log_every=2000,
        track_best=True, error_fn=real_error_vanilla, error_every=ERROR_EVERY,
    )

    t0 = time.perf_counter()
    net_trained = trainer.fit(x_ic=X_IC, u_ic=U_IC, x_bc=X_BC, u_bc=U_BC, T=T_END, nu=NU)[0]
    wall = time.perf_counter() - t0
    epochs = trainer.step_history[-1]['epochs']
    print(f'[{NAME}] done | epochs={epochs} | wall {wall:.1f}s')

    slices = [TimeSlice(net_trained, t).to(DEVICE) for t in TIMES]
    df = compute_metrics(slices, TIMES, [X_DIM], n_eval=N_EVAL, exact_fn=exact_fn, device=DEVICE)
    df = add_L1(df, slices)
    df.to_csv(f'{RESULTS_DIR}/{NAME}_metrics.csv')

    pd.DataFrame(trainer.step_history).to_csv(f'{RESULTS_DIR}/{NAME}_history.csv', index=False)
    pd.DataFrame(trainer.loss_history).to_csv(f'{RESULTS_DIR}/{NAME}_loss_history.csv', index=False)

    save_pinns([net_trained], trainer, 'MLP', PARAMS_VAN, f'{CHECKPOINTS_DIR}/{NAME}.pt')

    with open(f'{RESULTS_DIR}/{NAME}_summary.json', 'w') as f:
        json.dump({
            'method': NAME, 'epochs': epochs, 'wall_s': wall,
            'mean_L1': float(df['L1'].mean()), 'mean_L2': float(df['L2'].mean()),
            'mean_Linf': float(df['Linf'].mean()),
            'final_L1': float(df['L1'].iloc[-1]), 'final_L2': float(df['L2'].iloc[-1]),
            'final_Linf': float(df['Linf'].iloc[-1]),
        }, f, indent=2)

    open(f'{RESULTS_DIR}/{NAME}.DONE', 'w').close()
    print(f'[{NAME}] DONE marker written')


if __name__ == '__main__':
    main()
