#!/usr/bin/env python
"""
sc2.1 -- TD-PINNs, form='strong' (plain pointwise MSE, the baseline collocation
PINN residual -- no test-function projection). Standalone script (run under tmux):
    python train_td_strong.py
Writes results/td_strong_{metrics,history,loss_history,summary}.* and
checkpoints/td_strong.pt, then touches results/td_strong.DONE.

Tolerance-based early stopping is disabled (see common.py's TD_TOL) -- every
step always runs the full TD_EPOCHS budget. track_best=True keeps, per step,
the network weights from whichever epoch had the lowest training loss (not
necessarily the last epoch). error_fn=real_error_td logs the true relative
L2 error (vs the analytical solution) every ERROR_EVERY epochs into
trainer.loss_history, independent of the training loss.
"""
import time, json
import numpy as np
import torch
import pandas as pd

from common import (
    DEVICE, PARAMS_TD, DT, T_END, TIMES, LR, MAX_LR, MIN_LR,
    IC_WEIGHT, BC_WEIGHT, PDE_WEIGHT, N_EVAL, TD_EPOCHS, TD_TOL, IC_EPOCHS, ERROR_EVERY,
    COLLOCATION_FN, X_IC, U_IC, X_BC, U_BC, X_DIM, NU,
    td_residual, real_error_td, exact_fn, add_L1, RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import (
    PINArchitecture, PINLossFunction, PINTrainer, compute_metrics, save_pinns,
)

NAME = 'td_strong'


def main():
    print(f'[{NAME}] device={DEVICE} ic_epochs={IC_EPOCHS} pde_epochs/step={TD_EPOCHS} tol={TD_TOL}')
    torch.manual_seed(0)
    np.random.seed(0)

    loss_fn = PINLossFunction([X_DIM], td_residual, form='strong')
    net = PINArchitecture('MLP', PARAMS_TD)
    trainer = PINTrainer(
        net=net, scheme='TD-PINNS', loss_fn=loss_fn, spatial_dims=[X_DIM],
        epochs=TD_EPOCHS, ic_epochs=IC_EPOCHS, dt=DT, lr=LR, max_lr=MAX_LR, min_lr=MIN_LR, tolerance=TD_TOL,
        ic_weight=IC_WEIGHT, bc_weight=BC_WEIGHT, pde_weight=PDE_WEIGHT,
        collocation_fn=COLLOCATION_FN, device=DEVICE, verbose=True, log_every=5000,
        track_best=True, error_fn=real_error_td, error_every=ERROR_EVERY,
    )

    t0 = time.perf_counter()
    nets = trainer.fit(x_ic=X_IC, u_ic=U_IC, x_bc=X_BC, u_bc=U_BC, T=T_END, nu=NU)
    wall = time.perf_counter() - t0
    epochs = sum(h['epochs'] for h in trainer.step_history[1:])
    print(f'[{NAME}] done | epochs(PDE steps)={epochs} | wall {wall:.1f}s')

    df = compute_metrics(nets, TIMES, [X_DIM], n_eval=N_EVAL, exact_fn=exact_fn, device=DEVICE)
    df = add_L1(df, nets)
    df.to_csv(f'{RESULTS_DIR}/{NAME}_metrics.csv')

    pd.DataFrame(trainer.step_history).to_csv(f'{RESULTS_DIR}/{NAME}_history.csv', index=False)
    pd.DataFrame(trainer.loss_history).to_csv(f'{RESULTS_DIR}/{NAME}_loss_history.csv', index=False)

    save_pinns(nets, trainer, 'MLP', PARAMS_TD, f'{CHECKPOINTS_DIR}/{NAME}.pt')

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
