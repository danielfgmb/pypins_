#!/usr/bin/env python
"""
sc2.0 -- Vanilla PINNs, MLP backbone, form='strong'. Standalone script (run under tmux):
    python train_vanilla_mlp.py
Writes results/vanilla_mlp_{metrics,history,loss_history,summary}.* and
checkpoints/vanilla_mlp.pt, then touches results/vanilla_mlp.DONE.

Tolerance-based early stopping is disabled -- always trains the full
MLP_VAN_EPOCHS budget (matched to TD-PINNs' MLP_IC_EPOCHS + 5*MLP_TD_EPOCHS).
track_best=True keeps the weights from whichever epoch had the lowest
training loss. error_fn=real_error_vanilla logs the true relative L2 error
(averaged over all TIMES, vs analytical) every ERROR_EVERY_MLP epochs.
"""
import time, json
import numpy as np
import torch
import pandas as pd

from common import (
    DEVICE, PARAMS_VAN_MLP, T_END, TIMES, LR, MAX_LR, MIN_LR,
    IC_WEIGHT, BC_WEIGHT, PDE_WEIGHT, N_EVAL, MLP_VAN_EPOCHS, MLP_TOL, ERROR_EVERY_MLP,
    COLLOCATION_FN, X_IC, U_IC, X_BC, U_BC, X_DIM, NU,
    vanilla_residual, real_error_vanilla, exact_fn, add_L1, TimeSlice, RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import (
    PINArchitecture, PINLossFunction, PINTrainer, compute_metrics, save_pinns,
)

NAME = 'vanilla_mlp'


def main():
    print(f'[{NAME}] device={DEVICE} epochs={MLP_VAN_EPOCHS} tol={MLP_TOL}')
    torch.manual_seed(0)
    np.random.seed(0)

    loss_fn = PINLossFunction([X_DIM], vanilla_residual, form='strong')
    net = PINArchitecture('MLP', PARAMS_VAN_MLP)
    trainer = PINTrainer(
        net=net, scheme='VANILLA', loss_fn=loss_fn, spatial_dims=[X_DIM],
        epochs=MLP_VAN_EPOCHS, lr=LR, max_lr=MAX_LR, min_lr=MIN_LR, tolerance=MLP_TOL,
        ic_weight=IC_WEIGHT, bc_weight=BC_WEIGHT, pde_weight=PDE_WEIGHT,
        collocation_fn=COLLOCATION_FN, device=DEVICE, verbose=True, log_every=2000,
        track_best=True, error_fn=real_error_vanilla, error_every=ERROR_EVERY_MLP,
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

    save_pinns([net_trained], trainer, 'MLP', PARAMS_VAN_MLP, f'{CHECKPOINTS_DIR}/{NAME}.pt')

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
