#!/usr/bin/env python
"""
sc2.1 -- Vanilla PINNs, form='agnostic' (full nu>0 reduced-order weak form:
u_t strong via autograd, spatial part weakened via IBP + real quadrature per
time slice -- same spatial machinery as td_agnostic_loss, see NOTA_agnostic_v2.md).
Standalone script (run under tmux):
    python train_vanilla_agnostic.py
Writes results/vanilla_agnostic_{metrics,history,loss_history,summary}.* and
checkpoints/vanilla_agnostic.pt, then touches results/vanilla_agnostic.DONE.

NOTE: uses collocation_uniform directly (not collocation_last_residual). For
Vanilla it would have been a silent no-op anyway (_fit_vanilla never populates
the residual-history buffers that strategy needs), and for form='agnostic' it
would additionally be unsafe in general (see common.py's
COLLOCATION_FN_AGNOSTIC_SAFE comment -- the crash mode is TD-PINNs-specific,
but this keeps both agnostic scripts on the same, honestly-labelled strategy).

NOTE: time_colloc_nodes=VANILLA_AGNOSTIC_TIME_NODES makes PDE collocation a
genuine tensor-product (t, x) batch -- VANILLA_AGNOSTIC_TIME_NODES = NT+1
fixed time nodes (landing exactly on TD-PINNs' own step times), each with its
own n_colloc-point spatial sample -- required so vanilla_agnostic_loss can
group points by torch.unique(t) and do a real per-slice trapezoid quadrature,
instead of the old space-time Monte Carlo estimate. See
pinns_framework/trainer.py's time_colloc_nodes docstring.

NOTE: vanilla_agnostic_loss uses plain squaring (R**2), not the two-sample
(U-statistic) trick -- with a real quadrature estimate (not a noisy MC sample
mean) there's no E[R_hat^2]=R^2+Var(R_hat) bias to correct for in the first
place, so this is just the ordinary least-squares residual, same as td_agnostic_loss.
Tolerance-based early stopping is disabled anyway (see common.py's VAN_TOL),
so this run always uses the full VAN_EPOCHS budget regardless.

track_best=True keeps the network weights from whichever epoch had the lowest
training loss (not necessarily the last epoch). error_fn=real_error_vanilla
logs the true relative L2 error (averaged over all TIMES, vs the analytical
solution) every ERROR_EVERY epochs into trainer.loss_history, independent of
the training loss.
"""
import time, json
import numpy as np
import torch
import pandas as pd

from common import (
    DEVICE, PARAMS_VAN, T_END, TIMES, LR, MAX_LR, MIN_LR,
    IC_WEIGHT, BC_WEIGHT, PDE_WEIGHT, N_EVAL, VAN_EPOCHS, VAN_TOL, ERROR_EVERY,
    COLLOCATION_FN_AGNOSTIC_SAFE, X_IC, U_IC, X_BC, U_BC, X_DIM, NU, VANILLA_AGNOSTIC_TIME_NODES,
    vanilla_agnostic_loss, real_error_vanilla, exact_fn, add_L1, TimeSlice, RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import (
    PINArchitecture, PINLossFunction, PINTrainer, compute_metrics, save_pinns,
)

NAME = 'vanilla_agnostic'


def main():
    print(f'[{NAME}] device={DEVICE} epochs={VAN_EPOCHS} tol={VAN_TOL} '
          f'time_colloc_nodes={VANILLA_AGNOSTIC_TIME_NODES}')
    print(f'[{NAME}] using collocation_uniform (see module docstring).')
    torch.manual_seed(0)
    np.random.seed(0)

    loss_fn = PINLossFunction([X_DIM], vanilla_agnostic_loss, form='agnostic')
    net = PINArchitecture('MLP', PARAMS_VAN)
    trainer = PINTrainer(
        net=net, scheme='VANILLA', loss_fn=loss_fn, spatial_dims=[X_DIM],
        epochs=VAN_EPOCHS, lr=LR, max_lr=MAX_LR, min_lr=MIN_LR, tolerance=VAN_TOL,
        ic_weight=IC_WEIGHT, bc_weight=BC_WEIGHT, pde_weight=PDE_WEIGHT,
        collocation_fn=COLLOCATION_FN_AGNOSTIC_SAFE, device=DEVICE, verbose=True, log_every=2000,
        track_best=True, error_fn=real_error_vanilla, error_every=ERROR_EVERY,
        time_colloc_nodes=VANILLA_AGNOSTIC_TIME_NODES,
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
