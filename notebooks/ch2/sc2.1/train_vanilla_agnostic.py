#!/usr/bin/env python
"""
sc2.1 -- Vanilla PINNs, form='agnostic' (space-time weak form, Monte Carlo +
two-sample trick -- see common.py's vanilla_agnostic_loss and
NOTA_perdida_negativa.md). Standalone script (run under tmux):
    python train_vanilla_agnostic.py
Writes results/vanilla_agnostic_{metrics,history,loss_history,summary}.* and
checkpoints/vanilla_agnostic.pt, then touches results/vanilla_agnostic.DONE.

NOTE: uses collocation_uniform directly (not collocation_last_residual). For
Vanilla it would have been a silent no-op anyway (_fit_vanilla never populates
the residual-history buffers that strategy needs).

NOTE: no time_colloc_nodes -- the Monte Carlo two-sample-trick loss works on
ordinary random (t,x) collocation, same sampling as strong/weak_projection.
(A quadrature-based, tensor-product-batched version was tried and made to
work correctly, but the per-time-slice batching made it needlessly complex
and initially very slow before being vectorized -- reverted to this simpler,
faster, well-understood MC formulation instead. The loss CAN go negative
near convergence; that's expected, see the docstring in common.py.)

track_best=True keeps the network weights from whichever epoch had the lowest
training loss (not necessarily the last epoch). error_fn=real_error_vanilla
logs the true relative L2 error (averaged over all TIMES, vs the analytical
solution) every ERROR_EVERY epochs into trainer.loss_history, independent of
the training loss -- that's the number that actually matters, not the loss.
"""
import argparse, time, json
import numpy as np
import torch
import pandas as pd

from common import (
    DEVICE, PARAMS_VAN, T_END, TIMES, LR, MAX_LR, MIN_LR,
    IC_WEIGHT, BC_WEIGHT, PDE_WEIGHT, N_EVAL, VAN_EPOCHS, VAN_TOL, ERROR_EVERY,
    COLLOCATION_FN_AGNOSTIC_SAFE, X_IC, U_IC, X_BC, U_BC, X_DIM, NU,
    vanilla_agnostic_loss, real_error_vanilla, exact_fn, add_L1, TimeSlice, RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import (
    PINArchitecture, PINLossFunction, PINTrainer, compute_metrics, save_pinns,
)

NAME = 'vanilla_agnostic'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--checkpoint-every', type=int, default=None)
    args = ap.parse_args()
    epochs = args.epochs if args.epochs is not None else VAN_EPOCHS

    print(f'[{NAME}] device={DEVICE} epochs={epochs} tol={VAN_TOL} '
          f'checkpoint_every={args.checkpoint_every}')
    print(f'[{NAME}] using collocation_uniform (see module docstring).')
    torch.manual_seed(0)
    np.random.seed(0)

    loss_fn = PINLossFunction([X_DIM], vanilla_agnostic_loss, form='agnostic')
    net = PINArchitecture('MLP', PARAMS_VAN)
    trainer = PINTrainer(
        net=net, scheme='VANILLA', loss_fn=loss_fn, spatial_dims=[X_DIM],
        epochs=epochs, lr=LR, max_lr=MAX_LR, min_lr=MIN_LR, tolerance=VAN_TOL,
        ic_weight=IC_WEIGHT, bc_weight=BC_WEIGHT, pde_weight=PDE_WEIGHT,
        collocation_fn=COLLOCATION_FN_AGNOSTIC_SAFE, device=DEVICE, verbose=True, log_every=1000,
        track_best=True, error_fn=real_error_vanilla, error_every=ERROR_EVERY,
        checkpoint_every=args.checkpoint_every, checkpoint_dir=f'{CHECKPOINTS_DIR}/{NAME}_ckpts',
        checkpoint_prefix=NAME,
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
