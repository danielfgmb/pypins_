#!/usr/bin/env python
"""
sc2.2 -- generic single-run trainer: one (method, nu) combination.
    python train_one.py --method {td_strong,td_weak_projection,td_agnostic,
                                   vanilla_strong,vanilla_weak_projection,vanilla_agnostic,
                                   vanilla_agnostic_weak_time}
                         --nu <float>
                         [--ic-epochs N] [--pde-epochs N] [--van-epochs N]  (smoke-test overrides)

Writes results/{method}_nu{tag}_{metrics,history,loss_history,summary}.* and
checkpoints/{method}_nu{tag}.pt, then touches results/{method}_nu{tag}.DONE.
Ground truth is the nu-independent Hopf-Lax entropy solution (common.exact_solution);
nu only enters the residual/training, never the comparison target.
"""
import argparse, time, json
import numpy as np
import torch
import pandas as pd

from common import (
    DEVICE, PARAMS_TD, PARAMS_VAN, DT, T_END, TIMES,
    LR, MAX_LR, MIN_LR, IC_WEIGHT, BC_WEIGHT, PDE_WEIGHT,
    N_TEST, N_EVAL, IC_EPOCHS, TD_EPOCHS, TD_TOL, VAN_EPOCHS, VAN_TOL, ERROR_EVERY,
    COLLOCATION_FN, X_IC, U_IC, X_BC, U_BC, X_DIM, DOMAIN_VOLUME, SPACETIME_VOLUME,
    VANILLA_AGNOSTIC_TIME_NODES, VANILLA_AGNOSTIC_WEAK_TIME_NODES,
    td_residual, vanilla_residual, td_agnostic_loss, vanilla_agnostic_loss,
    vanilla_agnostic_weak_time_loss,
    real_error_td, real_error_vanilla, exact_fn, add_L1, TimeSlice,
    RESULTS_DIR, CHECKPOINTS_DIR,
)
from pinns_framework import (
    PINArchitecture, PINLossFunction, PINTrainer, compute_metrics, save_pinns,
)

METHODS = ('td_strong', 'td_weak_projection', 'td_agnostic',
           'vanilla_strong', 'vanilla_weak_projection', 'vanilla_agnostic',
           'vanilla_agnostic_weak_time')


def nu_tag(nu):
    return f'{nu:g}'.replace('.', 'p')


def build(method, nu):
    """Returns (scheme, loss_fn, params, epochs, ic_epochs_or_None, tol, error_fn, time_colloc_nodes)."""
    is_td = method.startswith('td_')
    scheme = 'TD-PINNS' if is_td else 'VANILLA'
    params = PARAMS_TD if is_td else PARAMS_VAN
    epochs = TD_EPOCHS if is_td else VAN_EPOCHS
    tol    = TD_TOL if is_td else VAN_TOL
    ic_ep  = IC_EPOCHS if is_td else None
    err_fn = real_error_td if is_td else real_error_vanilla
    time_colloc_nodes = None

    if method == 'td_strong':
        loss_fn = PINLossFunction([X_DIM], td_residual, form='strong')
    elif method == 'td_weak_projection':
        loss_fn = PINLossFunction([X_DIM], td_residual, form='weak_projection',
                                   n_test=N_TEST, domain_volume=DOMAIN_VOLUME)
    elif method == 'td_agnostic':
        loss_fn = PINLossFunction([X_DIM], td_agnostic_loss, form='agnostic')
    elif method == 'vanilla_strong':
        loss_fn = PINLossFunction([X_DIM], vanilla_residual, form='strong')
    elif method == 'vanilla_weak_projection':
        loss_fn = PINLossFunction([X_DIM], vanilla_residual, form='weak_projection',
                                   n_test=N_TEST, domain_volume=SPACETIME_VOLUME)
    elif method == 'vanilla_agnostic':
        loss_fn = PINLossFunction([X_DIM], vanilla_agnostic_loss, form='agnostic')
    elif method == 'vanilla_agnostic_weak_time':
        loss_fn = PINLossFunction([X_DIM], vanilla_agnostic_weak_time_loss, form='agnostic')
    else:
        raise ValueError(f'unknown method {method!r}')

    return scheme, loss_fn, params, epochs, ic_ep, tol, err_fn, time_colloc_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--nu', required=True, type=float)
    ap.add_argument('--ic-epochs', type=int, default=None)
    ap.add_argument('--pde-epochs', type=int, default=None)
    ap.add_argument('--van-epochs', type=int, default=None)
    ap.add_argument('--checkpoint-every', type=int, default=None)
    ap.add_argument('--log-every', type=int, default=5000)
    args = ap.parse_args()

    method, nu = args.method, args.nu
    NAME = f'{method}_nu{nu_tag(nu)}'

    scheme, loss_fn, params, epochs, ic_ep, tol, err_fn, time_colloc_nodes = build(method, nu)
    if args.pde_epochs is not None:
        epochs = args.pde_epochs
    if args.van_epochs is not None and scheme == 'VANILLA':
        epochs = args.van_epochs
    if args.ic_epochs is not None:
        ic_ep = args.ic_epochs

    print(f'[{NAME}] device={DEVICE} scheme={scheme} nu={nu} epochs={epochs} '
          f'ic_epochs={ic_ep} tol={tol}')
    torch.manual_seed(0)
    np.random.seed(0)

    net = PINArchitecture('MLP', params)
    kwargs = dict(
        net=net, scheme=scheme, loss_fn=loss_fn, spatial_dims=[X_DIM],
        epochs=epochs, dt=DT, lr=LR, max_lr=MAX_LR, min_lr=MIN_LR, tolerance=tol,
        ic_weight=IC_WEIGHT, bc_weight=BC_WEIGHT, pde_weight=PDE_WEIGHT,
        collocation_fn=COLLOCATION_FN, device=DEVICE, verbose=True, log_every=args.log_every,
        track_best=True, error_fn=err_fn, error_every=ERROR_EVERY,
    )
    if scheme == 'TD-PINNS':
        kwargs['ic_epochs'] = ic_ep
    if time_colloc_nodes is not None:
        kwargs['time_colloc_nodes'] = time_colloc_nodes
    if args.checkpoint_every is not None:
        kwargs['checkpoint_every'] = args.checkpoint_every
        kwargs['checkpoint_dir'] = f'{CHECKPOINTS_DIR}/{NAME}_ckpts'
        kwargs['checkpoint_prefix'] = NAME
    trainer = PINTrainer(**kwargs)

    t0 = time.perf_counter()
    result = trainer.fit(x_ic=X_IC, u_ic=U_IC, x_bc=X_BC, u_bc=U_BC, T=T_END, nu=nu)
    wall = time.perf_counter() - t0

    if scheme == 'TD-PINNS':
        nets = result
        epochs_done = sum(h['epochs'] for h in trainer.step_history[1:])
        slices = nets
    else:
        net_trained = result[0]
        epochs_done = trainer.step_history[-1]['epochs']
        slices = [TimeSlice(net_trained, t).to(DEVICE) for t in TIMES]

    print(f'[{NAME}] done | epochs={epochs_done} | wall {wall:.1f}s')

    df = compute_metrics(slices, TIMES, [X_DIM], n_eval=N_EVAL, exact_fn=exact_fn, device=DEVICE)
    df = add_L1(df, slices)
    df.to_csv(f'{RESULTS_DIR}/{NAME}_metrics.csv')

    pd.DataFrame(trainer.step_history).to_csv(f'{RESULTS_DIR}/{NAME}_history.csv', index=False)
    pd.DataFrame(trainer.loss_history).to_csv(f'{RESULTS_DIR}/{NAME}_loss_history.csv', index=False)

    save_pinns(slices if scheme == 'TD-PINNS' else [net_trained], trainer, 'MLP', params,
               f'{CHECKPOINTS_DIR}/{NAME}.pt')

    with open(f'{RESULTS_DIR}/{NAME}_summary.json', 'w') as f:
        json.dump({
            'method': method, 'nu': nu, 'epochs': epochs_done, 'wall_s': wall,
            'mean_L1': float(df['L1'].mean()), 'mean_L2': float(df['L2'].mean()),
            'mean_Linf': float(df['Linf'].mean()),
            'final_L1': float(df['L1'].iloc[-1]), 'final_L2': float(df['L2'].iloc[-1]),
            'final_Linf': float(df['Linf'].iloc[-1]),
        }, f, indent=2)

    open(f'{RESULTS_DIR}/{NAME}.DONE', 'w').close()
    print(f'[{NAME}] DONE marker written')


if __name__ == '__main__':
    main()
