"""
Dense evaluation grid and per-step diagnostic metrics for trained PINNs.

The evaluation grid is always a regular Cartesian meshgrid — independent of
whichever collocation strategy was used during training.  This makes it
suitable for pointwise error comparison against analytic solutions and for
numerical integration (mass, energy).
"""

import torch
import numpy as np

try:
    _np_trapezoid = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _np_trapezoid = np.trapz       # NumPy < 2.0


def make_eval_grid(spatial_dims, n_per_dim=200):
    """
    Build a regular Cartesian evaluation grid over the spatial domain.

    Parameters
    ----------
    spatial_dims : list[PINSpatialDimension]
    n_per_dim    : grid resolution along each axis

    Returns
    -------
    Tensor (n_per_dim^D, D) on CPU — same layout PINArchitecture expects.
    Indexing follows 'ij' convention: the first dimension of the grid
    corresponds to spatial_dims[0], the second to spatial_dims[1], etc.

    Examples
    --------
    1D : make_eval_grid([x_dim], 200)          → (200, 1)
    2D : make_eval_grid([x_dim, y_dim], 100)   → (10 000, 2)
    """
    axes  = [torch.linspace(d.domain[0], d.domain[1], n_per_dim) for d in spatial_dims]
    grids = torch.meshgrid(*axes, indexing='ij')
    return torch.stack([g.flatten() for g in grids], dim=1)  # (N, D)


def _trapz_nd(u_vals, spatial_dims, n_per_dim):
    """
    Approximate ∫_Ω u dΩ with the trapezoidal rule on a regular n_per_dim^D grid.

    u_vals   : (N, 1) or (N,) — values at the make_eval_grid points (CPU tensor)
    Returns  : float
    """
    D   = len(spatial_dims)
    vol = u_vals.squeeze(-1).reshape(*([n_per_dim] * D))
    # Integrate from the last axis down to the first (each trapz reduces one dim)
    for d in range(D - 1, -1, -1):
        lo, hi = spatial_dims[d].domain
        dx  = (hi - lo) / (n_per_dim - 1)
        vol = torch.trapezoid(vol, dx=torch.tensor(dx, dtype=vol.dtype), dim=d)
    return float(vol)


def compute_metrics(trained_nets, times, spatial_dims, n_eval=200,
                    exact_fn=None, device=None):
    """
    Evaluate trained networks on a dense uniform grid and compute diagnostics.

    Networks are evaluated at the same make_eval_grid points regardless of
    whether they were trained with TD-PINNs or Vanilla (both accept (N, D)
    spatial inputs and return (N, 1) outputs).

    Parameters
    ----------
    trained_nets : list[nn.Module]
        Output of trainer.fit() — always a list, one element per time level.
    times        : list[float]
        Time value associated with each network (index aligned with trained_nets).
    spatial_dims : list[PINSpatialDimension]
    n_eval       : grid resolution per spatial dimension.
    exact_fn     : optional callable(t: float, x_eval: Tensor (N, D)) -> Tensor (N, 1)
        Analytic solution at time t evaluated on the grid.  If provided,
        error columns are added to the output.
    device       : torch.device — auto-detected from the first network if None.

    Returns
    -------
    pd.DataFrame indexed by time with columns:
        mass, energy                          — always present
        exact_mass, exact_energy              — present when exact_fn is given
        L2, Linf, rel_L2                      — present when exact_fn is given
    """
    import pandas as pd

    if device is None:
        device = next(trained_nets[0].parameters()).device

    x_eval = make_eval_grid(spatial_dims, n_eval).to(device)

    rows = []
    for t, net in zip(times, trained_nets):
        net.eval()
        with torch.no_grad():
            u_pred = net(x_eval)
            if u_pred.dim() == 1:
                u_pred = u_pred.unsqueeze(1)

        row = {'t': t}
        u_cpu = u_pred.cpu()
        row['mass']   = _trapz_nd(u_cpu,        spatial_dims, n_eval)
        row['energy'] = _trapz_nd(u_cpu ** 2,   spatial_dims, n_eval)

        if exact_fn is not None:
            u_ex = exact_fn(t, x_eval)
            if isinstance(u_ex, np.ndarray):
                u_ex = torch.from_numpy(u_ex.astype(np.float32)).to(device)
            if u_ex.dim() == 1:
                u_ex = u_ex.unsqueeze(1)

            diff  = u_pred - u_ex
            l2    = float(diff.pow(2).mean().sqrt())
            linf  = float(diff.abs().max())
            denom = float(u_ex.pow(2).mean().sqrt()) + 1e-8

            row['exact_mass']   = _trapz_nd(u_ex.cpu(),        spatial_dims, n_eval)
            row['exact_energy'] = _trapz_nd((u_ex ** 2).cpu(), spatial_dims, n_eval)
            row['L2']     = l2
            row['Linf']   = linf
            row['rel_L2'] = l2 / denom

        rows.append(row)

    return pd.DataFrame(rows).set_index('t')


def compute_error(u_pred, u_exact, x):
    """
    Compute L2, Linf, and relative-L2 errors between two 1-D numpy arrays
    on a uniform grid x, using the trapezoidal rule for integration.

    Compatible with NumPy ≥ 1.20 (uses trapezoid when available, falls back
    to trapz for older installations).

    Parameters
    ----------
    u_pred  : array_like (N,) — predicted values
    u_exact : array_like (N,) — reference / exact values
    x       : array_like (N,) — spatial coordinates (must be sorted)

    Returns
    -------
    dict with keys 'L2', 'Linf', 'rel_L2'
    """
    u_pred  = np.asarray(u_pred,  dtype=float)
    u_exact = np.asarray(u_exact, dtype=float)
    x       = np.asarray(x,       dtype=float)

    diff = np.abs(u_pred - u_exact)
    L2   = float(np.sqrt(_np_trapezoid(diff ** 2, x)))
    Linf = float(diff.max())
    denom = float(np.sqrt(_np_trapezoid(u_exact ** 2, x))) + 1e-12
    return {"L2": L2, "Linf": Linf, "rel_L2": L2 / denom}
