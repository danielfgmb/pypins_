# PIN Framework

A modular framework for **Physics-Informed Networks (PINs)** built on PyTorch.  
Supports TD-PINs (Time-Decomposed), Vanilla PINs, MLP and KAN architectures,
strong and weak (Petrov–Galerkin) loss forms, adaptive collocation strategies,
and diagnostic metrics.

---

## Structure

```
Tesis/
├── pyproject.toml
├── requirements.txt
├── CHANGES.md                    # session handoff / change log
├── pinns_framework/              # installable package
│   ├── __init__.py
│   ├── architecture.py           # MLP, KAN network builders
│   ├── dimensions.py             # spatial domain descriptors
│   ├── loss.py                   # strong + weak PDE loss (Petrov–Galerkin)
│   ├── trainer.py                # TD-PINs and Vanilla training loops + IO
│   ├── collocation.py            # 6 adaptive collocation strategies
│   └── metrics.py                # eval grid, diagnostic DataFrame, compute_error
├── docs/
│   └── weak_form.md              # mathematical reference — weak / variational form
└── notebooks/
    ├── demo_burgers.ipynb                  # 1-D viscous Burgers (TD-PINs)
    ├── demo_burgers2d.ipynb                # 2-D viscous Burgers (TD-PINs)
    ├── kan_burgers_sawtooth.ipynb          # KAN sawtooth benchmark (strong form)
    └── kan_burgers_sawtooth_weak.ipynb     # KAN sawtooth — strong vs weak comparison
```

---

## Installation

```bash
# Create and activate a virtual environment (Python ≥ 3.9)
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Optional: KAN backend
pip install pykan
```

Or install dependencies directly from the pinned requirements file:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1-D Burgers — TD-PINs

```python
import torch
import numpy as np
from pinns_framework import (
    PINArchitecture, PINSpatialDimension,
    PINLossFunction, PINTrainer,
)

# Network: 1 spatial input → 1 output
net = PINArchitecture("MLP", {"layers": (1, 64, 64, 64, 1), "activation": "tanh"})

# Spatial domain
x_dim = PINSpatialDimension("x", domain=(-1.0, 1.0), n_colloc=200)

# PDE residual — Backward Euler discretisation
def burgers_residual(u, u_prev, x, dt, nu):
    du_dx  = torch.autograd.grad(u.sum(),     x, create_graph=True)[0]
    du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    return (u - u_prev) / dt + u * du_dx - nu * du_dxx

loss_fn = PINLossFunction([x_dim], burgers_residual)

trainer = PINTrainer(
    net=net, scheme="TD-PINNS", loss_fn=loss_fn,
    spatial_dims=[x_dim], epochs=3000, dt=0.1, lr=1e-2,
)

# IC: u(0,x) = -sin(πx)   BC: u(t,±1) = 0
x_ic = torch.linspace(-1, 1, 200).unsqueeze(1)
u_ic = -torch.sin(np.pi * x_ic)
x_bc = torch.tensor([[-1.0], [1.0]])
u_bc = torch.zeros(2, 1)

trained_nets = trainer.fit(x_ic, u_ic, x_bc, u_bc, T=1.0, nu=0.01/np.pi)
# → list[nn.Module], one per time step; index 0 = IC network
```

### Switching to KAN

```python
net = PINArchitecture("KAN", {"layers": (1, 16, 16, 1), "grid_size": 5, "spline_order": 3})
# Everything else is identical — trainer, loss, IC, BC unchanged.
```

### Switching to Weak Form

The `residual_fn` signature is **unchanged**. Only `PINLossFunction` changes:

```python
# Strong form (default) — pointwise MSE of residual
loss_fn = PINLossFunction([x_dim], burgers_residual)

# Weak form — Petrov–Galerkin projection onto 10 Legendre test functions
loss_fn = PINLossFunction([x_dim], burgers_residual, form="weak", n_test=10)

# Everything downstream (trainer, fit, metrics) is identical
```

See `docs/weak_form.md` for the full mathematical derivation.

### Vanilla PINs (single space-time network)

```python
net_v = PINArchitecture("MLP", {"layers": (2, 64, 64, 64, 1)})  # 2 = t + x

def burgers_residual_vanilla(u, t, x, nu):
    du_dt  = torch.autograd.grad(u.sum(),     t, create_graph=True)[0]
    du_dx  = torch.autograd.grad(u.sum(),     x, create_graph=True)[0]
    du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    return du_dt + u * du_dx - nu * du_dxx

trainer_v = PINTrainer(
    net=net_v, scheme="VANILLA", loss_fn=PINLossFunction([x_dim], burgers_residual_vanilla),
    spatial_dims=[x_dim], epochs=10000, lr=1e-3,
)
trained_nets_v = trainer_v.fit(x_ic, u_ic, x_bc, u_bc, T=1.0, nu=0.01/np.pi)
# → one-element list; trained_nets_v[0] is the full (t, x) network
```

### 2-D Problems

```python
x_dim = PINSpatialDimension("x", (-1.0, 1.0), n_colloc=2500)
y_dim = PINSpatialDimension("y", (-1.0, 1.0), n_colloc=2500)

net = PINArchitecture("MLP", {"layers": (2, 64, 64, 64, 1)})  # 2 inputs: x, y

def burgers2d_residual(u, u_prev, x, dt, nu):
    # x is (N, 2): column 0 = x-coord, column 1 = y-coord
    grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0][:, 0:1]
    du_dyy = torch.autograd.grad(du_dy.sum(), x, create_graph=True)[0][:, 1:2]
    return (u - u_prev) / dt + u * du_dx + u * du_dy - nu * (du_dxx + du_dyy)

trainer = PINTrainer(
    net=net, scheme="TD-PINNS",
    loss_fn=PINLossFunction([x_dim, y_dim], burgers2d_residual),
    spatial_dims=[x_dim, y_dim], epochs=5000, dt=0.05,
)
```

---

## Adaptive Collocation

Pass any of the six strategies to `PINTrainer` via `collocation_fn`.
All strategies share the same signature and are interchangeable.

```python
from pinns_framework import (
    collocation_uniform,       # random uniform (default behaviour)
    collocation_lhs,           # Latin Hypercube Sampling
    collocation_gradient,      # concentrate where ||∇u||₂ is large
    collocation_curvature,     # concentrate where Σ|∂²u/∂xd²| is large
    collocation_last_residual, # concentrate where PDE residual was high at end of prev step
    collocation_mean_residual, # concentrate where PDE residual was high on avg over prev step
)

trainer = PINTrainer(
    ...,
    collocation_fn=collocation_curvature,   # swap any strategy here
)
```

All adaptive strategies use `adaptive_frac=0.5` by default (50% adaptive, 50% uniform fill-in).
`last_residual` and `mean_residual` fall back to uniform at step 0 (no history yet).

---

## Saving and Loading

```python
from pinns_framework import save_pinns, load_pinns

# Save (works for TD-PINs and Vanilla, MLP and KAN)
save_pinns(
    trained_nets, trainer,
    arch="MLP",
    params_arch={"layers": (1, 64, 64, 64, 1), "activation": "tanh"},
    filepath="checkpoint.pt",
)

# Load — scheme and architecture are auto-detected from the checkpoint
loaded_nets = load_pinns("checkpoint.pt")
# → list[nn.Module] in both cases (Vanilla returns a one-element list)
```

---

## Evaluation and Metrics

```python
from pinns_framework import make_eval_grid, compute_metrics, compute_error

# Dense uniform evaluation grid — independent of collocation points
x_eval = make_eval_grid([x_dim], n_per_dim=500)  # (500, 1)

# Diagnostic DataFrame: mass, energy, and (if exact_fn given) L2, Linf, rel_L2
times = [i * trainer.dt for i in range(len(trained_nets))]

df = compute_metrics(
    trained_nets, times, [x_dim],
    n_eval=500,
    exact_fn=lambda t, x: -torch.sin(np.pi * x) * torch.exp(-t),  # example
)
print(df[["mass", "energy", "L2", "rel_L2"]])

# Point-wise error metrics between two numpy arrays (trapezoidal integration)
err = compute_error(u_pred_np, u_exact_np, x_np)
# → {"L2": float, "Linf": float, "rel_L2": float}
```

`exact_fn` signature: `(t: float, x_eval: Tensor (N, D)) -> Tensor (N, 1)`.  
Works for any spatial dimensionality.

---

## API Reference

### `PINArchitecture(arch, params_arch)`
Factory returning an `nn.Module`.

| `arch` | `params_arch` keys |
|---|---|
| `"MLP"` | `layers` (tuple), `activation` (`"tanh"`, `"relu"`, `"gelu"`, `"silu"`) |
| `"KAN"` | `layers` (tuple), `grid_size` (int), `spline_order` (int) |

### `PINSpatialDimension(name, domain, n_colloc)`
Describes one spatial axis. `n_colloc` is the number of collocation points
sampled per training step.

### `PINLossFunction(spatial_dims, residual_fn, form="strong", n_test=10, multi_dim_mode="total_degree", test_fn=None)`
Wraps `residual_fn` and computes the PDE loss. `residual_fn` receives keyword
arguments injected by the trainer:

| Scheme | Signature |
|---|---|
| TD-PINs | `residual_fn(u, u_prev, x, dt, **pde_kwargs)` |
| Vanilla | `residual_fn(u, t, x, **pde_kwargs)` |

`x` is always `(N, D)` with `requires_grad=True`.  
`**pde_kwargs` are constants forwarded from `trainer.fit(...)` (e.g. `nu=...`).

| `form` | Loss formula | Description |
|---|---|---|
| `"strong"` (default) | `(1/N) Σ R(xᵢ)²` | MSE of point-wise PDE residual |
| `"weak"` | `(1/K) Σₖ (∫ R vₖ dx)²` | Petrov–Galerkin projection onto Legendre test functions |

**Weak form parameters** (ignored when `form="strong"`):

| Parameter | Default | Description |
|---|---|---|
| `n_test` | `10` | Legendre basis functions per spatial dimension |
| `multi_dim_mode` | `"total_degree"` | `"total_degree"` — only multi-indices with Σkd ≤ n_test−1, K = C(n_test−1+D, D); `"full"` — all tensor products, K = n_test^D |
| `test_fn` | `None` | Custom callable `(x: Tensor(N,D), spatial_dims) -> Tensor(N,K)`; overrides the built-in Legendre bank |

Test function counts for `n_test=10`:

| D | `total_degree` K | `full` K |
|---|---|---|
| 1 | 10 | 10 |
| 2 | 55 | 100 |
| 3 | 220 | 1 000 |

The `residual_fn` signature is **identical** for both forms — only the loss aggregation changes.  
See `docs/weak_form.md` for the full mathematical derivation.

### `PINTrainer(...)`

| Parameter | Default | Description |
|---|---|---|
| `net` | — | `nn.Module` from `PINArchitecture` |
| `scheme` | — | `"TD-PINNS"` or `"VANILLA"` |
| `loss_fn` | — | `PINLossFunction` instance |
| `spatial_dims` | — | `list[PINSpatialDimension]` |
| `epochs` | `2000` | max optimisation steps per time level |
| `dt` | `None` | time step (TD-PINs only) |
| `time_integrator` | `"BE"` | `"BE"` (Backward Euler); CN/BDF2 planned |
| `lr` | `1e-2` | initial AdamW learning rate |
| `tolerance` | `1e-4` | early-stopping loss threshold |
| `ic_weight` | `10.0` | weight on IC term |
| `bc_weight` | `10.0` | weight on BC term |
| `pde_weight` | `0.1` | weight on PDE residual term |
| `collocation_fn` | `None` | adaptive collocation strategy |
| `verbose` | `True` | print training progress |
| `log_every` | `500` | print/callback frequency (epochs) |

`trainer.fit(x_ic, u_ic, x_bc, u_bc, T, **pde_kwargs)` always returns
`list[nn.Module]` — N+1 elements for TD-PINs, 1 element for Vanilla.

### `compute_error(u_pred, u_exact, x)`
Compute L2, Linf, and relative-L2 errors between two numpy arrays using the
trapezoidal rule. Compatible with NumPy ≥ 1.20 (handles the `trapz` → `trapezoid`
rename automatically).

```python
err = compute_error(u_pred_np, u_exact_np, x_np)
# → {"L2": float, "Linf": float, "rel_L2": float}
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `torch` | 2.10.0 | autodiff, network training |
| `numpy` | ≥ 1.20 (tested 2.4.2) | numerics |
| `matplotlib` | 3.10.8 | plotting |
| `pandas` | 3.0.1 | metrics DataFrame |
| `pykan` | 0.2.8 | KAN backend (optional) |
| `scipy` | 1.17.1 | available in environment |
| `scikit-learn` | 1.8.0 | available in environment |
