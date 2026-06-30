# Session Handoff — PIN Framework Changes

## What was done

### 0. TD-PINNs: convergence log + integrator-agnostic `u_prev` — `pinns_framework/trainer.py`

Two small, backward-compatible additions:

1. **`PINTrainer.step_history`** — a list populated by `_run_epochs`, one entry per IC
   fit / time step / vanilla run: `{"label", "epochs", "converged", "final_loss"}`.
   Purely diagnostic (reading it never affects training); reset at the start of each
   `fit()`. Lets callers see how many epochs each step took to reach `tolerance`.

2. **`u_prev` now keeps its graph w.r.t. `x`** inside `_fit_td` (was detached). The
   trainer stays integrator-agnostic — the time integrator is chosen entirely by the
   residual function (the "lf"): Backward Euler uses `u_prev.detach()`, while Explicit
   Euler / Crank–Nicolson differentiate `u_prev` to evaluate the frozen step's spatial
   operator. Backward Euler results are unchanged (its residual detaches `u_prev`, so no
   gradient reaches the frozen network).

3. **`max_lr` / `min_lr` constructor params.** The AdamW + `ReduceLROnPlateau` schedule is
   now tunable: `max_lr` is the peak/initial lr (defaults to `lr` for backward
   compatibility) and `min_lr` is the decay floor (was hard-coded `1e-8`).

### 1. Weak / Petrov–Galerkin form — `pinns_framework/loss.py`

The `form="weak"` placeholder (previously `NotImplementedError`) is now fully implemented.

**How it works:**  
Instead of minimising the pointwise MSE of the PDE residual, the weak form projects it
onto a bank of Legendre polynomial test functions and minimises the squared projections:

```
L_weak = (1/K) Σ_k  ((1/N) Σ_i R(x_i) v_k(x_i))²
```

where `v_k` are column-normalised Legendre polynomials P_0 … P_{K-1} mapped to the
physical domain via ξ = 2(x−a)/(b−a) − 1.

**New helpers (module-private):**
- `_legendre_1d(xi, n_max)` — three-term Legendre recurrence on (N,) → (N, n_max)
- `_build_test_matrix(x, spatial_dims, n_test, multi_dim_mode)` — builds the (N, K)
  column-normalised test matrix; returned tensor is fully detached (no grad flow)

**New `PINLossFunction` parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `form` | `"strong"` | `"strong"` or `"weak"` |
| `n_test` | `10` | Legendre basis functions per spatial dim |
| `multi_dim_mode` | `"total_degree"` | `"total_degree"` or `"full"` — how to combine dims |
| `test_fn` | `None` | Custom `(x, spatial_dims) → (N,K)` callable |

**Multi-dim test function counts (`n_test=10`):**

| D | `total_degree` K | `full` K |
|---|---|---|
| 1 | 10 | 10 |
| 2 | 55 | 100 |
| 3 | 220 | 1000 |

`total_degree` is the default for D ≥ 2 — polynomial growth instead of exponential.

**The `residual_fn` signature is unchanged.** Only the aggregation from R to loss differs.

```python
# Strong (default)
loss = PINLossFunction([x_dim], residual_fn)

# Weak — drop-in replacement
loss = PINLossFunction([x_dim], residual_fn, form="weak", n_test=10)

# 2-D weak, total-degree (default)
loss = PINLossFunction([x_dim, y_dim], residual_fn, form="weak", n_test=10)
```

---

### 2. `compute_error` — `pinns_framework/metrics.py`

New public function for computing L2 / Linf / rel-L2 errors between numpy arrays.
Handles the `np.trapz` → `np.trapezoid` rename (NumPy 2.0 removed `trapz`).

```python
from pinns_framework import compute_error

err = compute_error(u_pred_np, u_exact_np, x_np)
# → {'L2': float, 'Linf': float, 'rel_L2': float}
```

Compatibility shim at module level:
```python
try:
    _np_trapezoid = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _np_trapezoid = np.trapz       # NumPy < 2.0
```

**Important:** do NOT use `getattr(np, "trapezoid", np.trapz)` — the fallback argument
is evaluated eagerly and raises `AttributeError` on NumPy 2.0 where `trapz` is gone.

---

### 3. New notebook — `notebooks/kan_burgers_sawtooth_weak.ipynb`

KAN + TD-PINs on the Hopf–Cole sawtooth (ν=0.07, T=0.02, Δt=0.005, 4 steps).
Compares strong vs weak form on the same `KAN (1, 16, 16, 16, 1)` architecture.

Sections: problem setup → IC/BC → residual → loss comparison → training (both forms) →
solution plots → error table → Legendre projection heatmap → KAN activation comparison →
conservation diagnostics → save checkpoints.

**Known gotchas in the notebook:**

#### `np.trapz` removed in NumPy 2.0
Conservation cell (Section 10) uses `np.trapezoid` directly. If the environment has
NumPy < 2.0 replace with `np.trapz`.

#### Device mismatch in `compute_projections`
`x_proj_cpu` is created on CPU then moved to the network device inside the loop:
```python
dev = next(nets[1].parameters()).device
x = x_proj_cpu.clone().to(dev).requires_grad_(True)
```
Do not create `x_proj` directly on a device — always clone-and-move.

#### KAN `deepcopy` fails after training
After training, pykan caches non-leaf tensors (from `create_graph=True` autograd) as
model attributes. `copy.deepcopy` raises:
```
RuntimeError: Only Tensors created explicitly by the user (graph leaves) support deepcopy
```
Fix: do a `no_grad` forward pass on the original network before `deepcopy` to flush
the cache and replace non-leaf tensors with plain data tensors:
```python
def _kan_cpu(kan_net, x_1d):
    dev   = next(kan_net.net.parameters()).device
    x_cpu = x_1d.detach().cpu().reshape(-1, 1)
    with torch.no_grad():            # ← flushes non-leaf cached tensors
        kan_net.net.eval()
        kan_net.net(x_cpu.to(dev))
    model = copy.deepcopy(kan_net.net).cpu()
    with torch.no_grad():
        model(x_cpu)
    model.get_act(x_cpu)
    return model, x_cpu
```
This pattern must be used in **every** notebook that visualizes KAN activations after training.

---

### 4. Mathematical reference — `docs/weak_form.md`

1,305-line document covering: strong-form failure modes (NTK, stiff PDEs, shocks),
classical weak formulation derivation, variational PINs loss, Legendre polynomials
(orthogonality, spectral convergence, recurrence, normalization), Monte Carlo integration,
multi-dimensional extensions, connections to VPINNs / WAN / wPINNs, approximation theory
(Parseval, K≈√N rule), full API reference, and practical guidelines.

---

## Files changed

| File | Change |
|---|---|
| `pinns_framework/loss.py` | Implemented `form="weak"`, added `_legendre_1d`, `_build_test_matrix`, `n_test`, `multi_dim_mode`, `test_fn` parameters |
| `pinns_framework/metrics.py` | Added `_np_trapezoid` compat shim, added `compute_error()` |
| `pinns_framework/__init__.py` | Exported `compute_error` |
| `notebooks/kan_burgers_sawtooth_weak.ipynb` | New notebook (created) |
| `docs/weak_form.md` | New mathematical reference (created) |
| `README.md` | Updated `PINLossFunction` API reference section |

---

## Things to watch on the new server

- **Python / NumPy version**: the codebase targets NumPy ≥ 2.0 (`np.trapezoid`).
  If the server has NumPy < 2.0 the `try/except` shim in `metrics.py` handles it,
  but the conservation cell in the notebook calls `np.trapezoid` directly — patch
  those calls to `np.trapz` if needed.
- **pykan version**: tested with `pykan==0.2.8`. Later versions may change internal
  attribute names (`spline_postacts`, `acts`, etc.) used in the activation visualization.
- **Device**: `_auto_device()` in `trainer.py` prefers MPS → CUDA → CPU. On a CUDA
  server this is transparent. The `_kan_cpu` deep-copy pattern (flush + deepcopy)
  is device-agnostic and works on all three.
- **Disk space**: pykan creates a `./model` checkpoint directory on every `PINArchitecture("KAN", ...)` call. On a server with limited `/tmp` or home quota this can fill up during long runs — delete `./model` between experiments.
- **Re-running partial cells**: `nets_strong` and `nets_weak` survive kernel state.
  The error, projection, visualization, and conservation cells can all be re-run
  independently without retraining, as long as `sols_strong` / `sols_weak` are in scope
  (re-run the evaluation cell first if they are not).
