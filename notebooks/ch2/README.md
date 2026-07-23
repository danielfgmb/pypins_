# Chapter 2 experiments — strong vs. weak (variational) formulations for Physics-Informed Networks

This folder collects a series of numerical experiments comparing **strong-form** against
**weak-form (variational, Petrov–Galerkin-style)** loss formulations for Physics-Informed
Networks (PINs), built on the shared `pinns_framework` package (see `/home/df.gomezb/Tesis/pinns_framework/`).
The motivating question throughout is simple to state and hard to answer cleanly:

> Does moving from a pointwise ("strong") residual loss to a weak/variational one actually
> help — and specifically, does it help *because* it avoids differentiating the network's own
> approximation error through a second derivative?

Five sub-experiments (`sc2.0`–`sc2.4`) approach this from different angles: two use the viscous
Burgers equation (a genuine, time-dependent, shock-forming PDE) and one uses a much simpler,
literature-standard static benchmark (1D Poisson with a high-frequency solution) precisely
*because* the Burgers experiments turned out to be confounded by other effects. Reading the
sections in order (sc2.1 → sc2.2 → sc2.3 → sc2.4) roughly follows the actual investigative path:
started with a "natural" physical problem, found the picture muddy, and kept simplifying and
re-deriving until the underlying mechanism became visible.

**Status note:** all results below are final. `sc2.1`'s and `sc2.2`'s `vanilla_agnostic` /
`vanilla_agnostic_weak_time` rows finished their full-budget rerun (510000 / 410000 epochs,
matching every other method in those two experiments) after a bug fix (see §8) — the tables below
already reflect that final run.

---

## 1. Shared vocabulary: the three loss "forms"

Every experiment reuses the same `PINLossFunction` abstraction from `pinns_framework/loss.py`,
which wraps a user-supplied residual function and turns it into a scalar loss in one of three ways:

- **`strong`** — the classical PINN loss. `residual_fn` returns the pointwise PDE residual
  `R(x_i)` (e.g. `u_t + u u_x - nu u_xx` evaluated via autodiff at each collocation point), and the
  loss is simply `mean(R(x_i)^2)`. This requires differentiating the network **twice** in space
  (for the diffusion term `u_xx`) — for Poisson-type problems, differentiating it twice overall.

- **`weak_projection`** — a generic Petrov–Galerkin projection built into the framework. It still
  calls the *same* pointwise residual function as `strong` (so it still needs the second
  derivative!), but instead of squaring the residual directly, it projects it onto a bank of
  Legendre polynomial test functions `v_k` and squares *those* projections:
  `L = mean_k( (1/N) sum_i R(x_i) v_k(x_i) )^2`. This is the framework's automatic, "I don't want
  to derive anything by hand" weak form.

- **`agnostic`** — `residual_fn` returns the *finished scalar loss*; `PINLossFunction` does nothing
  extra. This is where a genuine, hand-derived weak form lives: integration by parts (IBP) is
  applied on paper first, moving a derivative off of `u` and onto the (known, exact) test function
  `v_k`, so that the residual you actually compute **only needs first derivatives of `u`**. This is
  the only one of the three forms that can structurally avoid the "second derivative of network
  noise" problem — `weak_projection` cannot, because it reuses the strong-form residual internally.

A recurring theme in this work is that this distinction — *"does the residual function itself ever
compute `u_xx`?"* — matters much more than whether the loss is "weak-sounding" in name. See sc2.3
below for the cleanest demonstration.

### Monte Carlo subtlety (domain-volume correction)

`weak_projection`'s projection `(1/N) sum_i R(x_i) v_k(x_i)` is a Monte Carlo *mean*, i.e. an
estimate of `E[R v_k] = (1/|Omega|) integral(R v_k dx)`, not of the integral itself. Left alone
this silently rescales the PDE loss by `1/|Omega|` relative to the IC/BC loss terms. `PINLossFunction`
exposes a `domain_volume` parameter (`|Omega|` for TD-PINNs, `|Omega|*T` for Vanilla, since Vanilla's
random collocation points also carry an independent random `t`) to correct this; every script in
this repo that uses `weak_projection` passes it explicitly.

### Two hand-derived "agnostic" variants (sc2.1, sc2.2, sc2.4)

For Vanilla PINNs specifically (single space-time network, `u(t,x)`), there are two different ways
to build an `agnostic` weak form, and both appear in the results below:

- **`vanilla_agnostic_loss`** — "space-weak only": `u_t` is computed exactly via autodiff (not
  weakened), and only the spatial part of the residual is integrated by parts against a vanishing
  Legendre test function `v_k(x)` (`v_k(+-1) = 0`, so the IBP boundary term drops and the BC loss
  handles the Dirichlet condition separately).
- **`vanilla_agnostic_weak_time_loss`** — "space+time weak": tests the residual against a **product**
  basis `v_k(x) * psi_m(t)` instead of just `v_k(x)`, giving `K_x * K_t` orthogonality conditions
  instead of `K_x`. `u_t` is still exact via autodiff in both variants — the "time weakening" here
  means "tested against more directions in time", not "differentiated by parts in time" (there is no
  natural boundary to eliminate at `t=0`, since that's exactly where the IC is enforced).

Both variants are estimated via **Monte Carlo with the two-sample (U-statistic) trick**: the
collocation batch is split into two independent halves A and B, and the loss is `R_A * R_B` instead
of `R_hat^2`. This is deliberate and documented in `sc2.1/NOTA_perdida_negativa.md`: a plain-squared
single-sample MC estimate `R_hat^2` is a *biased* estimator of the true `R^2`
(`E[R_hat^2] = R^2 + Var(R_hat)`), so it never gets below an artificial noise floor even once the
network has actually converged. `R_A*R_B` is unbiased (`E[R_A R_B] = R^2`) at the cost that a single
batch's value **can and regularly does go negative** near convergence — this is expected and is not
a bug; only the independently-tracked real error (see below) should be trusted, not the sign or
magnitude of this particular loss.

*(A tensor-product-batched, fully deterministic quadrature version of both variants was also
implemented and made numerically correct during this session — it removes the Monte Carlo noise
question entirely — but was ultimately abandoned in favor of the simpler MC formulation above: its
per-time-slice batching structure required 5–25x more, smaller, sequential GPU kernel launches per
epoch, which made it extremely slow under any concurrent GPU load. See "Framework changes" below.)*

### TD-PINNs vs. Vanilla PINNs

- **TD-PINNs**: one small network per discrete time step, trained sequentially via Crank–Nicolson-style
  time-stepping (`u^n` depends on the previous step's `u^{n-1}`, detached). Time is *always* handled
  strongly (a real finite-difference discretization), regardless of whether the spatial part of the
  loss is strong/weak_projection/agnostic.
- **Vanilla PINNs**: a single network `u(t,x)` over the whole space-time domain, trained on randomly
  sampled `(t,x)` collocation points (or, when needed, small tensor-product batches — see below).

### Real error vs. training loss

Every run in `sc2.1`–`sc2.4` tracks two independent numbers during training: the training loss
(whatever form-specific quantity is being minimized) and a periodically-evaluated **real relative
L2 error** against the known exact/analytical solution (`error_fn`, logged every `ERROR_EVERY`
epochs into `trainer.loss_history`, and also printed live in the console log). These are **not**
comparable across forms — `agnostic`'s loss can look enormous (`~1e4`–`1e6`) purely because of its
own internal normalization, while its real error is perfectly reasonable. **Only the real error
should be used to judge whether a method is actually working**; see `sc2.3/figs/sc23_loss_vs_error.png`
for a direct demonstration of this disconnect.

---

## 2. sc2.0 — MLP vs. KAN backbone, basic sanity sweep

**Setup:** toothsaw (sawtooth-like) Burgers profile, Cole–Hopf exact solution, `nu=0.07`. 2×2 grid:
{MLP, KAN} × {TD-PINNs, Vanilla}, small matched epoch budgets (KAN is ~2.5–5x slower per epoch in
wall-clock time than MLP, so both architectures were given the *same epoch count*, not the same
wall-clock budget, per explicit instruction).

**Result:** KAN converged dramatically faster *per epoch* than MLP at this small budget (TD-PINNs:
mean_L2 0.0117 KAN vs. 0.270 MLP at 4000 epochs; Vanilla: 0.0115 KAN vs. 0.0208 MLP at 4200 epochs),
though at much higher wall-clock cost per epoch. This experiment predates and is independent of the
strong-vs-weak investigation in sc2.1–sc2.4; included here for completeness. Figures: `sc2.0/figs/`.

---

## 3. sc2.1 — Burgers sawtooth, nu=0.07 (smooth periodic IC)

**Physics:** periodic sawtooth IC, exact Cole–Hopf solution, `nu=0.07` (mild viscosity — the
solution has a smooth but fairly steep front, not a true shock). Domain `x in [0, 2*pi]`,
`T_END=0.025`, 5 TD-PINNs steps.

**Methods (7):** `td_strong`, `td_weak_projection`, `td_agnostic`, `vanilla_strong`,
`vanilla_weak_projection`, `vanilla_agnostic`, `vanilla_agnostic_weak_time`. TD methods: 100000
epochs/step x 5 steps = 500000. Vanilla methods: 510000 epochs total (matched to TD's total for a
fair epoch-for-epoch comparison).

**Results** (`mean_L2` / `final_L2` over the 6 saved time snapshots):

| method | epochs | mean_L2 | final_L2 |
|---|---|---|---|
| TD-PINNs (strong) | 500000 | **4.42e-3** | 6.00e-3 |
| TD-PINNs (weak_projection) | 500000 | 2.29e-1 | 2.98e-1 |
| TD-PINNs (agnostic) | 500000 | 1.22e-1 | 1.47e-1 |
| Vanilla (strong) | 510000 | 1.30e-2 | 1.23e-2 |
| Vanilla (weak_projection) | 510000 | 2.19e-2 | 4.26e-2 |
| Vanilla (agnostic) | 510000 | 2.94e-2 | 5.44e-2 |
| Vanilla (agnostic, weak-time) | 510000 | 4.25e-2 | 8.28e-2 |

**Interpretation:** `strong` wins decisively for both schemes on this mild, well-resolved problem.
`weak_projection` and `agnostic` are both meaningfully worse for TD-PINNs specifically — a genuinely
counter-intuitive result investigated at length in `sc2.1/NOTA_analisis_agnostic.md`: a **missing
domain-volume normalization constant** in the Monte Carlo projection estimators was found and fixed
(a factor of `|Omega|` or `|Omega|*T` — see `PINLossFunction`'s `domain_volume` parameter above),
which substantially improved `vanilla_agnostic` (mean_L2 0.244 → 0.128 in an earlier, shorter-budget
version of this experiment) but did not fully close the gap to `strong`. At the full matching
510000-epoch budget, `vanilla_agnostic` reaches mean_L2=0.029 — competitive with (about 1.3x)
`weak_projection`'s 0.022, and both clearly behind `strong`'s 0.013. `vanilla_agnostic_weak_time`
(the richer, product-test-basis variant) is worse than the space-only `vanilla_agnostic` here
(0.042 vs 0.029) — more simultaneous test-function constraints did not pay off on this smooth,
well-resolved problem. The remaining gap to `strong` is consistent with this problem's smoothness:
with a nu large enough that the front is not close to a true discontinuity, `strong`'s extra
pointwise accuracy simply isn't punished by instability the way it is in sc2.3/sc2.4's
sharper-front experiments.

**Figures:** `sc2.1/figs/sc21_{errors,conservation,solutions,spectral,loss_curves}.png`.

---

## 4. sc2.2 — Burgers, Gaussian IC swept toward the shock regime (nu sweep)

**Physics:** Gaussian initial condition (not periodic sawtooth), same domain, `T_END=0.6`, only
`NT=4` TD-PINNs steps (kept deliberately coarse to bound the total compute of a 5-nu x 6/7-method
sweep). `nu in {0.05, 0.02, 0.01, 0.005, 0.001}` — the point of this experiment is to watch what
happens to each method **as nu shrinks toward the near-inviscid, shock-forming regime**.

**Methods:** same 7 as sc2.1. TD: 100000 epochs/step x 4 steps = 400000. Vanilla: 410000 (matched).

**Key finding — first look:** `notebooks/ch2/sc2.2/figs/sc22_analytical_shock.png` and
`sc22_nu_sweep_L2.png` show that at `nu=0.001` the true (Hopf–Lax vanishing-viscosity limit)
solution develops a boundary layer only ~0.001 units wide — three orders of magnitude thinner than
the domain — a legitimate near-discontinuity.

**Results at nu=0.01** (roughly the middle of the sweep; full table in `sc2.2/results/nu_sweep_summary.csv`):

| method | mean_L2 | final_L2 |
|---|---|---|
| TD-PINNs (strong) | 5.76e-2 | 1.21e-1 |
| TD-PINNs (weak_projection) | 7.14e-2 | 1.54e-1 |
| TD-PINNs (agnostic) | 5.87e-2 | 1.26e-1 |
| Vanilla (strong) | **2.65e-2** | 5.51e-2 |
| Vanilla (weak_projection) | 1.25e-1 | 2.28e-1 |
| Vanilla (agnostic) | 1.15e-1\* | 1.93e-1\* |
| Vanilla (agnostic, weak-time) | 1.17e-1\* | 2.12e-1\* |

\* *same caveat as sc2.1 — final full-budget rerun in progress at time of writing.*

**Interpretation:** `Vanilla (strong)` shows a striking **U-shaped** error curve across the nu sweep —
it *improves* as nu drops from 0.05 to ~0.01, then gets worse again toward nu=0.001 — while every
weak/agnostic method is nearly flat across the whole sweep, sitting well above `strong` everywhere.
At this coarse a time discretization (4 steps) and with uniform (non-adaptive) collocation, the
dominant error source for *every* method appears to be the discretization itself, not the
strong-vs-weak distinction — none of the 7 methods actually resolve the sharp nu=0.001 boundary
layer well (see `sc2.2/figs/sc22_spectral_vs_nu.png`: all methods' high-frequency spectral error
saturates near the analytical ceiling regardless of nu). This is the experiment that motivated
building sc2.3 as a cleaner, literature-grounded test of the same question, since this one's
signal is confounded by coarse time-stepping.

**Figures:** `sc2.2/figs/sc22_{nu_sweep,nu_sweep_L2,solutions_nu*,spectral_nu*,spectral_vs_nu,analytical_shock}.png`.

---

## 5. sc2.3 — 1D Poisson, high-frequency benchmark (the clean result)

**Why this experiment exists:** sc2.1/sc2.2 (real Burgers dynamics) turned out to be dominated by
discretization noise rather than the strong-vs-weak distinction itself. This experiment strips
everything else away: a **static** (no time-stepping) 1D Poisson problem,

```
-u''(x) = f(x),  x in [-1, 1],  u(-1) = u(1) = 0
u_exact(x) = sin(k*pi*x)  =>  f(x) = k^2 * pi^2 * sin(k*pi*x)
```

for `k in {1, 2, 4, 8, 16}` — the canonical benchmark from Kharazmi, Zhang & Karniadakis, *"Variational
Physics-Informed Neural Networks For Solving Partial Differential Equations"* (VPINN,
arXiv:1912.00873), specifically chosen because the literature already predicts that strong-form
PINNs degrade as the target frequency `k` grows (fitting a high-frequency, unit-amplitude target
forces large curvature into a small network, and differentiating that curvature twice amplifies
whatever approximation error remains — the same spectral-bias mechanism discussed in Wang, Teng &
Perdikaris, *"Understanding and mitigating gradient pathologies in PINNs"*).

**Methods:** `strong`, `weak_projection` (both still use `-u_xx - f` internally — see the vocabulary
section above, this is the crucial point), and `agnostic` — a **hand-derived IBP weak form**,
`R_k = integral(u' v_k' dx) - integral(f v_k dx)`, which only ever needs `u'` (never `u''`). Trained
with a bare Adam loop (no `PINTrainer`, since there's no time dimension), 100000 epochs, 100 Legendre
test functions, 10000 collocation points, checkpointed every 10000 epochs.

**Results** (`rel_L2` at 100k epochs, 100 test functions):

| k | strong | weak_projection | agnostic (true IBP weak form) |
|---|---|---|---|
| 1 | **1.05e-4** | 5.47e-1 | 1.28e-1 |
| 2 | **5.56e-4** | 9.98e-1 | 2.18e-1 |
| 4 | **4.79e-2** | 1.89e0 | 4.61e-1 |
| 8 | 1.33e1 | 3.31e0 | **6.78e0** |
| 16 | 1.16e1 | 3.56e0 | **3.96e0** |

**Interpretation — this is the clean demonstration the other experiments couldn't cleanly show:**

- At low k (1, 2, 4), `strong` is by far the best — an easy, smooth target is trivial to fit
  pointwise, and the weak forms need many simultaneous test-function constraints satisfied, which is
  a slower optimization landscape (their low-k gap did **not** close with more epochs — tested at
  8k/10k/100k epochs, essentially unchanged — so it's a real characteristic, not underfitting).
- Past k=4, `strong`'s error **explodes** (11–13x worse than at k=4) and is **non-monotonic across
  training budgets** (its k=16 error was 23.8 at 10k epochs, 41.7 at a different 10k-epoch config,
  10.6 at 100k epochs) — i.e. genuinely *unstable*, not just slow to converge.
- `weak_projection` (still computes `u''`!) inherits much of that fragility, though less severely
  than `strong`.
- `agnostic` (the only form that never computes `u''`) is the most *stable* across the whole k range
  and across training budgets — its error grows only mildly with k, and it doesn't blow up. At k=16
  it's ~3x better than `strong` and comparable to `weak_projection`.
- `sc2.3/figs/sc23_loss_vs_error.png` makes explicit that `agnostic`'s large-looking raw training
  loss numbers do **not** indicate a worse fit — the real error tells the opposite story.

This confirms the literature's actual claim precisely: **it's avoiding the second derivative, not
"being weak-form-flavored" in general, that gives robustness** — `weak_projection`, despite its name,
doesn't get this benefit because its residual function still differentiates `u` twice internally.

**Figures:** `sc2.3/figs/sc23_{error_vs_k,solutions_all_k,loss_curves,loss_vs_error,analytical}.png`.

---

## 6. sc2.4 — Burgers traveling shock (exact solution, genuinely time-dependent)

**Why this experiment exists:** sc2.3 is clean but static — it can't test the `agnostic` vs
`agnostic_weak_time` (space-only-weak vs. space+time-weak) distinction, which only makes sense with
a real time dimension. This experiment picks a Burgers problem with a **closed-form exact solution
for all t** (no need to wait for a smooth IC to steepen into a shock, unlike sc2.2):

```
u(x,t) = (u_L+u_R)/2 - (u_L-u_R)/2 * tanh[(u_L-u_R)(x - s*t)/(4*nu)]
s = (u_L+u_R)/2     (Rankine–Hugoniot shock speed)
```

the classical viscous Burgers traveling-shock profile (`u_L=2, u_R=0 => s=1`, `nu=0.02`). Verified
numerically to satisfy the PDE exactly (residual = 0.0) before use.

**Methods (4, Vanilla-only):** `strong`, `weak_projection`, `agnostic` (space-only-weak),
`agnostic_weak_time` (space+time-weak, product test basis) — all now using the Monte Carlo
two-sample-trick formulation (same as sc2.1/sc2.2). 20000 epochs, checkpointed every 2000.

**Results** (`mean_L2` / `final_L2` across `t in {0, 0.25, 0.5, 0.75, 1.0}`):

| method | mean_L2 | final_L2 |
|---|---|---|
| strong | **6.79e-4** | 8.72e-4 |
| weak_projection | 4.68e-3 | 8.34e-3 |
| agnostic | 1.27e-1 | 1.21e-1 |
| agnostic (weak-time) | 7.50e-2 | 1.29e-1 |

**Interpretation:** at this epoch budget, `strong` and `weak_projection` both track the moving shock
almost perfectly (see `sc2.4/figs/sc24_solutions_near_shock.png`). The MC-based `agnostic` variants
are far worse and, importantly, show **systematic (not just noisy) errors**: `agnostic` develops a
vertical offset bias (over/undershoots the far-field values `u_L`/`u_R`), and `agnostic_weak_time`
shows a phase lag (the predicted shock front trails the true one). This is consistent with the
sc2.3 finding that the MC-based agnostic forms need substantially more tuning (learning rate,
BC/PDE weighting) or training budget to reach the same precision as the pointwise forms on
"easy" regions of a solution — the shock-front is genuinely being resolved directionally correctly,
just not tightly.

**Figures:** `sc2.4/figs/sc24_{error_vs_t,solutions_near_shock}.png`.

---

## 7. Framework changes made during this work (`pinns_framework/`)

- **`PINLossFunction`** (`loss.py`) gained `domain_volume` (fixes the missing Monte-Carlo
  normalization constant in `weak_projection`, see §1).
- **`PINTrainer`** (`trainer.py`) gained:
  - `track_best` — snapshot and restore the network state from whichever epoch had the lowest
    training loss, instead of always keeping the final epoch.
  - `error_fn` / `error_every` — periodically evaluate a real (ground-truth) error metric during
    training, independent of whatever the training loss looks like, logged into `loss_history` and
    printed live in the console.
  - `ic_epochs` — separate, smaller epoch budget for TD-PINNs' initial-condition fitting step (which
    is a much easier problem than a full PDE step).
  - `time_colloc_nodes` — build Vanilla PDE collocation as a genuine tensor-product `(t,x)` batch
    (fixed time nodes x independent spatial samples per node) instead of one independent random `t`
    per point. *(Built to support a quadrature-based agnostic loss that was ultimately abandoned in
    favor of the simpler Monte Carlo formulation — see §1 — but left in the framework as it may be
    useful again.)*
  - `checkpoint_every` / `checkpoint_dir` / `checkpoint_prefix` — periodically save
    `net.state_dict()` during training (independent of `track_best`), so long runs interrupted by
    GPU contention or preemption don't lose all progress.

## 8. Known caveats / things to watch

- **GPU contention**: many of these runs were executed concurrently on a single shared GPU; wall-clock
  times recorded in `results/*_summary.json` reflect whatever contention existed at the time and
  should not be used to compare method speed against each other.
- **sc2.1/sc2.2 `vanilla_agnostic*` full-budget rerun**: see the status note at the top of this file.
- **A real bug was found and fixed** during this session: `vanilla_agnostic_loss` and
  `vanilla_agnostic_weak_time_loss` were, for a period, accidentally implemented identically in
  `sc2.1/common.py` and `sc2.2/common.py` (both using the product `v_k(x)*psi_m(t)` test basis) —
  caught by comparing their outputs bit-for-bit on the same batch. `sc2.4`'s versions were correct
  from the start. Fixed by making `vanilla_agnostic_loss` spatial-test-functions-only again.
- **sc2.2's `vanilla_agnostic` was, for a period, nearly nu-independent** (mean_L2 identical to 3
  significant figures across all 5 nu) — a symptom of the same bug above combined with a short
  (10000-epoch) training budget; resolved by the same fix, real (if modest) nu-dependence confirmed
  afterward.
