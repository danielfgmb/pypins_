# Chapter 3 experiments — PGDr-PIKAN: a cell-local Legendre basis for KAN trial + Petrov–Galerkin test spaces

**Status note:** this chapter is at the start of its validation ladder.
`sc3.0` and `sc3.1` (validation-ladder steps 1–3) are complete and final.
`sc3.2` runs the nonlinear Burgers benchmark (step 5's benchmark) but only
in single-cell form, deliberately staying inside step 3's "no interfaces
yet" restriction — it is a bridge experiment, not step 5 itself. Genuine
multi-cell adaptive h-refinement (step 4), multi-cell marching (finishing
step 5), and the global space–time mode (step 6) are **not built yet** and
are described only as proposed follow-up work, not results.

Chapter 2 established that what actually gives a Physics-Informed Network
robustness at high frequency / near shocks is avoiding the second derivative
of the network's own approximation error (a genuinely hand-derived weak/IBP
form, not merely a "weak-sounding" loss aggregation — see `sc2.3`). Every
ch2 method, however, still uses a *global* MLP (or pyKAN) trial space and
Monte-Carlo- or quadrature-estimated projections against a fixed test-function
bank. This chapter's contribution goes one step further: a **cell-local
Legendre basis shared by both the KAN trial space and the Petrov–Galerkin
test space**, so that every mass/stiffness/convection/flux integral involved
is *closed-form* (no Monte Carlo noise at all), and so that mesh refinement
transfers coefficients *exactly* rather than by re-projection.

The framework code lives in `pinns_framework/pgdr/` (`ref_tensors.py`,
`kan.py`, `assembly.py`, `solve.py`), reusable across every sc3.x experiment,
mirroring how `pinns_framework/` itself backs every ch2 experiment.

---

## 1. sc3.0 — reference-tensor and KAN-closure verification (validation-ladder steps 1–2)

**What this checks:** every reference-element tensor needed downstream —
mass, stiffness, and convection matrices (`M`, `K`, `C`), the nonlinear-term
triple-product tensors (`T`, `D`), and the exact h-refinement subdivision
matrices (`T^L`, `T^R`) — computed two independent ways (closed-form
formulas / sympy exact symbolic integration vs. plain Gauss–Legendre
quadrature) and asserted to agree. Separately, that a multi-layer
Legendre-KAN's output is *exactly* a polynomial of degree `p^L` (the entire
argument for using a Legendre basis instead of pyKAN's B-splines, which do
not stay splines on a fixed knot set under composition).

**What was run, in order:**

1. **`verify_ref_tensors.py`** — three checks in one script:
   - Reprints the design doc's own hand-worked stiffness values
     (`K11=2, K22=6, K33=12, K44=20, K13=2, K24=6, K35=12`) and the degree-4
     `T^L` matrix straight from `pinns_framework.pgdr.stiffness_matrix` /
     `transfer_matrices` — all match exactly.
   - Calls `pgdr.verify_all()`, which cross-checks `M, K, C, T, D, T^L, T^R`
     against an *independently implemented* Gauss–Legendre quadrature (a
     separate numpy code path, not reusing the torch recurrence the rest of
     the package uses) — passes in ~5s.
   - A third, more specific check beyond `verify_all()`: takes a random
     "parent" polynomial (random Legendre coefficients), evaluates it
     directly on a left/right child sub-cell, and compares that against the
     child's coefficients obtained via `T^L`/`T^R` and re-expanded — i.e. it
     tests the actual doc §1.5 claim ("parent-restricted-to-child minus
     child-expansion simplifies to identically 0"), not just that `T^L`/`T^R`
     individually match quadrature. Agreement: ~7e-15 (left), ~3e-15 (right).
2. **`kan_closure_test.py`** — builds 1-, 2-, and 3-layer `LegendreKAN`s at
   per-edge degree `p ∈ {2,3,4,5,6}` (15 configurations total), evaluates
   each on a fine grid, re-projects onto the Legendre basis at degree `p^L`
   via quadrature, and measures the max residual against the raw network
   output. If closure were only approximate this residual would grow with
   `p^L`; instead it stays within a couple of decades of float64 round-off
   (`2.22e-16`) across the whole sweep — from `2.2e-16` at `p^L=2` up to
   `4.7e-11` at the largest case tested (`p^L=216`, 3 layers, degree 6) — all
   15/15 pass the `<1e-9` threshold.
3. **`plot_kan_closure.py`** — plots that sweep: residual vs `p^L`, one line
   per depth, log-log. The flatness of the lines (not their absolute height)
   is the actual result — see `figs/sc30_kan_closure.png`.

**Figures/results:** `sc3.0/figs/sc30_kan_closure.{png,pdf}`,
`sc3.0/results/ref_tensors_report.json`,
`sc3.0/results/kan_closure_report.json`.

---

## 2. sc3.1 — single-cell Poisson: the linear-solve demo (validation-ladder step 3)

**Setup:** the identical benchmark as `ch2/sc2.3` — `-u'' = f`,
`u(-1)=u(1)=0`, `u_exact = sin(k*pi*x)`, `k in {1,2,4,8,16}` — but solved
here as a **single cell spanning the whole domain** via a direct
Legendre-Galerkin linear solve: no interfaces (that is left for the
refinement work in step 4), no optimiser, no training loop of any kind.
Trial and test space are both `span{P_0,...,P_{N-1})` (`N` grows with `k`,
40→68, to keep resolving the target frequency), with Dirichlet boundary
conditions built into the basis itself via Shen's recombination
`phi_k = P_k - P_{k+2}` (each vanishes at both endpoints automatically) —
this is what `pinns_framework.pgdr.assembly.dirichlet_recombination_basis`
does, chosen over a naive boundary-row-replacement (tau) method after the
latter turned out to be singular here (the raw-Legendre stiffness matrix's
entries saturate at `min(i,j)`, so its last rows/columns become nearly
redundant and can't pin down the two highest-degree coefficients no matter
which two rows get replaced).

**What was run, in order:**

1. **`solve_poisson.py`** — for each `k`, one call to
   `pgdr.solve_poisson_single_cell`: assemble the scaled stiffness matrix,
   project `f = k²π²sin(kπx)` onto the Legendre basis via quadrature,
   recombine into the Dirichlet basis, one `torch.linalg.solve`, done (~2s
   wall time per `k`, all of it quadrature/linear-algebra, no epochs). It
   also independently recomputes the stiffness matrix by raw quadrature and
   compares it to the closed-form one (the doc's step-3 "analytic vs
   quadrature to machine precision" check), and copies the solved
   coefficients into a `pgdr.LegendreEdge`'s `theta` to confirm the result
   literally **is** one KAN layer's output, not just KAN-flavored linear
   algebra.
2. **`plot_error_vs_k.py`** — pulls `sc2.3`'s PINN `rel_L2` numbers
   (`strong`/`weak_projection`/`agnostic`, same `k` sweep) straight from
   `ch2/sc2.3/results/*_summary.json`, combines them with this experiment's
   own numbers into `results/comparison_vs_sc23.csv`, and plots both on one
   log-log chart.

**Results** (`results/summary_all.csv`):

| k | N | rel_L2 | Linf | closed-form-vs-quadrature K (relative) |
|---|---|---|---|---|
| 1 | 40 | 2.72e-14 | 2.75e-14 | 2.12e-11 |
| 2 | 40 | 2.58e-14 | 2.72e-14 | 2.12e-11 |
| 4 | 40 | 4.01e-14 | 6.06e-14 | 2.12e-11 |
| 8 | 44 | 1.73e-08 | 2.04e-08 | 2.58e-11 |
| 16 | 68 | 3.69e-06 | 4.57e-06 | 6.18e-11 |

Every run also copies its solved Legendre coefficients into a single
`pinns_framework.pgdr.LegendreEdge`'s `theta` and re-evaluates it, checking
the result against the manual polynomial reconstruction — the two agree to
0.0 (`kan_edge_vs_manual` in each `results/sc31_k{k}_summary.json`), i.e.
this genuinely **is** a single KAN layer's output, not merely
"KAN-flavored" linear algebra.

**Interpretation:** see `figs/sc31_error_vs_k.png` (overlaid directly
against `sc2.3`'s PINN-trained numbers, same benchmark, same `k` sweep).
The direct solve is **8 to 13 orders of magnitude** more accurate than any
of `sc2.3`'s trained PINNs across the entire `k` range, including at
`k=8,16` where every strong/weak_projection/agnostic PINN in `sc2.3`
effectively failed (`rel_L2` of order 1–13). It also runs in about 2 seconds
per `k`, with no training dynamics at all.

**Honest caveat (do not oversell this result):** because this is a single
KAN layer, trial and test space are identical (`span{P_0..P_{N-1}}` on both
sides) — this is **Bubnov–Galerkin**, not yet the trial ≠ test
Petrov–Galerkin structure that is this chapter's actual novel claim (design
doc §3.3: "a single layer would collapse to Bubnov–Galerkin"). What `sc3.1`
demonstrates is the *plumbing*: reference-tensor correctness, KAN↔Legendre
closure, and boundary imposition, all validated end-to-end in a real solve
reaching spectral accuracy. The trial ≠ test argument from network depth,
and the harder nonlinear/multi-cell setting where it should matter, are
follow-up work (see below).

**Figures:** `sc3.1/figs/sc31_error_vs_k.{png,pdf}`.

---

## 3. sc3.2 — single-cell Burgers: where "one cell, one layer" starts to break

**Setup:** the identical benchmark as `ch2/sc2.4` — the viscous-Burgers
traveling shock with a closed-form exact solution for all `t`
(`u_L=2, u_R=0`, domain `[-5,5]`, `T_END=1`) — solved via
`pinns_framework.pgdr.solve_burgers_single_cell`: Crank–Nicolson time-
marching with Newton's method solving the nonlinear system at each step
(doc §2.4), still a single cell spanning the whole domain (no interior
interfaces, so §2.5's flagged open risk still doesn't come up) and still
one KAN layer (Bubnov–Galerkin, same phase-1 caveat as `sc3.1`). This is
the first real exercise of `ref_tensors.deriv_triple_matrix` (the nonlinear
flux tensor `D`) in an actual solve, and of non-homogeneous Dirichlet BCs
(`u_left ≠ u_right`, handled via `assembly.constant_dirichlet_lift` — the
domain edges are checked to sit at the exact boundary values to float
round-off for the whole simulated time window, which is what licenses
treating the BC as time-constant rather than needing a time-varying lift).

**Two viscosities, deliberately contrasted:**

- `nu=0.02` — identical to `sc2.4`. Shock width `≈ 4·nu/(u_L−u_R) = 0.04`,
  i.e. **0.4% of the domain width**.
- `nu=0.2` — 10x milder. Shock width `≈0.4`, i.e. 4% of the domain — the
  same feature, ten times wider relative to the cell.

**Results** (`results/summary_all.csv`):

| nu | N | mean_L2 | final_L2 | wall time |
|---|---|---|---|---|
| 0.02 | 60 | 5.57e-2 | 5.46e-2 | 32s |
| 0.2 | 40 | 1.83e-3 | 1.76e-3 | 7s |

Newton converges in 3–4 iterations every time step for both cases — the
nonlinear solve itself is well-behaved; the difference between the two
rows is entirely about whether the trial space (one global polynomial) can
*represent* the target, not about solver robustness. Confirmed directly: a
pure Legendre projection of the exact `t=0` profile (no dynamics at all)
gives `rel_L2 = 6.74e-2` at `N=40`, `4.74e-2` at `N=60` for `nu=0.02` —
essentially the same number the full time-marched solve lands on, meaning
the error is coming entirely from *approximation*, not from time-stepping
or Newton convergence.

**`figs/sc32_solutions.png`** makes this visible directly: at `nu=0.2` the
single-cell solution tracks the exact profile to the eye; at `nu=0.02` it
shows textbook Gibbs-type ringing around the front — a single global
polynomial cell genuinely cannot represent a feature this localized without
an impractically large mode count (`N=200` only gets the pure-projection
error down to `~7e-3`, and generating the nonlinear-term tensor `D` at that
size is no longer cheap — cubic-ish growth in the sympy/wigner_3j
generation, ~31s already at `N=60`).

**`figs/sc32_comparison_vs_sc24.png`** — the honest, slightly more nuanced
comparison against `sc2.4`'s 4 *trained* methods on the harder `nu=0.02`
case: the single-cell direct solve (`5.57e-2`) is **worse** than `sc2.4`'s
`strong` (`6.79e-4`) and `weak_projection` (`4.68e-3`) — those PINNs, with a
global MLP trial space and gradient descent, can concentrate representation
power near the shock in a way a fixed global polynomial basis cannot — but
it is actually **better** than `sc2.4`'s `agnostic` (`1.27e-1`) and
`agnostic_weak_time` (`7.50e-2`), and gets there in ~30 seconds of direct
linear algebra instead of a training run.

**Takeaway:** this is *not* a bug or a regression from `sc3.1` — it is
exactly what motivates chapter 3 phase 2. A single global cell is precisely
the setting the design doc's h-refinement (§4) exists to fix: concentrate
small, cheap cells where the residual indicator is large (near the shock)
and leave the rest coarse, rather than paying for a huge global mode count
everywhere to resolve one narrow feature.

**Figures:** `sc3.2/figs/sc32_{solutions,comparison_vs_sc24}.{png,pdf}`.

---

## 4. Known caveats / proposed follow-up work (not built yet)

- **Bubnov–Galerkin only so far** — see `sc3.1`'s caveat above. Demonstrating
  the actual trial ≠ test Petrov–Galerkin mechanism needs a KAN with depth
  ≥ 2 solved nonlinearly in `theta` (or with inner layers frozen and only the
  outer layer solved directly per the design doc's §3.4 training fork) —
  not attempted yet.
- **No multi-cell mesh / interface flux yet** — `sc3.1` and `sc3.2` are both
  deliberately a single cell to sidestep the design doc's flagged open risk
  (§2.5, the interface flux for the convective and viscous terms at cell
  boundaries). This is the first thing needed for adaptive h-refinement
  (step 4) and remains unverified. `sc3.2`'s `nu=0.02` result is the
  concrete demonstration of *why* it's needed: a single global cell cannot
  resolve a genuinely sharp front, no matter how many modes (see `sc3.2`
  above) — refinement should let a handful of small cells near the shock do
  what an impractically large global `N` would otherwise have to.
- **Marching-mode Burgers exists only in single-cell form** — `sc3.2`
  exercises Crank–Nicolson + Newton and the nonlinear flux tensor `D`
  (`ref_tensors.deriv_triple_matrix`) for the first time in a real solve,
  but step 5 of the validation ladder (proper multi-cell marching) still
  needs the interface flux above; the global space–time mode (step 6, via
  Kronecker assembly with the plain triple-product tensor `T`) hasn't been
  started at all.
- **Degree growth `p^L`** — not yet a concern at the depths tested (`p^L`
  up to 64 in `sc3.0`), but will need a truncation policy once deeper
  networks are used in a real solve rather than just a closure check.
