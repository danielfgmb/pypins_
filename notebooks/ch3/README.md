# Chapter 3 experiments — PGDr-PIKAN: a cell-local Legendre basis for KAN trial + Petrov–Galerkin test spaces

**Status note:** this chapter is at the start of its validation ladder. `sc3.0`
and `sc3.1` below (validation-ladder steps 1–3) are complete and their
numbers are final. Steps 4–6 — adaptive h-refinement, the nonlinear Burgers
benchmark solved in marching mode, and the global space–time mode — are
**not built yet** and are described only as proposed follow-up work, not
results.

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

**Results:**

- `verify_ref_tensors.py`: all of `M, K, C, T, D, T^L, T^R` match independent
  quadrature to the requested tolerance; runs in ~5s. Reproduces the design
  doc's own hand-worked check values exactly — `K11=2, K22=6, K33=12, K44=20,
  K13=2, K24=6, K35=12`, and the degree-4 `T^L` matrix.
- `kan_closure_test.py`: for 1-, 2-, and 3-layer Legendre-KANs (degree 4 per
  edge, so `p^L` = 4, 16, 64 respectively), the max residual between the raw
  network output and its Legendre re-projection is ≤ ~5e-12 in every case —
  i.e. machine-zero, confirming the closure argument holds at real
  (non-toy) depth.

**Figures/results:** `sc3.0/results/ref_tensors_report.json`,
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

## 3. Known caveats / proposed follow-up work (not built yet)

- **Bubnov–Galerkin only so far** — see `sc3.1`'s caveat above. Demonstrating
  the actual trial ≠ test Petrov–Galerkin mechanism needs a KAN with depth
  ≥ 2 solved nonlinearly in `theta` (or with inner layers frozen and only the
  outer layer solved directly per the design doc's §3.4 training fork) —
  not attempted yet.
- **No multi-cell mesh / interface flux yet** — `sc3.1` is deliberately a
  single cell to sidestep the design doc's flagged open risk (§2.5, the
  interface flux for the convective and viscous terms at cell boundaries).
  This is the first thing needed for adaptive h-refinement (step 4) and
  remains unverified.
- **No nonlinear (Burgers) benchmark yet** — steps 5–6 of the validation
  ladder (marching-mode Burgers with a closed-form traveling-shock solution,
  then the global space–time mode via Kronecker assembly) are not started;
  `triple_matrix`/`deriv_triple_matrix` in `ref_tensors.py` are verified but
  unused until then.
- **Degree growth `p^L`** — not yet a concern at the depths tested (`p^L`
  up to 64 in `sc3.0`), but will need a truncation policy once deeper
  networks are used in a real solve rather than just a closure check.
