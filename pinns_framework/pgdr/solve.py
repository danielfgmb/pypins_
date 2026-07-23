"""
Direct linear solves for PGDr-PIKAN (chapter 3). Phase 1 scope: a single
cell spanning the whole domain, steady linear BVPs, solved by one
torch.linalg.solve -- no training loop (design doc SS7 validation-ladder
step 3: "linear solve vs exact solution"). Newton iteration (nonlinear
Burgers) and Crank-Nicolson time-marching are added to this same module
once that work starts (phase 2+) -- not stubbed here.
"""
import numpy as np
import torch

from .ref_tensors import DTYPE, stiffness_matrix, legendre_basis, legendre_and_deriv
from .assembly import jacobian_scale, dirichlet_recombination_basis


def _project_rhs(f_fn, domain, N, n_quad):
    """RHS_j = int f(x) P_j(xi(x)) dx via Gauss-Legendre quadrature. Exact to
    float round-off whenever n_quad comfortably exceeds f's effective
    Legendre spectral decay point on this cell (f need not be a polynomial --
    e.g. a sinusoid here -- "comfortably" just means past where its
    quadrature error has decayed below float round-off)."""
    lo, hi = domain
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    xi = torch.as_tensor(xq, dtype=DTYPE)
    x_phys = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
    w = torch.as_tensor(wq, dtype=DTYPE) * (hi - lo) / 2.0  # dx = (h/2) dxi
    f_vals = f_fn(x_phys)
    P = legendre_basis(xi, N)
    return (P * (f_vals * w).unsqueeze(-1)).sum(dim=0)


def solve_poisson_single_cell(f_fn, exact_fn, domain, N, n_quad=400, n_eval=2000):
    """-u'' = f on `domain`, u(lo)=u(hi)=0, single cell, Legendre-Galerkin
    (doc SS2, single-cell special case -- no interface flux term). Trial =
    test = span{phi_0..phi_{N-3}} (assembly.dirichlet_recombination_basis),
    which is the same {P_0..P_{N-1}} Bubnov-Galerkin space from doc SS3.3
    re-expressed in a basis that vanishes at both endpoints automatically
    (chapter 3 phase-1 caveat: a single KAN layer collapses to Bubnov-
    Galerkin; the trial != test Petrov-Galerkin argument needs depth --
    phase 2+).

    Returns a dict with:
      c                   -- (N,) Legendre coefficients of the solution
      rel_L2, Linf         -- error vs exact_fn on a fine eval grid
      quad_vs_closed_form  -- max |closed-form K - quadrature-recomputed K|
                              relative to max|K|, the doc's step-3 "analytic
                              vs quadrature stiffness to machine precision"
                              check
    """
    lo, hi = domain

    K = stiffness_matrix(N) * jacobian_scale('viscous', hi - lo)
    b = _project_rhs(f_fn, domain, N, n_quad)

    # Dirichlet recombination (Shen 1994): phi_k = P_k - P_{k+2} vanishes at
    # both endpoints by construction, so solving in {phi_k} builds u(+-1)=0
    # into the basis -- no boundary row-replacement needed (see
    # assembly.dirichlet_recombination_basis's docstring for why a naive tau
    # row-replacement is singular for this particular K).
    B = dirichlet_recombination_basis(N)
    K_tilde = B.T @ K @ B
    b_tilde = B.T @ b
    a = torch.linalg.solve(K_tilde, b_tilde)
    c = B @ a

    # -- analytic (closed-form) vs quadrature stiffness, to machine precision --
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    xi = torch.as_tensor(xq, dtype=DTYPE)
    _, dP = legendre_and_deriv(xi, N)
    w = torch.as_tensor(wq, dtype=DTYPE)
    K_quad = torch.einsum('qi,qj,q->ij', dP, dP, w) * jacobian_scale('viscous', hi - lo)
    # Relative, not absolute: K's entries grow like N^2 (SS1.3's
    # min(i,j)(min(i,j)+1)), so an absolute residual alone overstates the
    # disagreement at large N.
    quad_vs_closed_form = float((K - K_quad).abs().max() / K.abs().max())

    # -- error vs exact solution on a fine eval grid --
    x_eval = torch.linspace(lo, hi, n_eval, dtype=DTYPE)
    xi_eval = 2.0 * (x_eval - lo) / (hi - lo) - 1.0
    P_eval = legendre_basis(xi_eval, N)
    u_pred = (P_eval * c).sum(dim=-1)
    u_exact = exact_fn(x_eval)

    diff = u_pred - u_exact
    l2 = float(torch.sqrt((diff ** 2).mean()))
    denom = float(torch.sqrt((u_exact ** 2).mean())) + 1e-12
    linf = float(diff.abs().max())

    return {
        'c': c,
        'rel_L2': l2 / denom,
        'Linf': linf,
        'quad_vs_closed_form': quad_vs_closed_form,
    }
