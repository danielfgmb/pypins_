"""
Per-cell scaling factors and Dirichlet boundary imposition shared by every
PGDr-PIKAN solve, whether a single global cell (chapter 3 phase 1, see
solve.py) or a multi-cell DG mesh (phase 2+ -- refinement, Burgers
marching/space-time).
"""
import torch

from .ref_tensors import DTYPE


def jacobian_scale(term, h):
    """Reference-to-physical Jacobian factor for a cell of width h, under
    x = x_c + (h/2)*xi (design doc SS2.3): each term carries
    (h/2)^(1 - #derivatives).

    term : 'mass'    (0 derivatives)                  -> h/2
           'flux'     (1 derivative, Jacobians cancel) -> 1
           'viscous'  (2 derivatives)                  -> 2/h
    """
    if term == 'mass':
        return h / 2.0
    if term == 'flux':
        return 1.0
    if term == 'viscous':
        return 2.0 / h
    raise ValueError("term must be 'mass', 'flux', or 'viscous'")


def dirichlet_recombination_basis(N):
    """Basis-change matrix B, shape (N, N-2), whose columns are the raw-
    Legendre coefficients of the Shen/Dirichlet-recombined basis
    phi_k = P_k - P_{k+2}, k = 0..N-3.

    Each phi_k vanishes at BOTH endpoints automatically and exactly:
    phi_k(+-1) = P_k(+-1) - P_{k+2}(+-1) = (+-1)^k - (+-1)^(k+2) = 0, since
    (+-1)^2 = 1. Using {phi_k} as trial AND test space for a homogeneous-
    Dirichlet problem builds the boundary condition into the basis itself --
    no row-replacement (tau method) needed.

    NOTE ON WHY NOT TAU: a naive tau method (replacing 2 rows of the raw-
    Legendre Galerkin system with u(+-1)=g boundary rows) looks simpler but
    is singular here: K_ij = min(i,j)(min(i,j)+1) *saturates* once i or j
    exceeds the other, so for large N the top few rows/columns of K are
    almost redundant and neither candidate pair of rows to replace can
    distinguish the two highest-degree coefficients. The recombined basis
    below is the standard, well-conditioned fix (Shen 1994).

    A solution's raw-Legendre coefficients are recovered via c = B @ a,
    where a are the coefficients in the {phi_k} basis.
    """
    B = torch.zeros(N, N - 2, dtype=DTYPE)
    idx = torch.arange(N - 2)
    B[idx, idx] = 1.0
    B[idx + 2, idx] = -1.0
    return B


def constant_dirichlet_lift(N, g_left, g_right):
    """Raw-Legendre coefficients (length N) of the unique degree-1 polynomial
    u_lift(xi) = c0 + c1*xi with u_lift(-1)=g_left, u_lift(1)=g_right (P_0=1,
    P_1=xi, all higher modes exactly zero): c0=(g_left+g_right)/2,
    c1=(g_right-g_left)/2.

    Used to reduce a NON-homogeneous Dirichlet problem (e.g. Burgers with
    u(lo)=u_left, u(hi)=u_right) to a homogeneous one solved in
    dirichlet_recombination_basis's {phi_k}: writing u = u_lift + w, w
    vanishes at both endpoints exactly like every phi_k, so w's coefficients
    live in the SAME recombined basis as the Poisson case (solve.py).
    """
    c = torch.zeros(N, dtype=DTYPE)
    c[0] = (g_left + g_right) / 2.0
    if N > 1:
        c[1] = (g_right - g_left) / 2.0
    return c


def upwind_trace(a, u_left, u_right):
    """Upwind numerical trace for a LINEAR advective flux a*u at one face,
    given the trace value from the left cell (u_left) and the right cell
    (u_right) (design doc SS2.5's convective flux, linear-advection special
    case -- Burgers' actual flux u^2/2 needs the shock-dependent Godunov
    flux instead, not this).

    a >= 0 : flow left-to-right, upstream side is LEFT -> returns u_left
    a <  0 : flow right-to-left, upstream side is RIGHT -> returns u_right
    """
    return u_left if a >= 0 else u_right


def sipg_penalty(N, h, penalty_coeff=10.0):
    """Symmetric-interior-penalty coefficient sigma = penalty_coeff * N^2 / h
    for the viscous interface term (design doc SS2.5's flagged risk: "penalty
    parameter scaling with p^2/h"). penalty_coeff is an empirical constant,
    not derived -- tuned against a manufactured exact solution in
    notebooks/ch3/sc3.3, not fixed in advance.
    """
    return penalty_coeff * (N ** 2) / h


def godunov_flux_burgers(u_left, u_right):
    """Exact Godunov (Riemann-problem) numerical flux for Burgers' flux
    f(u) = u^2/2 at one interface (design doc SS2.5's flagged risk for the
    nonlinear case -- unlike linear advection's trivial sign-of-a upwind,
    upwind_trace above, the flux here depends on whether the local Riemann
    problem is a shock or a rarefaction):

        u_left >= u_right : shock, speed s=(u_left+u_right)/2
                             -> f(u_left) if s>=0 else f(u_right)
        u_left <  u_right : rarefaction
                             -> f(u_left)  if u_left  >= 0
                             -> f(u_right) if u_right <= 0
                             -> 0 (sonic point, f(0)=0)  otherwise

    u_left, u_right are 0-dim tensors (traces built from the coefficient
    vector, so differentiable); the branch actually taken is smooth in its
    inputs, which is all Newton's method needs -- the true flux's non-
    smoothness is a measure-zero set of interface states, standard for
    Godunov fluxes under Newton/Picard iteration.
    """
    if u_left >= u_right:
        s = 0.5 * (u_left + u_right)
        return 0.5 * u_left ** 2 if s >= 0 else 0.5 * u_right ** 2
    if u_left >= 0:
        return 0.5 * u_left ** 2
    if u_right <= 0:
        return 0.5 * u_right ** 2
    return torch.zeros_like(u_left)
