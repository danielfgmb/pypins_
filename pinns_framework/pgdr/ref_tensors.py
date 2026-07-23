"""
Reference-element tensors for PGDr-PIKAN (chapter 3): the cell-local
Legendre basis shared by the KAN trial space and the Petrov-Galerkin test
space. All objects here are exact integrals on the reference interval
xi in [-1, 1] against Legendre polynomials P_0 .. P_{N-1} -- see
notebooks/ch3/ for the design doc this implements ("PGDr-PIKAN --
Implementation reference"); section refs below (SS1.x) point there.

Closed-form pieces (mass/stiffness/convection/endpoints, SS1.1-1.3, 1.6) are
plain vectorised formulas -- fast for any N, no sympy involved. The
triple-product tensors (SS1.4) and subdivision/transfer matrices (SS1.5)
have no simple closed form and are generated via sympy once, lazily and
cached (functools.lru_cache) -- not needed until nonlinear (Burgers) /
refinement work (chapter 3 phase 2+), so this stays out of the fast path.

verify_all() is the "import-time assert" from SS1.7: every entry is
cross-checked against an *independent* re-implementation (plain numpy +
Gauss-Legendre quadrature, not reusing the torch recurrence below) so
agreement is a real check, not the same bug reflected twice. Deliberately
NOT run at bare `import` time (that stays instant) -- call it explicitly,
e.g. from notebooks/ch3/sc3.0/verify_ref_tensors.py.
"""
import functools

import numpy as np
import torch

DTYPE = torch.float64


# ── closed-form reference matrices (SS1.1 - 1.3, 1.6) ────────────────────────

def mass_matrix(N):
    """M_ij = int P_i P_j dxi = 2/(2i+1) on the diagonal, 0 off it (SS1.1)."""
    i = torch.arange(N, dtype=DTYPE)
    return torch.diag(2.0 / (2.0 * i + 1.0))


def stiffness_matrix(N):
    """K_ij = int P_i' P_j' dxi = min(i,j)(min(i,j)+1) iff i+j even (SS1.3)."""
    ii, jj = torch.meshgrid(torch.arange(N, dtype=DTYPE), torch.arange(N, dtype=DTYPE), indexing='ij')
    m = torch.minimum(ii, jj)
    even = (ii + jj) % 2 == 0
    return torch.where(even, m * (m + 1.0), torch.zeros_like(m))


def convection_matrix(N):
    """C_ij = int P_i P_j' dxi = 2 iff j>i and i+j odd, else 0 (SS1.2)."""
    ii, jj = torch.meshgrid(torch.arange(N, dtype=DTYPE), torch.arange(N, dtype=DTYPE), indexing='ij')
    odd = (ii + jj) % 2 == 1
    cond = odd & (jj > ii)
    return torch.where(cond, torch.full_like(ii, 2.0), torch.zeros_like(ii))


def endpoint_values(N):
    """(left, right): P_j(-1) = (-1)^j, P_j(1) = 1 (SS1.6)."""
    j = torch.arange(N, dtype=DTYPE)
    return (-1.0) ** j, torch.ones(N, dtype=DTYPE)


# ── canonical torch Legendre basis (used operationally by kan.py / solve.py) ─

def legendre_basis(xi, N):
    """(..., N) tensor of P_0..P_{N-1}(xi) via the 3-term recurrence."""
    xi = torch.as_tensor(xi, dtype=DTYPE)
    polys = [torch.ones_like(xi)]
    if N > 1:
        polys.append(xi.clone())
    for n in range(1, N - 1):
        polys.append(((2 * n + 1) * xi * polys[-1] - n * polys[-2]) / (n + 1))
    return torch.stack(polys, dim=-1)


def legendre_and_deriv(xi, N):
    """(P, dP): P_0..P_{N-1} and their xi-derivatives at xi, both (..., N).
    dP via SS1.2: P_j'(xi) = sum_{m<j, m+j odd} (2m+1) P_m(xi)."""
    P = legendre_basis(xi, N)
    dP = torch.zeros_like(P)
    for j in range(N):
        ms = [m for m in range(j) if (m + j) % 2 == 1]
        if ms:
            dP[..., j] = sum((2 * m + 1) * P[..., m] for m in ms)
    return P, dP


# ── triple-product tensors (SS1.4) -- sympy wigner_3j, lazy + cached ─────────

@functools.lru_cache(maxsize=None)
def _triple_matrix_np(N):
    from sympy.physics.wigner import wigner_3j
    T = np.zeros((N, N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if (i + j + k) % 2 != 0:
                    continue  # wigner_3j(...;0,0,0) vanishes unless i+j+k even
                if k < abs(i - j) or k > i + j:
                    continue  # triangle inequality
                T[i, j, k] = 2.0 * float(wigner_3j(i, j, k, 0, 0, 0)) ** 2
    return T


def triple_matrix(N):
    """T_ijk = int P_i P_j P_k dxi = 2*wigner_3j(i,j,k;0,0,0)^2 (SS1.4).
    No closed form -- generated via sympy once per N, cached."""
    return torch.from_numpy(_triple_matrix_np(N)).to(DTYPE)


@functools.lru_cache(maxsize=None)
def _deriv_triple_matrix_np(N):
    T = _triple_matrix_np(N)
    D = np.zeros((N, N, N), dtype=np.float64)
    for k in range(N):
        ms = [m for m in range(k) if (m + k) % 2 == 1]
        if not ms:
            continue
        coeffs = np.array([2 * m + 1 for m in ms], dtype=np.float64)
        D[:, :, k] = np.tensordot(T[:, :, ms], coeffs, axes=([2], [0]))
    return D


def deriv_triple_matrix(N):
    """D_ijk = int P_i P_j P_k' dxi = sum_{m<k, m+k odd} (2m+1) T_ijm (SS1.4)."""
    return torch.from_numpy(_deriv_triple_matrix_np(N)).to(DTYPE)


# ── subdivision / transfer matrices (SS1.5) -- sympy, lazy + cached ──────────

@functools.lru_cache(maxsize=None)
def _transfer_matrix_np(N, side):
    import sympy as sp
    s = sp.symbols('s')
    P = [sp.legendre(i, s) for i in range(N)]
    sub = (s + side) / 2
    Tm = np.zeros((N, N), dtype=np.float64)
    for j in range(N):
        for i in range(N):
            expr = sp.expand(sp.legendre(i, sub) * P[j])
            val = sp.Rational(2 * j + 1, 2) * sp.integrate(expr, (s, -1, 1))
            Tm[j, i] = float(val)
    return Tm


def transfer_matrices(N, side):
    """Exact coefficient transfer under h-bisection (SS1.5): side=-1 is the
    left child (T^L), side=+1 the right child (T^R). No closed form beyond
    this integral -- sympy, cached per (N, side)."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 (left child) or +1 (right child)")
    return torch.from_numpy(_transfer_matrix_np(N, side)).to(DTYPE)


# ── independent numpy verification path (deliberately NOT sharing code with ─
#    legendre_basis/legendre_and_deriv above -- see module docstring) ────────

def _legendre_vals_np(x, N):
    x = np.asarray(x, dtype=np.float64)
    P = np.empty((len(x), N), dtype=np.float64)
    P[:, 0] = 1.0
    if N > 1:
        P[:, 1] = x
    for n in range(1, N - 1):
        P[:, n + 1] = ((2 * n + 1) * x * P[:, n] - n * P[:, n - 1]) / (n + 1)
    return P


def _legendre_deriv_vals_np(P, N):
    dP = np.zeros_like(P)
    for j in range(N):
        ms = [m for m in range(j) if (m + j) % 2 == 1]
        if ms:
            dP[:, j] = sum((2 * m + 1) * P[:, m] for m in ms)
    return dP


def verify_all(N=20, N_expensive=12, atol=1e-9):
    """Cross-check every reference tensor against independent Gauss-Legendre
    quadrature (SS1.7's "import-time assert"). Raises AssertionError on
    mismatch, returns True otherwise. M/K/C/endpoints are checked at N (fast,
    closed-form); the sympy-only T/D/T^L/T^R are checked at the smaller
    N_expensive (not needed until phase 2+, kept cheap)."""
    n_quad = N + 10
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    P = _legendre_vals_np(xq, N)
    dP = _legendre_deriv_vals_np(P, N)

    M_q = np.einsum('qi,qj,q->ij', P, P, wq)
    K_q = np.einsum('qi,qj,q->ij', dP, dP, wq)
    C_q = np.einsum('qi,qj,q->ij', P, dP, wq)

    assert np.allclose(mass_matrix(N).numpy(), M_q, atol=atol), "mass_matrix mismatch vs quadrature"
    assert np.allclose(stiffness_matrix(N).numpy(), K_q, atol=atol), "stiffness_matrix mismatch vs quadrature"
    assert np.allclose(convection_matrix(N).numpy(), C_q, atol=atol), "convection_matrix mismatch vs quadrature"

    left, right = endpoint_values(N)
    j = np.arange(N)
    assert np.allclose(left.numpy(), (-1.0) ** j), "endpoint_values(left) mismatch"
    assert np.allclose(right.numpy(), np.ones(N)), "endpoint_values(right) mismatch"

    # T, D: integrand degree up to 3*(N_expensive-1) -> need a higher quad order
    n_quad2 = 3 * N_expensive + 10
    xq2, wq2 = np.polynomial.legendre.leggauss(n_quad2)
    P2 = _legendre_vals_np(xq2, N_expensive)
    dP2 = _legendre_deriv_vals_np(P2, N_expensive)

    T_q = np.einsum('qi,qj,qk,q->ijk', P2, P2, P2, wq2)
    D_q = np.einsum('qi,qj,qk,q->ijk', P2, P2, dP2, wq2)

    assert np.allclose(triple_matrix(N_expensive).numpy(), T_q, atol=atol), "triple_matrix mismatch vs quadrature"
    assert np.allclose(deriv_triple_matrix(N_expensive).numpy(), D_q, atol=atol), "deriv_triple_matrix mismatch vs quadrature"

    # Transfer matrices: (T^side)_ji = (2j+1)/2 * int P_i((s+side)/2) P_j(s) ds
    scale = (2 * np.arange(N_expensive) + 1) / 2.0
    for side in (-1, 1):
        sub = (xq2 + side) / 2.0
        Pi_sub = _legendre_vals_np(sub, N_expensive)     # (n_quad2, N_expensive) -- P_i(sub)
        Pj_s = _legendre_vals_np(xq2, N_expensive)        # (n_quad2, N_expensive) -- P_j(s)
        Tm_q = np.einsum('qi,qj,q->ji', Pi_sub, Pj_s, wq2) * scale[:, None]
        assert np.allclose(transfer_matrices(N_expensive, side).numpy(), Tm_q, atol=atol), \
            f"transfer_matrix (side={side}) mismatch vs quadrature"

    return True
