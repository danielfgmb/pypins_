"""
1D multi-cell mesh for PGDr-PIKAN (chapter 3, phase 2) and its global
assembly for a steady linear advection-diffusion problem.

Mesh1D is deliberately just a general (possibly non-uniform) sorted list of
cell edges -- phase 2a's "uniform, fixed" mesh is just its trivial initial
case. Adaptive refinement (phase 2b, refine.py) replaces individual cells
with their two children (exact via ref_tensors.transfer_matrices) without
ever needing a different representation: in 1D, bisecting a cell only ever
introduces a NEW INTERIOR face at the midpoint -- the cell's two outer
faces (shared with its neighbors) don't move, so every interface stays a
clean, conforming one-cell-to-one-cell interface regardless of refinement
history. (The "non-conforming/hanging node" complication from adaptive
2D/3D FEM simply doesn't arise here.)

assemble_advection_diffusion is the one place design doc SS2.5's flagged
"sole unverified analytic component" -- the interface flux F_hat -- lives:
standard DG symmetric interior penalty (SIPG) for the viscous term, upwind
for the (linear) advective term, both applied through a single per-face
residual contribution. Boundary conditions are imposed weakly via a ghost
state fixed at the prescribed Dirichlet value, using the SAME flux/penalty
machinery as interior faces (simpler than phase 1's exact recombination
trick, which only works for a single cell's own physical boundary).

Implementation choice: the per-mode residual R(c) of the whole mesh is
written out directly and explicitly (one clear formula per volume/face
term, matching the weak form term-by-term), and the (affine) system's
matrix A and right-hand side b are recovered via A = jacobian(R), b =
-R(0) (torch.autograd.functional.jacobian) -- rather than hand-deriving and
transcribing every term into explicit outer-product matrix updates. Given
how easy interface/SIPG sign errors are to make, this trades a bit of
performance for removing an entire class of transcription bugs: the only
thing that has to be right is the residual formula itself, which can be
checked term-by-term against the weak form, not an error-prone hand-built
Jacobian.
"""
import numpy as np
import torch

from .ref_tensors import DTYPE, stiffness_matrix, convection_matrix, endpoint_values, legendre_and_deriv, legendre_basis
from .assembly import jacobian_scale, upwind_trace, sipg_penalty


class Mesh1D:
    """A 1D mesh: a sorted list of `n_cells+1` cell edges."""

    def __init__(self, edges):
        edges = list(float(e) for e in edges)
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("edges must be strictly increasing")
        self.edges = edges

    @property
    def n_cells(self):
        return len(self.edges) - 1

    def cell(self, e):
        """(lo, hi) physical bounds of cell e."""
        return self.edges[e], self.edges[e + 1]

    def h(self, e):
        lo, hi = self.cell(e)
        return hi - lo

    @classmethod
    def uniform(cls, lo, hi, n_cells):
        return cls(np.linspace(lo, hi, n_cells + 1).tolist())


def residual(c_flat, mesh, N, a, nu, f_fn, g_left, g_right, penalty_coeff, n_quad):
    """Per-mode residual of the whole mesh for
    d/dx(a*u) - nu*u'' = f,  u(lo)=g_left, u(hi)=g_right,
    at the given (possibly non-affine-independent, i.e. requires_grad) c_flat.
    Zero for the exact discrete solution; affine in c_flat otherwise -- see
    module docstring for why the caller recovers A, b from this via autograd
    rather than reading them off directly.
    """
    n_cells = mesh.n_cells
    K = stiffness_matrix(N)
    C = convection_matrix(N)
    left_v, right_v = endpoint_values(N)
    xi_ends = torch.tensor([-1.0, 1.0], dtype=DTYPE)
    _, dP_ends = legendre_and_deriv(xi_ends, N)
    dleft_v, dright_v = dP_ends[0], dP_ends[1]

    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    xi_q = torch.as_tensor(xq, dtype=DTYPE)
    Pq = legendre_basis(xi_q, N)
    wq_t = torch.as_tensor(wq, dtype=DTYPE)

    def c_of(e):
        return c_flat[e * N:(e + 1) * N]

    R_parts = [None] * n_cells

    # -- volume terms, per cell (doc SS2.2's weak form, no time derivative) --
    for e in range(n_cells):
        lo_e, hi_e = mesh.cell(e)
        h_e = hi_e - lo_e
        ce = c_of(e)

        r = -a * (C.transpose(0, 1) @ ce)                          # -a * int u psi' dx
        r = r + nu * jacobian_scale('viscous', h_e) * (K @ ce)      # +nu * int u' psi' dx

        x_phys = 0.5 * (hi_e - lo_e) * xi_q + 0.5 * (hi_e + lo_e)
        w = wq_t * jacobian_scale('mass', h_e)
        f_vals = f_fn(x_phys)
        r = r - (Pq * (f_vals * w).unsqueeze(-1)).sum(dim=0)        # - int f psi dx

        R_parts[e] = r

    # -- interior faces --
    for i in range(1, n_cells):
        e_L, e_R = i - 1, i
        h_L, h_R = mesh.h(e_L), mesh.h(e_R)
        cL, cR = c_of(e_L), c_of(e_R)

        uL = right_v @ cL
        uR = left_v @ cR
        uxL = (2.0 / h_L) * (dright_v @ cL)
        uxR = (2.0 / h_R) * (dleft_v @ cR)
        jump_u = uL - uR
        avg_ux = 0.5 * (uxL + uxR)
        sigma = sipg_penalty(N, 0.5 * (h_L + h_R), penalty_coeff)
        u_hat = upwind_trace(a, uL, uR)

        # advective (upwind)
        R_parts[e_L] = R_parts[e_L] + a * u_hat * right_v
        R_parts[e_R] = R_parts[e_R] - a * u_hat * left_v

        # viscous (SIPG: consistency + symmetry + penalty)
        R_parts[e_L] = R_parts[e_L] + (
            -nu * avg_ux * right_v - nu * 0.5 * (dright_v * (2.0 / h_L)) * jump_u + sigma * jump_u * right_v
        )
        R_parts[e_R] = R_parts[e_R] + (
            nu * avg_ux * left_v - nu * 0.5 * (dleft_v * (2.0 / h_R)) * jump_u - sigma * jump_u * left_v
        )

    # -- domain boundary faces (ghost state = prescribed Dirichlet value) --
    h0 = mesh.h(0)
    c0 = c_of(0)
    u0 = left_v @ c0
    ux0 = (2.0 / h0) * (dleft_v @ c0)
    jump0 = g_left - u0
    sigma0 = sipg_penalty(N, h0, penalty_coeff)
    u_hat0 = g_left if a >= 0 else u0
    R_parts[0] = R_parts[0] + (-a * u_hat0 * left_v)
    R_parts[0] = R_parts[0] + (nu * ux0 * left_v - nu * (dleft_v * (2.0 / h0)) * jump0 - sigma0 * jump0 * left_v)

    hN = mesh.h(n_cells - 1)
    cN = c_of(n_cells - 1)
    uN = right_v @ cN
    uxN = (2.0 / hN) * (dright_v @ cN)
    jumpN = uN - g_right
    sigmaN = sipg_penalty(N, hN, penalty_coeff)
    u_hatN = uN if a >= 0 else g_right
    R_parts[-1] = R_parts[-1] + (a * u_hatN * right_v)
    R_parts[-1] = R_parts[-1] + (-nu * uxN * right_v - nu * (dright_v * (2.0 / hN)) * jumpN + sigmaN * jumpN * right_v)

    return torch.cat(R_parts)


def assemble_advection_diffusion(mesh, N, a, nu, f_fn, g_left, g_right, penalty_coeff=10.0, n_quad=200):
    """Global (A, b) for d/dx(a*u) - nu*u'' = f on `mesh`, u(lo)=g_left,
    u(hi)=g_right, via the residual/jacobian trick (see module docstring).
    A @ c_flat = b; c_flat.view(mesh.n_cells, N) recovers per-cell Legendre
    coefficients.
    """
    ndof = mesh.n_cells * N
    zeros = torch.zeros(ndof, dtype=DTYPE)

    def res(c):
        return residual(c, mesh, N, a, nu, f_fn, g_left, g_right, penalty_coeff, n_quad)

    b = -res(zeros)
    A = torch.autograd.functional.jacobian(res, zeros)
    return A, b
