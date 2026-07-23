"""
Adaptive h-refinement (design doc SS4) for PGDr-PIKAN (chapter 3, phase 2b):
train/solve -> indicate -> mark -> bisect -> transfer -> resume.

Adapted for a direct (non-training) linear solve rather than the doc's
iterative-training framing ("free -- these are the loss terms already
computed"): the indicator here is a standard hierarchical a posteriori
estimator instead -- after solving at degree N (the weak residual against
test modes 0..N-1 is exactly zero by construction of Galerkin), evaluate
the SAME weak-form residual (mesh.residual, volume + face terms together)
against a handful of enriched test modes N..N+enrich-1. That's exactly
where the leftover error lives, and it costs no new solve.

Bisection/transfer reuses ref_tensors.transfer_matrices (exactness already
confirmed in sc3.0) and needs no non-conforming-interface handling: in 1D,
bisecting a cell only ever introduces a new INTERIOR face at the midpoint,
never disturbing its neighbors' faces -- see mesh.py's module docstring.
"""
import torch

from .ref_tensors import DTYPE, transfer_matrices
from .mesh import Mesh1D, residual as mesh_residual


def compute_indicators(mesh, c_per_cell, N, a, nu, f_fn, g_left, g_right,
                        enrich=4, penalty_coeff=10.0, n_quad=200):
    """Per-cell error indicator eta_e (doc SS4: eta_e =
    sqrt(sum_j r_{e,j}^2 / ||P_j||^2)), reinterpreted for a direct solve as
    the weak-form residual (volume + face terms, via mesh.residual) tested
    against the `enrich` modes just beyond what was actually solved for
    (j = N..N+enrich-1) -- these are exactly zero for j < N by Galerkin
    orthogonality, so this is free of the trivial part of the residual and
    isolates the genuine unresolved error.

    c_per_cell : (n_cells, N) already-solved Legendre coefficients.
    Returns   : (n_cells,) tensor of eta_e.
    """
    n_cells = mesh.n_cells
    N_hi = N + enrich

    c_padded = torch.zeros(n_cells, N_hi, dtype=DTYPE)
    c_padded[:, :N] = c_per_cell
    R_hi = mesh_residual(c_padded.reshape(-1), mesh, N_hi, a, nu, f_fn, g_left, g_right,
                          penalty_coeff, n_quad).view(n_cells, N_hi)
    extra = R_hi[:, N:]  # (n_cells, enrich) -- residual against the enriched-only modes

    j = torch.arange(N, N_hi, dtype=DTYPE)
    norm_j = 2.0 / (2.0 * j + 1.0)  # ||P_j||^2, doc SS1.1
    return torch.sqrt((extra ** 2 / norm_j).sum(dim=1))


def dorfler_mark(etas, theta=0.5):
    """Doerfler/bulk-chasing marking (doc SS4): the smallest set of cells
    whose eta_e^2 sum to at least a `theta` fraction of the total.
    Returns a sorted list of marked cell indices.
    """
    eta2 = etas ** 2
    total = float(eta2.sum())
    order = torch.argsort(eta2, descending=True).tolist()
    marked, cum = [], 0.0
    for idx in order:
        marked.append(idx)
        cum += float(eta2[idx])
        if total <= 0.0 or cum >= theta * total:
            break
    return sorted(marked)


def refine_mesh(mesh, c_per_cell, N, marked):
    """Bisect every cell in `marked` (doc SS4's bisect+transfer): h -> h/2,
    child coefficients via T^L/T^R (ref_tensors.transfer_matrices, exact --
    sc3.0's transfer-exactness check). Unmarked cells are copied unchanged.

    Note: for this direct-solve setting the transfer is NOT required for
    correctness (a direct linear solve has no iterative state to warm-start)
    -- it's used here to make the old solution's exact restriction onto the
    new mesh available as a diagnostic before the caller re-solves. It
    becomes genuinely load-bearing once Newton warm-starts matter (phase 2c,
    nonlinear).

    Returns (new_mesh, new_c_per_cell).
    """
    TL = transfer_matrices(N, side=-1)
    TR = transfer_matrices(N, side=1)
    marked = set(marked)

    new_edges = [mesh.edges[0]]
    new_c = []
    for e in range(mesh.n_cells):
        lo_e, hi_e = mesh.cell(e)
        if e in marked:
            mid = 0.5 * (lo_e + hi_e)
            new_edges.append(mid)
            new_edges.append(hi_e)
            new_c.append(TL @ c_per_cell[e])
            new_c.append(TR @ c_per_cell[e])
        else:
            new_edges.append(hi_e)
            new_c.append(c_per_cell[e])

    return Mesh1D(new_edges), torch.stack(new_c)
