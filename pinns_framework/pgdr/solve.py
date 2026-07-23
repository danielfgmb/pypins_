"""
Direct linear/nonlinear solves for PGDr-PIKAN (chapter 3).

solve_poisson_single_cell / solve_burgers_single_cell: phase 1, a single
cell spanning the whole domain -- no interior interfaces, so design doc
SS2.5's flagged open risk (the interface flux) never comes up. One
torch.linalg.solve (Poisson) or Crank-Nicolson + Newton (Burgers), no
training loop.

solve_advection_diffusion_multi_cell: phase 2a, a genuine multi-cell mesh
(mesh.py) -- the first solve that actually needs SS2.5's interface flux
(upwind advective + SIPG viscous, mesh.assemble_advection_diffusion). Still
one global torch.linalg.solve (steady, linear problem) -- Newton-across-cells
for the nonlinear Burgers case is future work (phase 2c).
"""
import numpy as np
import torch

from .ref_tensors import DTYPE, mass_matrix, stiffness_matrix, deriv_triple_matrix, legendre_basis, legendre_and_deriv
from .assembly import jacobian_scale, dirichlet_recombination_basis, constant_dirichlet_lift
from .mesh import assemble_advection_diffusion
from .kan import project_to_legendre


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


def solve_burgers_single_cell(exact_fn, domain, nu, u_left, u_right, N, n_steps, T_end,
                               snapshot_times=None, newton_tol=1e-10, newton_max_iter=20,
                               n_eval=2000):
    """u_t + (u^2/2)_x = nu*u_xx on `domain`, u(lo,t)=u_left, u(hi,t)=u_right
    for all t (a CONSTANT-in-time Dirichlet BC -- valid here because the
    caller is expected to have checked the domain edges sit far enough from
    whatever moving feature exact_fn describes that the true boundary values
    are constant to float round-off; see notebooks/ch3/sc3.2/common.py for
    that check on the traveling-shock benchmark this was built for).

    Single cell (no interior interfaces -- see module docstring), one KAN
    layer (Bubnov-Galerkin, doc SS3.3's phase-1 caveat, same as
    solve_poisson_single_cell). Non-homogeneous Dirichlet BCs are handled by
    writing u = u_lift + w (assembly.constant_dirichlet_lift), with w's
    coefficients solved for in the SAME vanishing-at-both-endpoints
    recombined basis as the Poisson case -- testing against functions that
    vanish at the domain edges makes the weak form's boundary bracket term
    (doc SS2.2) drop out automatically, exactly as it did there; nothing
    here needs SS2.5's (still unbuilt) interface flux.

    Time-marched via Crank-Nicolson (doc SS2.4) with Newton's method solving
    the nonlinear system at each step (the flux term is quadratic in the
    unknown coefficients, via ref_tensors.deriv_triple_matrix -- the first
    real exercise of that tensor in an actual solve).

    Returns a dict: {t: {'c': (N,) coeffs, 'rel_L2': ..., 'Linf': ...}} for
    each requested snapshot time actually reached during marching, plus
    'newton_iters': list of per-step iteration counts (diagnostic).
    """
    lo, hi = domain
    h = hi - lo

    M = mass_matrix(N) * jacobian_scale('mass', h)
    Kv = nu * stiffness_matrix(N) * jacobian_scale('viscous', h)
    D = deriv_triple_matrix(N) * jacobian_scale('flux', h)  # flux scale is always 1 (SS2.3)

    B = dirichlet_recombination_basis(N)
    M_tilde = B.T @ M @ B
    Kv_tilde = B.T @ Kv @ B

    c_lift = constant_dirichlet_lift(N, u_left, u_right)
    k_lift = B.T @ (Kv @ c_lift)  # constant vector -- lift's own viscous contribution

    def flux_and_jac(a):
        c = c_lift + B @ a
        # F_t = (1/2) sum_{i,k} c_i c_k D[i,k,t]  (doc SS2.4's D(c,c)_j)
        F = 0.5 * torch.einsum('i,k,ikt->t', c, c, D)
        # dF_t/dc_m = sum_k D[m,k,t] c_k  (D symmetric in its first two indices)
        J_c = torch.einsum('mkt,k->tm', D, c)
        return B.T @ F, B.T @ J_c @ B

    # -- initial condition: Galerkin L2 projection of exact_fn(0, .) onto the
    # trial space, then solved so u(0,x) = u_lift(x) + w(0,x) matches it as
    # closely as possible while EXACTLY satisfying the boundary values (any a
    # keeps w vanishing at the boundary, by construction of B) --
    c_target0 = project_to_legendre(lambda x: exact_fn(0.0, x), domain, N)
    a = torch.linalg.solve(M_tilde, B.T @ (M @ (c_target0 - c_lift)))

    if snapshot_times is None:
        snapshot_times = [0.0, 0.25 * T_end, 0.5 * T_end, 0.75 * T_end, T_end]
    snapshot_times = sorted(snapshot_times)

    x_eval = torch.linspace(lo, hi, n_eval, dtype=DTYPE)
    xi_eval = 2.0 * (x_eval - lo) / (hi - lo) - 1.0
    P_eval = legendre_basis(xi_eval, N)

    def evaluate(t, a_vec):
        c = c_lift + B @ a_vec
        u_pred = (P_eval * c).sum(dim=-1)
        u_exact = exact_fn(t, x_eval)
        diff = u_pred - u_exact
        l2 = float(torch.sqrt((diff ** 2).mean()))
        denom = float(torch.sqrt((u_exact ** 2).mean())) + 1e-12
        return {'c': c, 'rel_L2': l2 / denom, 'Linf': float(diff.abs().max())}

    results = {}
    snap_idx = 0
    if snapshot_times and abs(snapshot_times[0]) < 1e-12:
        results[0.0] = evaluate(0.0, a)
        snap_idx = 1

    dt = T_end / n_steps
    t = 0.0
    newton_iters = []
    for _ in range(n_steps):
        F_now, _ = flux_and_jac(a)
        visc_now = Kv_tilde @ a + k_lift

        a_next = a.clone()
        for it in range(1, newton_max_iter + 1):
            F_next, J_next = flux_and_jac(a_next)
            visc_next = Kv_tilde @ a_next + k_lift

            G = (M_tilde @ (a_next - a)) / dt - 0.5 * ((F_next - visc_next) + (F_now - visc_now))
            dG = M_tilde / dt - 0.5 * (J_next - Kv_tilde)

            delta = torch.linalg.solve(dG, -G)
            a_next = a_next + delta
            if float(delta.abs().max()) < newton_tol:
                break
        newton_iters.append(it)

        a = a_next
        t += dt

        while snap_idx < len(snapshot_times) and t >= snapshot_times[snap_idx] - 1e-9:
            results[snapshot_times[snap_idx]] = evaluate(snapshot_times[snap_idx], a)
            snap_idx += 1

    return {'snapshots': results, 'newton_iters': newton_iters}


def solve_advection_diffusion_multi_cell(mesh, N, a, nu, f_fn, exact_fn, g_left, g_right,
                                          penalty_coeff=10.0, n_quad=200, n_eval=2000):
    """d/dx(a*u) - nu*u'' = f on `mesh` (mesh.Mesh1D), u(lo)=g_left,
    u(hi)=g_right -- design doc SS2, first solve that genuinely needs the
    interface flux F_hat (SS2.5), via mesh.assemble_advection_diffusion
    (upwind advective + SIPG viscous). One global torch.linalg.solve, still
    no training loop, same spirit as solve_poisson_single_cell.

    exact_fn(x) -- manufactured exact solution, used only to report
    rel_L2/Linf on a fine eval grid.

    Returns a dict: {'c': (n_cells, N) per-cell coefficients, 'rel_L2',
    'Linf'}.
    """
    A, b = assemble_advection_diffusion(mesh, N, a, nu, f_fn, g_left, g_right,
                                         penalty_coeff=penalty_coeff, n_quad=n_quad)
    c_flat = torch.linalg.solve(A, b)
    c = c_flat.view(mesh.n_cells, N)

    lo, hi = mesh.edges[0], mesh.edges[-1]
    x_eval = torch.linspace(lo, hi, n_eval, dtype=DTYPE)
    u_pred = torch.empty_like(x_eval)
    for e in range(mesh.n_cells):
        lo_e, hi_e = mesh.cell(e)
        mask = (x_eval >= lo_e) & (x_eval <= hi_e)
        xi_e = 2.0 * (x_eval[mask] - lo_e) / (hi_e - lo_e) - 1.0
        u_pred[mask] = (legendre_basis(xi_e, N) * c[e]).sum(dim=-1)
    u_exact = exact_fn(x_eval)

    diff = u_pred - u_exact
    l2 = float(torch.sqrt((diff ** 2).mean()))
    denom = float(torch.sqrt((u_exact ** 2).mean())) + 1e-12
    linf = float(diff.abs().max())

    return {'c': c, 'rel_L2': l2 / denom, 'Linf': linf}
