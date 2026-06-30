import numpy as np
import pandas as pd


class BurgersFD1D:
    """
    1-D viscous Burgers equation solver using the explicit upwind-central scheme
    from CFD Python (Barba et al.), Step 4:

        u_i^{n+1} = u_i^n
                  - u_i^n * (dt/dx) * (u_i^n - u_{i-1}^n)   <- upwind convection
                  + nu * (dt/dx^2) * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)  <- central diffusion

    Periodic boundary conditions: u[0] = u[-1] at all times.

    Stability requirements (von Neumann / CFL):
        CFL  = max|u| * dt / dx  <= 1
        r    = nu * dt / dx^2   <= 0.5
    """

    def __init__(self, nu, x0, xf, nx, dt):
        self.nu  = nu
        self.dt  = dt
        self.nx  = nx
        self.x   = np.linspace(x0, xf, nx)
        self.dx  = self.x[1] - self.x[0]

    # ── Stability ──────────────────────────────────────────────────────────

    def cfl_number(self, u):
        return float(np.max(np.abs(u))) * self.dt / self.dx

    def diffusion_number(self):
        return self.nu * self.dt / self.dx ** 2

    def stability_report(self, u0):
        cfl = self.cfl_number(u0)
        r   = self.diffusion_number()
        return {
            'CFL':              round(cfl, 6),
            'CFL_limit':        1.0,
            'CFL_ok':           cfl <= 1.0,
            'diffusion_r':      round(r, 6),
            'diffusion_limit':  0.5,
            'diffusion_ok':     r <= 0.5,
            'scheme_stable':    cfl <= 1.0 and r <= 0.5,
            'dx':               round(self.dx, 6),
            'dt':               self.dt,
        }

    # ── Solver ─────────────────────────────────────────────────────────────

    def _step(self, u):
        un = u.copy()
        # Interior points
        u[1:-1] = (un[1:-1]
                   - un[1:-1] * (self.dt / self.dx) * (un[1:-1] - un[:-2])
                   + self.nu  * (self.dt / self.dx ** 2) * (un[2:] - 2 * un[1:-1] + un[:-2]))
        # Left boundary: periodic — left neighbour is un[-2] (= u[nx-2])
        u[0] = (un[0]
                - un[0] * (self.dt / self.dx) * (un[0] - un[-2])
                + self.nu * (self.dt / self.dx ** 2) * (un[1] - 2 * un[0] + un[-2]))
        u[-1] = u[0]   # enforce periodicity
        return u

    def solve_at_times(self, u0, times):
        """
        Advance from t=0 to each value in `times` (must be multiples of dt).
        Returns dict {t: u_array}.
        """
        times = sorted(t for t in times if t > 0)
        u         = u0.copy()
        t_current = 0.0
        results   = {0.0: u.copy()}
        for t_target in times:
            n_steps = max(1, round((t_target - t_current) / self.dt))
            for _ in range(n_steps):
                u = self._step(u)
            t_current += n_steps * self.dt
            results[t_target] = u.copy()
        return results


# ── Error metrics ──────────────────────────────────────────────────────────

def compute_errors(u_pred, u_exact):
    """Return dict with L1, L2, Linf, rel_L2 errors."""
    diff  = u_pred - u_exact
    l1    = float(np.mean(np.abs(diff)))
    l2    = float(np.sqrt(np.mean(diff ** 2)))
    linf  = float(np.max(np.abs(diff)))
    rel   = l2 / (float(np.sqrt(np.mean(u_exact ** 2))) + 1e-12)
    return {'L1': l1, 'L2': l2, 'Linf': linf, 'rel_L2': rel}


def _trapz(u, dx):
    fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    return fn(u, dx=dx)


def mass(u, dx):
    """Discrete integral ∫u dx (should be conserved for periodic Burgers)."""
    return float(_trapz(u, dx))


def energy(u, dx):
    """Discrete L2 norm squared ½∫u² dx (dissipates over time)."""
    return float(0.5 * _trapz(u ** 2, dx))


class BurgersFD_CN:
    """
    1-D viscous Burgers equation — Crank-Nicolson finite-difference solver.

    Time discretisation (O(dt^2, dx^2)):

        (u^{n+1} - u^n) / dt
        + 0.5 * [ u^{n+1} * du^{n+1}/dx  +  u^n * du^n/dx ]   <- convection (semi-implicit)
        - 0.5 * nu * [ d2u^{n+1}/dx2  +  d2u^n/dx2 ]           <- diffusion  (fully implicit)
        = 0

    The nonlinear convection term is linearised via **Picard iteration**:
        ũ_i^{(k+1)} ≈ u^n + (u_i^{n+1,(k)} - u^n)  =>  convection coeff ≈ ũ_i^{(k)}

    At each Picard step we solve the tridiagonal system A x = rhs
    with periodic corners.  Convergence is declared when
    max|u^{(k+1)} - u^{(k)}| < picard_tol  or  after max_picard_iter steps.

    Parameters
    ----------
    nu : float
    x0, xf : float      domain limits
    nx : int            number of grid points (including both endpoints)
    dt : float          time step
    picard_tol : float  convergence criterion for Picard loop (default 1e-8)
    max_picard_iter : int  safety cap (default 50)
    """

    def __init__(self, nu, x0, xf, nx, dt,
                 picard_tol=1e-8, max_picard_iter=50):
        self.nu              = nu
        self.dt              = dt
        self.nx              = nx
        self.picard_tol      = picard_tol
        self.max_picard_iter = max_picard_iter
        self.x   = np.linspace(x0, xf, nx)
        self.dx  = self.x[1] - self.x[0]

    # ── Stability ──────────────────────────────────────────────────────────

    def cfl_number(self, u):
        return float(np.max(np.abs(u))) * self.dt / self.dx

    def diffusion_number(self):
        return self.nu * self.dt / self.dx ** 2

    def stability_report(self, u0):
        cfl = self.cfl_number(u0)
        r   = self.diffusion_number()
        return {
            'CFL':              round(cfl, 6),
            'CFL_limit':        'unconditional (CN)',
            'CFL_ok':           True,
            'diffusion_r':      round(r, 6),
            'diffusion_limit':  'unconditional (CN)',
            'diffusion_ok':     True,
            'scheme_stable':    True,
            'dx':               round(self.dx, 6),
            'dt':               self.dt,
        }

    # ── Tridiagonal solver with periodic corners (Thomas + correction) ─────

    @staticmethod
    def _solve_periodic_tridiag(lower, diag, upper, rhs):
        """
        Solve A x = rhs where A is tridiagonal with corners
          A[0, N-1]   = lower[0]   (top-right corner)
          A[N-1, 0]   = upper[-1]  (bottom-left corner)
        using the Sherman-Morrison / cyclic reduction trick.
        """
        N = len(diag)
        # Augmented system: A_mod = A  with corners zeroed
        d  = diag.copy()
        l  = lower.copy()
        u  = upper.copy()
        b  = rhs.copy()

        # Corner values
        alpha = lower[0]       # A[0, N-1]
        beta  = upper[N - 1]   # A[N-1, 0]

        # Modify diagonal to absorb the rank-1 perturbation
        gamma      = -d[0]
        d[0]      -= gamma
        d[N - 1]  -= alpha * beta / gamma

        # Solve A_mod x = b  (standard tridiagonal)
        x = BurgersFD_CN._thomas(l, d, u, b)

        # Solve A_mod z = [gamma, 0, ..., 0, alpha]^T
        bz       = np.zeros(N)
        bz[0]    = gamma
        bz[N-1]  = alpha
        z = BurgersFD_CN._thomas(l, d, u, bz)

        # Sherman-Morrison correction
        factor = (x[0] + beta * x[N-1] / gamma) / (1.0 + z[0] + beta * z[N-1] / gamma)
        x -= factor * z
        return x

    @staticmethod
    def _thomas(lower, diag, upper, rhs):
        """Standard Thomas algorithm for tridiagonal systems."""
        N  = len(diag)
        c_ = upper.copy().astype(float)
        d_ = rhs.copy().astype(float)
        a_ = lower.copy().astype(float)
        w  = diag.copy().astype(float)

        # Forward sweep
        for i in range(1, N):
            m     = a_[i] / w[i - 1]
            w[i] -= m * c_[i - 1]
            d_[i]-= m * d_[i - 1]

        # Back substitution
        x = np.zeros(N)
        x[N - 1] = d_[N - 1] / w[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = (d_[i] - c_[i] * x[i + 1]) / w[i]
        return x

    # ── Time step ──────────────────────────────────────────────────────────

    def _step(self, u_n):
        N  = self.nx
        dx = self.dx
        dt = self.dt
        nu = self.nu
        r  = nu * dt / (2.0 * dx ** 2)   # diffusion coefficient (half-step)

        # Explicit RHS contribution from u^n
        #   rhs_i = u_i^n  -  (dt/2) * u_i^n * (u_i^n - u_{i-1}^n) / dx
        #                   +  r * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
        u = u_n.copy()
        # Central shift (periodic)
        u_p = np.roll(u, -1)   # u_{i+1}
        u_m = np.roll(u,  1)   # u_{i-1}

        rhs = (u
               - (dt / (2.0 * dx)) * u * (u - u_m)   # upwind convection at t^n
               + r * (u_p - 2.0 * u + u_m))           # diffusion at t^n

        # Picard iteration for implicit part
        u_new = u.copy()   # initial guess: u^n
        for _ in range(self.max_picard_iter):
            u_tilde = u_new.copy()
            u_tilde_p = np.roll(u_tilde, -1)

            # Convection coefficient at t^{n+1}: upwind with ũ
            c = dt / (2.0 * dx) * u_tilde   # shape (N,)

            # Tridiagonal coefficients
            diag  = 1.0 + c + 2.0 * r + np.zeros(N)
            lower = -(c + r) + np.zeros(N)   # A[i, i-1]
            upper = -r        + np.zeros(N)  # A[i, i+1]

            # Adjust RHS for convection term (explicit upwind: u_tilde * (u - u_{i-1}) / dx)
            # Full CN: implicit part already in matrix; rhs stays as computed above
            u_new_iter = self._solve_periodic_tridiag(lower, diag, upper, rhs)
            u_new_iter[-1] = u_new_iter[0]   # enforce periodicity

            if np.max(np.abs(u_new_iter - u_new)) < self.picard_tol:
                u_new = u_new_iter
                break
            u_new = u_new_iter

        u_new[-1] = u_new[0]
        return u_new

    def solve_at_times(self, u0, times):
        """
        Advance from t=0 to each value in `times` (should be multiples of dt).
        Returns dict {t: u_array}.
        """
        times     = sorted(t for t in times if t > 0)
        u         = u0.copy()
        t_current = 0.0
        results   = {0.0: u.copy()}
        for t_target in times:
            n_steps = max(1, round((t_target - t_current) / self.dt))
            for _ in range(n_steps):
                u = self._step(u)
            t_current += n_steps * self.dt
            results[t_target] = u.copy()
        return results


def build_comparison_table(pinn_sols, fd_sols, exact_func, times, x):
    """
    Build a DataFrame with per-time-step error metrics for PINN and FD.

    Parameters
    ----------
    pinn_sols  : dict {t: np.ndarray}  evaluated on x
    fd_sols    : dict {t: np.ndarray}
    exact_func : callable(t, x) -> np.ndarray
    times      : list of floats
    x          : np.ndarray  (common evaluation grid)
    """
    dx = x[1] - x[0]
    rows = []
    for t in times:
        u_ex = exact_func(t=t, x=x)
        pe   = compute_errors(pinn_sols[t], u_ex) if t in pinn_sols else {}
        fe   = compute_errors(fd_sols[t],   u_ex) if t in fd_sols   else {}
        rows.append({
            't':               round(t, 6),
            'PINN_L2':         pe.get('L2',     np.nan),
            'FD_L2':           fe.get('L2',     np.nan),
            'PINN_Linf':       pe.get('Linf',   np.nan),
            'FD_Linf':         fe.get('Linf',   np.nan),
            'PINN_rel_L2':     pe.get('rel_L2', np.nan),
            'FD_rel_L2':       fe.get('rel_L2', np.nan),
            'PINN_mass':       mass(pinn_sols[t], dx) if t in pinn_sols else np.nan,
            'FD_mass':         mass(fd_sols[t],   dx) if t in fd_sols   else np.nan,
            'exact_mass':      mass(u_ex,          dx),
            'PINN_energy':     energy(pinn_sols[t], dx) if t in pinn_sols else np.nan,
            'FD_energy':       energy(fd_sols[t],   dx) if t in fd_sols   else np.nan,
            'exact_energy':    energy(u_ex,          dx),
        })
    return pd.DataFrame(rows).set_index('t')
