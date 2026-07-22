import torch
from itertools import product as _iproduct


# ── Test-function helpers ─────────────────────────────────────────────────────

def _legendre_1d(xi, n_max):
    """
    Evaluate Legendre polynomials P_0 … P_{n_max-1} at xi (N,) → (N, n_max).
    Three-term recurrence:  P_{k+1} = ((2k+1) x P_k - k P_{k-1}) / (k+1)
    """
    N = xi.shape[0]
    if n_max == 0:
        return xi.new_empty(N, 0)
    polys = [torch.ones(N, device=xi.device, dtype=xi.dtype)]
    if n_max > 1:
        polys.append(xi.clone())
    for k in range(1, n_max - 1):
        p_next = ((2 * k + 1) * xi * polys[-1] - k * polys[-2]) / (k + 1)
        polys.append(p_next)
    return torch.stack(polys, dim=1)  # (N, n_max)


def _build_test_matrix(x, spatial_dims, n_test, multi_dim_mode="total_degree"):
    """
    Build the (N, K) matrix of column-normalised Legendre test function values
    at collocation points x (N, D).

    Each test function is a tensor product of 1-D Legendre polynomials:
        v_{k1,...,kD}(x) = P_{k1}(ξ1) · … · P_{kD}(ξD)
    where ξd = 2(xd − lo_d)/(hi_d − lo_d) − 1 maps to [-1, 1].

    multi_dim_mode : "full" | "total_degree"
        "full"         — all n_test^D tensor products (K = n_test^D)
        "total_degree" — only multi-indices with Σ kd ≤ n_test-1
                         (K = C(n_test-1+D, D) ≈ n_test^D / D!)
                         Standard choice for D ≥ 2; avoids the exponential blow-up.

    Returns a fully-detached (N, K) tensor; gradients do not flow through it.
    """
    D = len(spatial_dims)
    with torch.no_grad():
        # Evaluate 1-D Legendre banks for each dimension (normalised coords)
        legs = []
        for d, dim in enumerate(spatial_dims):
            lo, hi = dim.domain
            xi = (2.0 * (x[:, d] - lo) / (hi - lo) - 1.0).detach()
            legs.append(_legendre_1d(xi, n_test))  # (N, n_test)

        if D == 1:
            V = legs[0]  # (N, n_test)

        elif multi_dim_mode == "full":
            # Tensor product of all degree combinations: K = n_test^D
            V = legs[0]
            for d in range(1, D):
                V = (V.unsqueeze(2) * legs[d].unsqueeze(1)).reshape(x.shape[0], -1)

        else:  # total_degree (default for D >= 2)
            # Only include (k1,...,kD) with k1 + … + kD ≤ n_test - 1
            # K = C(n_test - 1 + D, D) — polynomial growth, not exponential
            max_deg = n_test - 1
            multi_indices = [
                idx for idx in _iproduct(range(n_test), repeat=D)
                if sum(idx) <= max_deg
            ]
            cols = []
            for idx in multi_indices:
                col = legs[0][:, idx[0]].clone()
                for d in range(1, D):
                    col = col * legs[d][:, idx[d]]
                cols.append(col)
            V = torch.stack(cols, dim=1)  # (N, K)

        # Column-wise L2 normalisation: each test function contributes equally
        norms = V.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return V / norms  # (N, K)


# ── Loss function ─────────────────────────────────────────────────────────────

class PINLossFunction:
    """
    Wraps the user-defined PDE residual function and turns it into a scalar loss.

    Parameters
    ----------
    spatial_dims : list[PINSpatialDimension]
        Spatial dimensions of the problem.

    residual_fn : callable
        Computes the PDE residual tensor (N, 1).  The trainer calls it with
        keyword arguments that depend on the training scheme.

    form : "strong" | "weak_projection" | "agnostic"
        "strong" (default) — minimises MSE of the point-wise residual:
            L = (1/N) Σ_i R(x_i)²

        "weak_projection" — Petrov–Galerkin projection (VPINN formulation):
            L = (1/K) Σ_k (∫_Ω R(u) v_k dx)²
              ≈ (1/K) Σ_k (domain_volume · (1/N) Σ_i R(x_i) v_k(x_i))²
            where {v_k} is a bank of Legendre polynomial test functions.
            The residual_fn signature is UNCHANGED — only the aggregation differs.
            NOTE: (1/N) Σ_i R(x_i) v_k(x_i) is a Monte Carlo estimate of
            E[R v_k] = (1/domain_volume) ∫_Ω R v_k dx, NOT of the integral
            itself — it is missing a factor of domain_volume = |Ω| (the
            measure of whatever is actually being averaged over: just the
            spatial domain for TD-PINNs, but |Ω|·T for VANILLA, since each
            collocation point there also carries an independent random t that
            gets marginalized out by the same average). domain_volume lets
            you supply that factor explicitly; it defaults to 1.0 (old
            behaviour — the average is used as-is, silently biased low by
            1/domain_volume when domain_volume > 1, or biased high when
            domain_volume < 1, relative to the true weak residual).

        "agnostic" — residual_fn already returns the finished scalar loss.
            PINLossFunction performs no reduction or projection at all; it
            just calls residual_fn(**kwargs) and returns the result as-is.
            Use this when you derive the weak form yourself (e.g. integration
            by parts to drop an order of derivative) and want full control
            over how the PDE loss is assembled.

    n_test : int  (default 10, only used when form="weak_projection")
        Number of Legendre basis functions per spatial dimension.

    multi_dim_mode : "total_degree" | "full"  (only used when form="weak_projection" and D ≥ 2)
        How to build multi-dimensional test functions from 1-D Legendre banks.

        "total_degree" (default) — include only multi-indices (k1,…,kD) with
            k1 + … + kD ≤ n_test-1.  K = C(n_test-1+D, D) — polynomial in n_test.
            Example: D=2, n_test=10 → K=55 instead of 100.
                     D=3, n_test=10 → K=220 instead of 1000.

        "full" — all tensor-product combinations.  K = n_test^D.
            Manageable for D=2 but grows exponentially for D ≥ 3.

    test_fn : callable or None  (only used when form="weak_projection")
        Custom test function matrix builder.  Signature:
            test_fn(x: Tensor (N, D), spatial_dims) -> Tensor (N, K)
        When provided, n_test and multi_dim_mode are ignored.

    domain_volume : float  (default 1.0, only used when form="weak_projection")
        Scale factor applied to the Monte Carlo average before squaring — see
        the "weak_projection" note above. Pass |Ω| (the product of each
        spatial dimension's length) for TD-PINNs, or |Ω|·T for VANILLA (T =
        the trainer's fit() time horizon), so the computed loss matches the
        true weak-residual magnitude instead of being off by 1/domain_volume.
        Left at the default 1.0, behaviour is unchanged from before this
        parameter existed.

    Residual function signature conventions
    ----------------------------------------
    TD-PINNs (Backward Euler):
        residual_fn(u, u_prev, x, dt, **pde_kwargs) -> Tensor (N, 1)

        u        : current network output at collocation points
        u_prev   : previous-step output, already detached
        x        : collocation points with requires_grad=True
        dt       : time step size
        **pde_kwargs : constants forwarded from trainer.fit()

    Vanilla PINNs (Raissi et al.):
        residual_fn(u, t, x, **pde_kwargs) -> Tensor (N, 1)

        u        : network output u(t, x)
        t        : time collocation points with requires_grad=True
        x        : spatial collocation points with requires_grad=True
        **pde_kwargs : constants

    When form="agnostic", residual_fn returns a scalar Tensor (the finished
    loss) instead of an (N, 1) residual — everything else about its inputs
    is unchanged.

    Example (1-D Burgers — same residual_fn works for strong and weak_projection)
    -------------------------------------------------------------------------
    def burgers_residual(u, u_prev, x, dt, nu=0.01):
        du_dx  = torch.autograd.grad(u.sum(),     x, create_graph=True)[0]
        du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
        return (u - u_prev) / dt + u * du_dx - nu * du_dxx

    # Strong form (default)
    loss_fn = PINLossFunction([x_dim], burgers_residual)

    # Weak projection — 10 Legendre test functions, no other changes
    loss_fn = PINLossFunction([x_dim], burgers_residual, form="weak_projection", n_test=10)

    # Agnostic — residual_fn computes and returns the finished scalar loss
    def burgers_weak_loss(u, u_prev, x, dt, nu=0.01):
        du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        # ... integrate by parts, build your own projection ...
        return loss  # scalar Tensor
    loss_fn = PINLossFunction([x_dim], burgers_weak_loss, form="agnostic")
    """

    def __init__(self, spatial_dims, residual_fn, form="strong",
                 n_test=10, multi_dim_mode="total_degree", test_fn=None,
                 domain_volume=1.0):
        self.spatial_dims    = spatial_dims
        self.residual_fn     = residual_fn
        if form == "weak":  # backward-compat alias for the renamed "weak_projection"
            form = "weak_projection"
        if form not in ("strong", "weak_projection", "agnostic"):
            raise ValueError("form must be 'strong', 'weak_projection', or 'agnostic'")
        if multi_dim_mode not in ("total_degree", "full"):
            raise ValueError("multi_dim_mode must be 'total_degree' or 'full'")
        self.form           = form
        self.n_test         = n_test
        self.multi_dim_mode = multi_dim_mode
        self.test_fn        = test_fn
        self.domain_volume  = domain_volume

    def __call__(self, **kwargs):
        """
        Compute the PDE loss scalar.  All kwargs are forwarded to residual_fn.
        The trainer is responsible for providing the correct arguments.
        """
        result = self.residual_fn(**kwargs)

        if self.form == "agnostic":
            # residual_fn already returns the finished scalar loss — no
            # reduction or projection to do here.
            return result

        residual = result  # (N, 1)

        if self.form == "strong":
            return (residual ** 2).mean()

        # ── weak_projection / Petrov–Galerkin form ────────────────────
        x = kwargs.get("x")
        if x is None:
            raise RuntimeError(
                "weak_projection form requires 'x' to be a keyword argument.  "
                "The trainer always supplies it; if calling the loss "
                "directly, pass x=<collocation_tensor>."
            )

        if self.test_fn is not None:
            V = self.test_fn(x, self.spatial_dims)          # (N, K)
        else:
            V = _build_test_matrix(
                x, self.spatial_dims, self.n_test, self.multi_dim_mode
            )  # (N, K)

        # Monte Carlo approximation:  ∫ R v_k dx ≈ domain_volume · (1/N) Σ_i R_i v_k(x_i)
        # residual (N, 1) * V (N, K)  →  (N, K)  →  mean over N  →  (K,)
        integrals = (residual * V).mean(dim=0) * self.domain_volume
        return (integrals ** 2).mean()

    def __repr__(self):
        base = (
            f"PINLossFunction(form={self.form!r}, "
            f"dims={[d.name for d in self.spatial_dims]}"
        )
        if self.form == "weak_projection":
            base += (f", n_test={self.n_test}, multi_dim_mode={self.multi_dim_mode!r}"
                      f", domain_volume={self.domain_volume!r}")
        return base + ")"
