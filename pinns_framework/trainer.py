import torch
import torch.nn.functional as F
from copy import deepcopy

from .architecture import PINArchitecture


def _auto_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PINTrainer:
    """
    Trains a physics-informed neural network under one of two schemes.

    scheme : "TD-PINNS" | "VANILLA"
    ─────────────────────────────────────────────────────────────────
    "TD-PINNS"
        Time-Decomposed PINNs.  One network is trained per time step.
        Time is NOT an input to the network — it is handled implicitly
        by the stepping loop.  The residual function receives
        (u, u_prev, x, dt, **pde_kwargs).

        time_integrator options
            "BE"   — Backward Euler  (O(Δt),  implemented)
            "CN"   — Crank–Nicolson  (O(Δt²), placeholder)
            "BDF2" — BDF-2           (O(Δt²), placeholder)

    "VANILLA"
        Raissi et al. single-network approach.  The network takes
        (t, x) as input — t is prepended automatically by the trainer.
        Make sure layers[0] == n_spatial_dims + 1.
        The residual function receives (u, t, x, **pde_kwargs).

    Parameters
    ----------
    net              : nn.Module returned by PINArchitecture
    scheme           : "TD-PINNS" or "VANILLA"
    loss_fn          : PINLossFunction instance
    spatial_dims     : list[PINSpatialDimension]
    epochs           : max optimisation steps per time step (or total for VANILLA)
    dt               : time step size (TD-PINNs only)
    time_integrator  : "BE" | "CN" | "BDF2" (TD-PINNs only)
    lr               : initial learning rate for AdamW (used as max_lr when
                       max_lr is not given — kept for backward compatibility)
    max_lr           : peak/initial lr the optimiser starts from (defaults to lr)
    min_lr           : lower bound the ReduceLROnPlateau scheduler decays toward
    tolerance        : early-stopping threshold on the loss
    ic_weight        : weight on the initial-condition MSE term
    bc_weight        : weight on the boundary-condition MSE term
    pde_weight       : weight on the PDE residual MSE term
    device           : torch.device (auto-detected if None)
    verbose          : print training progress
    log_every        : print frequency (epochs); also the default sampling
                       interval for loss_history logging
    track_best       : if True, each _run_epochs call (one per TD-PINNs step,
                       or the one VANILLA run) keeps the network weights from
                       whichever epoch had the lowest training loss, and
                       restores them at the end instead of leaving the
                       final-epoch weights in place. Off by default (backward
                       compatible) since the extra state_dict copy has a cost.
    error_fn         : optional callable(net, t) -> float, evaluated every
                       error_every epochs (and logged into self.loss_history
                       alongside the loss) to track a ground-truth error
                       metric independent of the training loss. For TD-PINNs,
                       t is that step's fixed physical time (0.0 for the IC
                       step); for VANILLA, t is always None since one network
                       covers every t — aggregate internally if needed.
    error_every      : epoch interval for error_fn evaluation; defaults to
                       log_every when not given.
    ic_epochs        : TD-PINNs only — epoch budget for step 0 (the initial-
                       condition fit), which is a much easier problem than a
                       PDE step and typically needs far fewer epochs. Defaults
                       to `epochs` (backward compatible: same budget for every
                       stage). Ignored for VANILLA.
    time_colloc_nodes : VANILLA only — if set (an int), PDE collocation is
                       sampled as a genuine tensor-product batch: this many
                       fixed time nodes evenly spaced over [0, T] (via
                       torch.linspace, inclusive of both ends), each paired
                       with its own independent n_colloc-point spatial sample.
                       Needed by loss forms that require several points
                       sharing the exact same t to do a real (non-Monte-Carlo)
                       spatial quadrature per time slice — e.g. a weak-form
                       'agnostic' residual_fn that groups collocation points
                       by torch.unique(t). Left at the default None (backward
                       compatible), t_col is one independent torch.rand(0,T)
                       draw per point, as before — fine for strong/
                       weak_projection forms, but every point then has its own
                       essentially-unique t, which breaks per-slice quadrature.
    """

    def __init__(
        self,
        net,
        scheme,
        loss_fn,
        spatial_dims,
        epochs=2000,
        dt=None,
        time_integrator="BE",
        lr=1e-2,
        max_lr=None,
        min_lr=1e-8,
        tolerance=1e-4,
        ic_weight=10.0,
        bc_weight=10.0,
        pde_weight=0.1,
        device=None,
        verbose=True,
        log_every=500,
        collocation_fn=None,
        track_best=False,
        error_fn=None,
        error_every=None,
        ic_epochs=None,
        time_colloc_nodes=None,
    ):
        self.net = net
        self.scheme = scheme.upper().replace("-", "_").replace(" ", "_")
        self.loss_fn = loss_fn
        self.spatial_dims = spatial_dims
        self.epochs = epochs
        self.ic_epochs = ic_epochs if ic_epochs is not None else epochs
        self.dt = dt
        self.time_integrator = time_integrator.upper()
        self.lr = lr
        # Learning-rate bounds for the ReduceLROnPlateau scheduler:
        #   max_lr — initial (peak) lr the optimiser starts from; falls back to
        #            `lr` when not given (backward compatible).
        #   min_lr — floor the scheduler will not decay below.
        self.max_lr = max_lr if max_lr is not None else lr
        self.min_lr = min_lr
        self.tolerance = tolerance
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.pde_weight = pde_weight
        self.device = device if device is not None else _auto_device()
        self.verbose = verbose
        self.log_every = log_every
        self.collocation_fn = collocation_fn
        self.track_best = track_best
        self.error_fn = error_fn
        self.error_every = error_every if error_every is not None else log_every
        self.time_colloc_nodes = time_colloc_nodes
        # Residual histories populated during TD-PINNs training; used by
        # collocation_last_residual and collocation_mean_residual.
        self._last_residual = None  # (x_ref, mag_abs) from final epoch of prev step
        self._mean_residual = None  # (x_ref, mean_mag_abs) averaged over prev step
        # Per-call convergence log, populated by _run_epochs (one entry per
        # IC fit / time step / vanilla run).  Purely diagnostic — reading it
        # never changes training.  Each entry:
        #   {"label", "epochs", "converged", "final_loss", "best_loss"}
        self.step_history = []
        # Per-epoch(-ish) time series, sampled every error_every epochs (and
        # at the final epoch of each stage): {"label", "epoch", "loss",
        # "error"}. "error" is only present when error_fn is set. Reset at
        # the start of every fit() call, same as step_history.
        self.loss_history = []

    # ──────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────

    def fit(self, x_ic, u_ic, x_bc, u_bc, T, **pde_kwargs):
        """
        Train the model.

        x_ic       : (N_ic, D)  spatial coords at t=0
        u_ic       : (N_ic, 1)  known solution at t=0
        x_bc       : (N_bc, D)  boundary spatial coords
        u_bc       : (N_bc, 1)  boundary values (assumed time-constant)
        T          : float       final simulation time
        **pde_kwargs             forwarded verbatim to residual_fn
                                 (e.g. nu=0.07, Re=100)

        Returns
        -------
        TD-PINNs : list of nn.Module, one per time level (index 0 = IC)
        VANILLA  : single nn.Module trained over the full (t,x) domain
        """
        dev = self.device
        x_ic = x_ic.to(dev)
        u_ic = u_ic.to(dev)
        x_bc = x_bc.to(dev)
        u_bc = u_bc.to(dev)

        self.step_history = []  # fresh log for this fit() call
        self.loss_history = []

        if self.scheme == "TD_PINNS":
            return self._fit_td(x_ic, u_ic, x_bc, u_bc, T, **pde_kwargs)
        if self.scheme == "VANILLA":
            return self._fit_vanilla(x_ic, u_ic, x_bc, u_bc, T, **pde_kwargs)
        raise ValueError(
            f"Unknown scheme '{self.scheme}'. Use 'TD-PINNS' or 'VANILLA'."
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _sample_colloc(self, n=None, net=None):
        """
        Sample collocation points and return an (N, D) leaf tensor with
        requires_grad=True.

        If collocation_fn is set, delegates to it (passing the current network
        and the residual histories from the previous step).  Otherwise falls
        back to independent uniform sampling per spatial dimension.
        """
        n = n or self.spatial_dims[0].n_colloc
        if self.collocation_fn is not None:
            pts = self.collocation_fn(
                self.spatial_dims, n, self.device,
                net=net,
                last_residual=self._last_residual,
                mean_residual=self._mean_residual,
            )
            return pts.detach().requires_grad_(True)
        pts = [dim.sample(n, device=self.device) for dim in self.spatial_dims]
        return torch.cat(pts, dim=1).requires_grad_(True)

    def _make_optim(self, net):
        opt = torch.optim.AdamW(net.parameters(), lr=self.max_lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=300, factor=0.5, min_lr=self.min_lr
        )
        return opt, sched

    def _run_epochs(self, opt, sched, loss_fn_callable, label="",
                    epoch_callback=None, net=None, error_cb=None, max_epochs=None):
        """
        Generic epoch loop with early stopping.

        epoch_callback : optional callable(epoch: int) invoked at every
                         log_every interval and at the final epoch (whether
                         by convergence or by reaching max epochs).  Used to
                         collect residual snapshots for adaptive collocation.
        net            : optional network being trained.  If it carries an
                         adaptive grid-refinement policy (a KANNet with
                         refine_every set), the grid is made finer whenever the
                         loss stalls — see the refinement block below.  When the
                         net has no such policy (e.g. an MLP) this is a no-op.
                         Also the network whose weights are snapshotted for
                         track_best.
        error_cb       : optional zero-arg callable() -> float, evaluated
                         every self.error_every epochs (and at the final
                         epoch) and recorded into self.loss_history alongside
                         the training loss. Independent of epoch_callback.
        max_epochs     : epoch budget for this call; defaults to self.epochs.
                         Lets callers (e.g. the TD-PINNs IC step, via
                         self.ic_epochs) use a different budget than the rest.
        """
        max_epochs = max_epochs if max_epochs is not None else self.epochs
        epochs_run = 0
        final_loss = float("nan")
        converged  = False

        best_loss  = float("inf")
        best_state = None

        # ── adaptive grid refinement (opt-in, KAN only) ─────────────────
        # Active only when the net advertises a refinement policy.  We track
        # the loss at the start of each refinement window; at the end of the
        # window we refine *only if* the relative improvement was too small
        # (the loss is not reducing fast enough), which is the efficient part —
        # no refinement is spent while the net is still making good progress.
        refine_on = bool(getattr(net, "refine_every", None)) and hasattr(net, "refine_grid")
        win_start = None       # loss at the start of the current window
        n_refines = 0

        log_now = False  # tracks whether the final loop iteration already logged

        for epoch in range(1, max_epochs + 1):
            opt.zero_grad()
            loss = loss_fn_callable()
            loss.backward()
            opt.step()
            sched.step(loss.detach())

            epochs_run = epoch
            final_loss = loss.item()

            if self.track_best and final_loss < best_loss:
                best_loss = final_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

            log_now = (epoch % self.log_every == 0)
            error_now = (epoch % self.error_every == 0)

            if self.verbose and log_now:
                print(f"  [{label}] epoch {epoch:5d} | loss {loss.item():.4e}")

            if error_now:
                entry = {"label": label, "epoch": epoch, "loss": final_loss}
                if error_cb is not None:
                    entry["error"] = error_cb()
                self.loss_history.append(entry)

            if epoch_callback is not None and log_now:
                epoch_callback(epoch)

            if refine_on:
                if win_start is None:
                    win_start = loss.item()
                elif epoch % net.refine_every == 0:
                    improved = (win_start - loss.item()) / (abs(win_start) + 1e-12)
                    if improved < net.refine_tol:
                        new_g = net.refine_grid(self.device)
                        if new_g is None:
                            refine_on = False          # cap reached — stop checking
                        else:
                            opt, sched = self._make_optim(net)  # new params
                            n_refines += 1
                            if self.verbose:
                                print(f"  [{label}] grid refined -> {new_g} "
                                      f"at epoch {epoch} "
                                      f"(window improvement {improved:+.1%} "
                                      f"< {net.refine_tol:.0%})")
                    win_start = loss.item()

            if loss.item() < self.tolerance:
                converged = True
                if self.verbose:
                    print(
                        f"  [{label}] converged at epoch {epoch}"
                        f" | loss {loss.item():.4e}"
                    )
                if not error_now:
                    entry = {"label": label, "epoch": epoch, "loss": final_loss}
                    if error_cb is not None:
                        entry["error"] = error_cb()
                    self.loss_history.append(entry)
                if epoch_callback is not None:
                    epoch_callback(epoch)
                break

        # Make sure the very last epoch is always represented in loss_history,
        # even when self.epochs isn't a multiple of error_every.
        if not converged and not error_now:
            entry = {"label": label, "epoch": epochs_run, "loss": final_loss}
            if error_cb is not None:
                entry["error"] = error_cb()
            self.loss_history.append(entry)

        if self.track_best and best_state is not None:
            net.load_state_dict(best_state)

        # Diagnostic record: how many epochs this call needed to reach the
        # tolerance (or the cap, if it never did).  Additive — no effect on
        # the trained networks.
        self.step_history.append({
            "label": label,
            "epochs": epochs_run,
            "converged": converged,
            "final_loss": final_loss,
            "best_loss": best_loss if self.track_best else None,
            "refinements": n_refines,
        })

    # ──────────────────────────────────────────────────────────────────
    # TD-PINNs
    # ──────────────────────────────────────────────────────────────────

    def _fit_td(self, x_ic, u_ic, x_bc, u_bc, T, **pde_kwargs):
        if self.dt is None:
            raise ValueError("TD-PINNS requires dt.")
        if self.time_integrator != "BE":
            raise NotImplementedError(
                f"Time integrator '{self.time_integrator}' is not yet implemented. "
                "Only 'BE' (Backward Euler) is available.\n"
                "CN and BDF2 are planned — for now, encode the time scheme "
                "directly in your residual_fn if needed."
            )

        n_steps = round(T / self.dt)
        trained_nets = []

        # ── step 0: fit initial condition ──────────────────────────────
        if self.verbose:
            print("=== TD-PINNs | step 0 — initial condition ===")
        net0 = deepcopy(self.net).to(self.device)
        opt, sched = self._make_optim(net0)

        def ic_loss():
            return (
                self.ic_weight * F.mse_loss(net0(x_ic), u_ic)
                + self.bc_weight * F.mse_loss(net0(x_bc), u_bc)
            )

        err_cb0 = (lambda _n=net0: self.error_fn(_n, 0.0)) if self.error_fn is not None else None
        self._run_epochs(opt, sched, ic_loss, label="step 0 / IC", net=net0,
                          error_cb=err_cb0, max_epochs=self.ic_epochs)
        trained_nets.append(net0)
        prev_net = net0

        # ── steps 1 … N_t ─────────────────────────────────────────────
        for step in range(1, n_steps + 1):
            if self.verbose:
                print(
                    f"=== TD-PINNs | step {step}/{n_steps}"
                    f"  (t = {step * self.dt:.4g}) ==="
                )
            net_n = deepcopy(self.net).to(self.device)
            opt, sched = self._make_optim(net_n)

            # ── Residual tracking for adaptive collocation ─────────────
            # A fixed uniform grid is evaluated at every log_every epoch so
            # that _last_residual and _mean_residual stay meaningful even when
            # the training collocation points change each epoch.
            _res_snapshots = []
            _res_x = None
            if self.collocation_fn is not None:
                n_res  = self.spatial_dims[0].n_colloc
                _res_x = torch.cat(
                    [d.sample(n_res, device=self.device) for d in self.spatial_dims],
                    dim=1,
                )  # (N, D) — no grad, reused across epochs

            def epoch_cb(epoch, _net=net_n, _prev=prev_net):
                if _res_x is None:
                    return
                x_ev = _res_x.clone().requires_grad_(True)
                with torch.enable_grad():
                    u   = _net(x_ev)
                    u_p = _prev(x_ev)   # graph kept (see pde_loss note)
                    res = self.loss_fn.residual_fn(
                        u=u, u_prev=u_p, x=x_ev, dt=self.dt, **pde_kwargs
                    )
                _res_snapshots.append(res.detach().abs())

            # Capture by default-arg to avoid the late-binding closure pitfall
            def pde_loss(nn_=net_n, pn_=prev_net):
                x = self._sample_colloc(net=nn_)
                u = nn_(x)
                # u_prev is the frozen previous-step output, evaluated on the
                # SAME grad-enabled x so its graph w.r.t. x is preserved.  This
                # is what lets the residual function (the "lf") take spatial
                # derivatives of the previous step and thereby choose the time
                # integrator itself (Backward Euler, Explicit Euler,
                # Crank-Nicolson, …) — the trainer stays integrator-agnostic.
                # Residuals that only need the value detach u_prev themselves;
                # those terms then contribute no gradient to the frozen net.
                u_prev = pn_(x)
                pde = self.loss_fn(u=u, u_prev=u_prev, x=x, dt=self.dt, **pde_kwargs)
                bc  = F.mse_loss(nn_(x_bc), u_bc)
                return self.pde_weight * pde + self.bc_weight * bc

            err_cb = (
                (lambda _n=net_n, _t=step * self.dt: self.error_fn(_n, _t))
                if self.error_fn is not None else None
            )
            self._run_epochs(
                opt, sched, pde_loss, label=f"step {step}",
                epoch_callback=epoch_cb if self.collocation_fn is not None else None,
                net=net_n, error_cb=err_cb,
            )

            # Store residual histories for the next step's collocation
            if _res_snapshots and _res_x is not None:
                self._last_residual = (_res_x, _res_snapshots[-1])
                self._mean_residual = (_res_x, torch.stack(_res_snapshots).mean(dim=0))

            trained_nets.append(net_n)
            prev_net = net_n

        return trained_nets

    # ──────────────────────────────────────────────────────────────────
    # Vanilla PINNs (Raissi et al.)
    # ──────────────────────────────────────────────────────────────────

    def _fit_vanilla(self, x_ic, u_ic, x_bc, u_bc, T, **pde_kwargs):
        """
        Single-network approach.  The network's first input dimension
        must be t, so layers[0] should equal n_spatial_dims + 1.

        The trainer prepends t to x automatically when evaluating the
        network.  The residual function receives (u, t, x, **pde_kwargs).
        """
        if self.verbose:
            print("=== Vanilla PINNs | training over full (t, x) domain ===")

        net = deepcopy(self.net).to(self.device)
        opt, sched = self._make_optim(net)
        n_col = self.spatial_dims[0].n_colloc

        def total_loss():
            # ── initial condition (t = 0) ──────────────────────────
            t_ic = torch.zeros(x_ic.shape[0], 1, device=self.device)
            ic = F.mse_loss(net(torch.cat([t_ic, x_ic], dim=1)), u_ic)

            # ── boundary condition (random t in [0, T]) ────────────
            t_bc = torch.rand(x_bc.shape[0], 1, device=self.device) * T
            bc = F.mse_loss(net(torch.cat([t_bc, x_bc], dim=1)), u_bc)

            # ── PDE collocation ────────────────────────────────────
            if self.time_colloc_nodes is not None:
                # Tensor-product batch: fixed time nodes x independent
                # n_colloc-point spatial samples per node (see
                # time_colloc_nodes in the class docstring).
                t_nodes = torch.linspace(0, T, self.time_colloc_nodes, device=self.device)
                x_parts, t_parts = [], []
                for tv in t_nodes:
                    x_parts.append(self._sample_colloc(n_col, net=net))
                    t_parts.append(torch.full((n_col, 1), float(tv), device=self.device))
                x_col = torch.cat(x_parts, dim=0)
                t_col = torch.cat(t_parts, dim=0).requires_grad_(True)
            else:
                t_col = (torch.rand(n_col, 1, device=self.device) * T).requires_grad_(True)
                x_col = self._sample_colloc(n_col, net=net)
            u = net(torch.cat([t_col, x_col], dim=1))
            pde = self.loss_fn(u=u, t=t_col, x=x_col, **pde_kwargs)

            return (
                self.ic_weight * ic
                + self.bc_weight * bc
                + self.pde_weight * pde
            )

        err_cb = (lambda _n=net: self.error_fn(_n, None)) if self.error_fn is not None else None
        self._run_epochs(opt, sched, total_loss, label="VANILLA", net=net, error_cb=err_cb)
        return [net]


# ── IO utilities ──────────────────────────────────────────────────────────────

def save_pinns(trained_nets, trainer, arch, params_arch, filepath):
    """
    Save trained networks to a .pt checkpoint.

    trained_nets : list[nn.Module] — always a list (TD-PINNs has N+1 elements,
                   Vanilla has exactly one)
    trainer      : PINTrainer instance (provides the scheme string)
    arch         : "MLP" or "KAN"
    params_arch  : dict forwarded to PINArchitecture (layers, activation, …)
    filepath     : destination path, e.g. "burgers_tdpinns.pt"
    """
    state_dicts = [net.state_dict() for net in trained_nets]
    torch.save(
        {
            "scheme":      trainer.scheme,
            "arch":        arch,
            "params_arch": params_arch,
            "state_dicts": state_dicts,
        },
        filepath,
    )
    scheme_label = "TD-PINNs" if trainer.scheme == "TD_PINNS" else "Vanilla"
    print(
        f"Saved {len(state_dicts)} network(s) "
        f"[{arch} / {scheme_label}] → '{filepath}'"
    )


def load_pinns(filepath, device=None):
    """
    Load any PINNs checkpoint → list[nn.Module].

    Scheme detection is automatic — the caller does not need to know whether the
    checkpoint was produced by TD-PINNs or Vanilla training:
      - TD-PINNs: returns one network per time step (index 0 = IC network)
      - Vanilla:  returns a one-element list containing the full (t, x) network

    Works for both MLP and KAN architectures.
    """
    ckpt = torch.load(filepath, map_location=device or "cpu", weights_only=False)
    nets = []
    for sd in ckpt["state_dicts"]:
        net = PINArchitecture(ckpt["arch"], ckpt["params_arch"])
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    scheme_label = "TD-PINNs" if ckpt["scheme"] == "TD_PINNS" else "Vanilla"
    print(
        f"Loaded {len(nets)} {scheme_label} [{ckpt['arch']}] "
        f"network(s) from '{filepath}'"
    )
    return nets
