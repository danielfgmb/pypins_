"""
Collocation point sampling strategies for PINNs training.

All public strategies share the signature:

    strategy(spatial_dims, n, device, *, net=None, last_residual=None,
             mean_residual=None, adaptive_frac=0.5, **kwargs) -> Tensor (N, D)

The trainer calls the chosen strategy before each training step, injecting
the current network and the residual data stored from the previous step.
Strategies that do not need certain kwargs simply ignore them via **kwargs.
"""

import torch


# ── Internal helpers ──────────────────────────────────────────────────────────

def _uniform_pts(spatial_dims, n, device):
    """(N, D) uniformly sampled points, one column per spatial dimension."""
    cols = [
        torch.empty(n, 1, device=device).uniform_(d.domain[0], d.domain[1])
        for d in spatial_dims
    ]
    return torch.cat(cols, dim=1)


def _lhs_pts(spatial_dims, n, device):
    """
    (N, D) Latin Hypercube Sample.
    Each marginal is stratified into n equal intervals with one random point
    per interval; columns are independently shuffled.
    """
    D    = len(spatial_dims)
    cuts = torch.linspace(0.0, 1.0, n + 1, device=device)
    u    = torch.rand(n, D, device=device)
    s    = cuts[:-1].unsqueeze(1) + (cuts[1:] - cuts[:-1]).unsqueeze(1) * u
    for d in range(D):
        s[:, d] = s[torch.randperm(n, device=device), d]
    pts = torch.empty_like(s)
    for d, dim in enumerate(spatial_dims):
        lo, hi = dim.domain
        pts[:, d] = lo + s[:, d] * (hi - lo)
    return pts


def _net_grads(net, x):
    """
    Compute first- and second-order spatial derivatives of net(x) without
    leaving any gradient tracking behind.

    x       : (N, D) — does NOT need requires_grad
    Returns : (grad1, grad2) both (N, D), fully detached
      grad1[:, d] = ∂u/∂x_d
      grad2[:, d] = ∂²u/∂x_d²  (diagonal of the Hessian)
    """
    x_in = x.detach().requires_grad_(True)
    D    = x_in.shape[1]

    with torch.enable_grad():
        u = net(x_in)
        if u.dim() == 1:
            u = u.unsqueeze(1)

        grad1 = torch.autograd.grad(u.sum(), x_in, create_graph=True)[0]  # (N, D)

        grad2_cols = []
        for d in range(D):
            g2 = torch.autograd.grad(
                grad1[:, d].sum(), x_in,
                create_graph=False,
                retain_graph=(d < D - 1),
            )[0][:, d]
            grad2_cols.append(g2)

        grad2 = torch.stack(grad2_cols, dim=1)  # (N, D)

    return grad1.detach(), grad2.detach()


def _adaptive_select(spatial_dims, n, device, mag_fn, oversample=3, adaptive_frac=0.5):
    """
    Sample adaptive_frac*n*oversample candidate points, rank by mag_fn,
    keep the top adaptive_frac*n, fill the rest with uniform points.

    mag_fn : callable(x: Tensor (M, D)) -> Tensor (M,)
             higher value = more important region
    """
    n_a    = max(1, int(n * adaptive_frac))
    n_u    = n - n_a
    n_cand = n_a * oversample

    candidates = _uniform_pts(spatial_dims, n_cand, device)
    mag        = mag_fn(candidates).detach()

    idx          = torch.argsort(mag / (mag.sum() + 1e-8), descending=True)[:n_a]
    pts_adaptive = candidates[idx].detach()
    pts_uniform  = _uniform_pts(spatial_dims, n_u, device)

    return torch.cat([pts_adaptive, pts_uniform], dim=0)


def _residual_select(x_ref, mag, n, spatial_dims, device, adaptive_frac):
    """
    Select the top adaptive_frac*n points from x_ref by residual magnitude,
    fill the remaining slots with fresh uniform samples.

    x_ref : (M, D) — fixed evaluation grid from the previous step
    mag   : (M, 1) or (M,) — residual magnitude at those points
    """
    n_a      = max(1, int(n * adaptive_frac))
    n_u      = n - n_a
    mag_flat = mag.squeeze(-1).to(device)

    idx          = torch.argsort(mag_flat / (mag_flat.sum() + 1e-8), descending=True)[:n_a]
    pts_adaptive = x_ref[idx].to(device).detach()
    pts_uniform  = _uniform_pts(spatial_dims, n_u, device)

    return torch.cat([pts_adaptive, pts_uniform], dim=0)


# ── Public strategies ─────────────────────────────────────────────────────────

def collocation_uniform(spatial_dims, n, device, **kwargs):
    """Uniform random sampling over the full spatial domain."""
    return _uniform_pts(spatial_dims, n, device)


def collocation_lhs(spatial_dims, n, device, **kwargs):
    """Latin Hypercube Sampling — better marginal coverage than pure random."""
    return _lhs_pts(spatial_dims, n, device)


def collocation_gradient(spatial_dims, n, device, *, net, adaptive_frac=0.5, **kwargs):
    """
    Concentrate adaptive_frac of points where ||∇u||₂ is largest.
    Needs net to evaluate first-order spatial derivatives.
    """
    def mag_fn(x):
        g1, _ = _net_grads(net, x)
        return g1.norm(dim=1)

    return _adaptive_select(spatial_dims, n, device, mag_fn, adaptive_frac=adaptive_frac)


def collocation_curvature(spatial_dims, n, device, *, net, adaptive_frac=0.5, **kwargs):
    """
    Concentrate adaptive_frac of points where ∑_d |∂²u/∂x_d²| is largest.
    Needs net to evaluate second-order spatial derivatives.
    """
    def mag_fn(x):
        _, g2 = _net_grads(net, x)
        return g2.abs().sum(dim=1)

    return _adaptive_select(spatial_dims, n, device, mag_fn, adaptive_frac=adaptive_frac)


def collocation_last_residual(spatial_dims, n, device, *, last_residual=None,
                               adaptive_frac=0.5, **kwargs):
    """
    Concentrate adaptive_frac of points where the PDE residual was highest
    at the END of the previous training step (single final-epoch snapshot).
    Falls back to uniform when no history is available (first step).
    """
    if last_residual is None:
        return collocation_uniform(spatial_dims, n, device)
    x_ref, mag = last_residual
    return _residual_select(x_ref, mag, n, spatial_dims, device, adaptive_frac)


def collocation_mean_residual(spatial_dims, n, device, *, mean_residual=None,
                               adaptive_frac=0.5, **kwargs):
    """
    Concentrate adaptive_frac of points where the PDE residual magnitude was
    highest ON AVERAGE across all logged epochs of the previous training step.
    Smoother signal than last_residual; falls back to uniform at the first step.
    """
    if mean_residual is None:
        return collocation_uniform(spatial_dims, n, device)
    x_ref, mag = mean_residual
    return _residual_select(x_ref, mag, n, spatial_dims, device, adaptive_frac)
