"""
Thin, from-scratch PyTorch Legendre-KAN (chapter 3, design doc SS3) --
deliberately NOT built on pykan (architecture.py's KANNet): pykan's basis is
B-splines, and composed B-splines are not splines on the same knots, so the
analytic reference-tensor machinery in ref_tensors.py would not apply past
one layer (doc SS3.1's argument for Legendre over B-splines). Every edge here
is a plain Legendre expansion, so the whole network is exactly a polynomial
of degree <= p^L, exactly re-expandable in Legendre via project_to_legendre.
"""
import torch
import torch.nn as nn
import numpy as np

from .ref_tensors import DTYPE, legendre_basis


class LegendreEdge(nn.Module):
    """One scalar->scalar KAN edge: phi(x) = sum_i theta_i P_i(xi(x)), where
    xi is the affine map of x from `domain` to [-1, 1]. theta are the
    trainable coefficients -- doc SS3.2's theta (and, for a single-layer
    network, theta *is* c: see notebooks/ch3/sc3.1)."""

    def __init__(self, degree, domain=(-1.0, 1.0)):
        super().__init__()
        self.degree = degree
        self.domain = domain
        self.theta = nn.Parameter(torch.empty(degree, dtype=DTYPE))
        nn.init.normal_(self.theta, std=1.0 / degree)

    def _to_xi(self, x):
        lo, hi = self.domain
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def forward(self, x):
        xi = self._to_xi(x)
        P = legendre_basis(xi, self.degree)
        return (P * self.theta).sum(dim=-1)


class LegendreKANLayer(nn.Module):
    """width_in x width_out grid of LegendreEdges; output_j = sum_i edge_ij(x_i)
    -- standard KAN layer semantics."""

    def __init__(self, width_in, width_out, degree, domain=(-1.0, 1.0)):
        super().__init__()
        self.width_in, self.width_out = width_in, width_out
        self.edges = nn.ModuleList([
            LegendreEdge(degree, domain=domain)
            for _ in range(width_in * width_out)
        ])

    def forward(self, x):
        # x: (..., width_in) -> (..., width_out)
        out = []
        for j in range(self.width_out):
            acc = self.edges[j * self.width_in](x[..., 0])
            for i in range(1, self.width_in):
                acc = acc + self.edges[j * self.width_in + i](x[..., i])
            out.append(acc)
        return torch.stack(out, dim=-1)


class LegendreKAN(nn.Module):
    """Stack of LegendreKANLayers, `widths=(1, ..., 1)` convention matching
    architecture.py's `layers` tuple. Composed output is exactly a
    polynomial of degree <= degree^n_layers (doc SS3.1)."""

    def __init__(self, widths=(1, 1), degree=5, domain=(-1.0, 1.0)):
        super().__init__()
        self.widths = widths
        self.degree = degree
        self.n_layers = len(widths) - 1
        self.layers = nn.ModuleList([
            LegendreKANLayer(widths[l], widths[l + 1], degree, domain=domain)
            for l in range(self.n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def project_to_legendre(fn, domain, degree, n_quad=None):
    """Exact Legendre coefficients c of `fn` on `domain` (doc SS3.2's c(theta)
    map), via Gauss-Legendre quadrature + orthogonality. `fn` maps a
    (n_quad, 1) physical-x tensor -> (n_quad, 1) tensor (e.g. a LegendreKAN).
    Exact to float round-off whenever n_quad comfortably exceeds fn's
    polynomial degree on `domain` -- Gauss-Legendre of order n integrates
    polynomials of degree <= 2n-1 exactly."""
    if n_quad is None:
        n_quad = 4 * degree + 8
    lo, hi = domain
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    xi = torch.as_tensor(xq, dtype=DTYPE)
    x_phys = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
    with torch.no_grad():
        f_vals = fn(x_phys.unsqueeze(-1)).squeeze(-1)
    P = legendre_basis(xi, degree)  # (n_quad, degree)
    w = torch.as_tensor(wq, dtype=DTYPE)
    norm = 2.0 / (2.0 * torch.arange(degree, dtype=DTYPE) + 1.0)
    return (P * (f_vals * w).unsqueeze(-1)).sum(dim=0) / norm


def closure_residual(kan_net, domain, eval_degree=None, n_eval=2000):
    """Doc SS3.1's cheap validation test: evaluate `kan_net` on a fine grid,
    project to Legendre at degree = degree^n_layers (or `eval_degree` if
    given), reconstruct, and return the max |residual| between the raw
    network output and its Legendre reconstruction. Should be machine-zero
    for any Legendre-KAN however deep -- that IS the SS3.1 closure argument
    (polynomials closed under composition/products)."""
    if eval_degree is None:
        eval_degree = kan_net.degree ** kan_net.n_layers
    lo, hi = domain
    x = torch.linspace(lo, hi, n_eval, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        raw = kan_net(x).squeeze(-1)
    c = project_to_legendre(kan_net, domain, eval_degree)
    xi = 2.0 * (x.squeeze(-1) - lo) / (hi - lo) - 1.0
    P = legendre_basis(xi, eval_degree)
    recon = (P * c).sum(dim=-1)
    return float((raw - recon).abs().max())
