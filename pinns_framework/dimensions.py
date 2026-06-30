import torch


class PINSpatialDimension:
    """
    Describes one spatial dimension: its name, physical domain, and
    how many collocation points to sample per training step.

    The trainer uses this to build the collocation tensor that is
    passed (with requires_grad=True) to the residual function.

    Parameters
    ----------
    name      : label used in error messages and logging
    domain    : (low, high) — physical bounds of the dimension
    n_colloc  : default number of interior collocation points

    Example
    -------
    x = PINSpatialDimension("x", domain=(0.0, 1.0), n_colloc=200)
    y = PINSpatialDimension("y", domain=(0.0, 1.0), n_colloc=200)
    """

    def __init__(self, name: str, domain: tuple, n_colloc: int = 100):
        low, high = domain
        if low >= high:
            raise ValueError(f"domain must satisfy low < high, got {domain}")
        self.name = name
        self.domain = domain
        self.n_colloc = n_colloc

    def sample(self, n=None, device="cpu") -> torch.Tensor:
        """Return an (n, 1) tensor of uniformly sampled interior points."""
        n = n or self.n_colloc
        low, high = self.domain
        return torch.empty(n, 1, device=device).uniform_(low, high)

    def __repr__(self):
        return (
            f"PINSpatialDimension(name={self.name!r}, "
            f"domain={self.domain}, n_colloc={self.n_colloc})"
        )
