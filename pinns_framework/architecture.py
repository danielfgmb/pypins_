import torch.nn as nn

# pykan's network class is `MultKAN`. Older code imported `KAN` (`from kan import KAN`),
# which newer pykan no longer exposes as a constructor — import `MultKAN` directly.
try:
    from kan import MultKAN
    _HAS_KAN = True
except ImportError:
    _HAS_KAN = False

class MLP(nn.Module):
    """Feedforward network with Xavier initialization — the classic PINN backbone."""

    def __init__(self, layers=(1, 64, 64, 64, 1), activation="tanh"):
        super().__init__()
        act_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation not in act_map:
            raise ValueError(f"activation must be one of {list(act_map)}")
        act_cls = act_map[activation]

        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(act_cls())
        self.net = nn.Sequential(*mods)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class KANNet(nn.Module):
    """
    Kolmogorov–Arnold Network via pykan.

    Install: pip install pykan
    Compatible with torch.autograd.grad so spatial derivatives work the
    same way as for MLP — no changes needed in the residual function.

    Optional adaptive grid refinement
    ---------------------------------
    When ``refine_every`` is set, the network advertises a refinement policy
    that ``PINTrainer`` honours during training: every ``refine_every`` epochs
    the trainer measures how fast the loss is dropping and, only if it is *not*
    dropping fast enough (relative window improvement below ``refine_tol``),
    calls :meth:`refine_grid` to make the spline grid finer (pykan's grid
    extension). Leaving ``refine_every=None`` disables it entirely — the net
    behaves exactly as before.
    """

    def __init__(self, layers=(1, 32, 32, 1), grid_size=5, spline_order=3,
                 grid_range=(-1.0, 1.0), grid_eps=0.02,
                 refine_every=None, refine_factor=1.1, refine_tol=0.1,
                 refine_max_grid=None):
        super().__init__()
        if not _HAS_KAN:
            raise ImportError(
                "pykan is not installed.\n"
                "Run: pip install pykan"
            )
        # grid_range : spline knot extent (set to the input domain)
        # grid_eps   : 0 -> fully adaptive (quantile) grid, 1 -> uniform grid
        self.net = MultKAN(
            width=list(layers),
            grid=grid_size,
            k=spline_order,
            grid_range=list(grid_range),
            grid_eps=grid_eps,
        )
        # ── adaptive grid-refinement policy (all optional) ──────────────
        # refine_every    : check for refinement every N epochs (None = off)
        # refine_factor   : grid-size multiplier per refinement (1.1 = +10 %)
        # refine_tol      : minimum relative loss improvement expected over the
        #                   window; refine only when improvement is below it
        #                   ("the loss is not reducing fast enough")
        # refine_max_grid : hard cap on grid intervals (None = uncapped)
        self.refine_every    = refine_every
        self.refine_factor   = refine_factor
        self.refine_tol      = refine_tol
        self.refine_max_grid = refine_max_grid

    def forward(self, x):
        return self.net(x)

    def refine_grid(self, device=None):
        """
        Make the spline grid finer via pykan's ``MultKAN.refine``.

        The inner ``MultKAN`` is replaced **in place**, so external references
        to this ``KANNet`` wrapper (trainer loss closures, the optimiser target)
        stay valid — only the optimiser must be rebuilt afterwards because the
        parameter tensors are new.

        Returns the new grid size, or ``None`` if refinement is disabled or the
        ``refine_max_grid`` cap has been reached (so the caller can stop trying).
        """
        if not self.refine_every:
            return None
        cur   = int(self.net.grid)
        new_g = max(cur + 1, int(round(cur * self.refine_factor)))
        if self.refine_max_grid is not None and new_g > self.refine_max_grid:
            return None
        # pykan stores its device as a plain attribute that ``.to()`` does NOT
        # update; refine() reads it to build the finer model, so sync it to the
        # parameters' real device first to avoid a cuda/cpu mismatch.
        dev = device if device is not None else next(self.net.parameters()).device
        self.net.device = dev
        refined = self.net.refine(new_g).to(dev)
        refined.device = dev
        self.net = refined
        return new_g


def PINArchitecture(arch="MLP", params_arch=None):
    """
    Factory that returns an nn.Module for the chosen architecture.

    arch        : "MLP" or "KAN"
    params_arch : dict of kwargs forwarded to the constructor

    MLP kwargs  : layers (tuple), activation (str)
    KAN kwargs  : layers (tuple), grid_size (int), spline_order (int),
                  grid_range (tuple, spline knot extent), grid_eps (float,
                  0 = adaptive grid, 1 = uniform grid),
                  refine_every (int|None — adaptive grid refinement interval in
                  epochs, None = off), refine_factor (float, grid growth per
                  refinement, 1.1 = +10%), refine_tol (float, min relative loss
                  improvement before refining), refine_max_grid (int|None, cap)

    Example
    -------
    net = PINArchitecture("MLP", {"layers": (1, 64, 64, 1), "activation": "tanh"})
    net = PINArchitecture("KAN", {"layers": (1, 16, 1), "grid_size": 5})
    """
    params = params_arch or {}
    if arch == "MLP":
        return MLP(**params)
    if arch == "KAN":
        return KANNet(**params)
    raise ValueError(f"Unknown arch '{arch}'. Choose 'MLP' or 'KAN'.")
