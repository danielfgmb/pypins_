import torch.nn as nn
from kan import *

"""
try:
    from kan import KAN as PyKAN
    _HAS_KAN = True
except ImportError:
    _HAS_KAN = False
"""
_HAS_KAN = True

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
    """

    def __init__(self, layers=(1, 32, 32, 1), grid_size=5, spline_order=3):
        super().__init__()
        if not _HAS_KAN:
            raise ImportError(
                "pykan is not installed.\n"
                "Run: pip install pykan"
            )
        self.net = MultKAN(
            width=list(layers),
            grid=grid_size,
            k=spline_order,
        )

    def forward(self, x):
        return self.net(x)


def PINArchitecture(arch="MLP", params_arch=None):
    """
    Factory that returns an nn.Module for the chosen architecture.

    arch        : "MLP" or "KAN"
    params_arch : dict of kwargs forwarded to the constructor

    MLP kwargs  : layers (tuple), activation (str)
    KAN kwargs  : layers (tuple), grid_size (int), spline_order (int)

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
