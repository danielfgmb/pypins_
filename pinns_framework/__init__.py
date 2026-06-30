from .architecture import PINArchitecture
from .dimensions import PINSpatialDimension
from .loss import PINLossFunction
from .trainer import PINTrainer, save_pinns, load_pinns
from .collocation import (
    collocation_uniform,
    collocation_lhs,
    collocation_gradient,
    collocation_curvature,
    collocation_last_residual,
    collocation_mean_residual,
)
from .metrics import make_eval_grid, compute_metrics, compute_error

__all__ = [
    "PINArchitecture",
    "PINSpatialDimension",
    "PINLossFunction",
    "PINTrainer",
    "save_pinns",
    "load_pinns",
    # Collocation strategies
    "collocation_uniform",
    "collocation_lhs",
    "collocation_gradient",
    "collocation_curvature",
    "collocation_last_residual",
    "collocation_mean_residual",
    # Evaluation & metrics
    "make_eval_grid",
    "compute_metrics",
    "compute_error",
]
