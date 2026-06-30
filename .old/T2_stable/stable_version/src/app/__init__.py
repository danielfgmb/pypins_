from .model import TD_PINNS, burgers_1d_loss, burgers_2d_loss
from .trainer import TrainerBurger1D, TrainerBurger2D
from .logger import Logger
from .grapher import (
    plot_1d, plot_1d_evolution,
    plot_comparison_1d, plot_comparison_evolution,
    plot_error_evolution, plot_conservation,
    plot_2d_contour, plot_2d_evolution,
)
from .finite_diff import BurgersFD1D, BurgersFD_CN, compute_errors, build_comparison_table
from .higher_order import TrainerBurger1D_HO, burgers_1d_cn_loss, burgers_1d_bdf2_loss
