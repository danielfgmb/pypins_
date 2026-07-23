from .ref_tensors import (
    mass_matrix,
    stiffness_matrix,
    convection_matrix,
    endpoint_values,
    triple_matrix,
    deriv_triple_matrix,
    transfer_matrices,
    legendre_basis,
    legendre_and_deriv,
    verify_all,
)
from .kan import (
    LegendreEdge,
    LegendreKANLayer,
    LegendreKAN,
    project_to_legendre,
    closure_residual,
)
from .assembly import (
    jacobian_scale,
    dirichlet_recombination_basis,
    constant_dirichlet_lift,
    upwind_trace,
    sipg_penalty,
)
from .mesh import Mesh1D, assemble_advection_diffusion
from .refine import compute_indicators, dorfler_mark, refine_mesh
from .solve import (
    solve_poisson_single_cell,
    solve_burgers_single_cell,
    solve_advection_diffusion_multi_cell,
)

__all__ = [
    "mass_matrix",
    "stiffness_matrix",
    "convection_matrix",
    "endpoint_values",
    "triple_matrix",
    "deriv_triple_matrix",
    "transfer_matrices",
    "legendre_basis",
    "legendre_and_deriv",
    "verify_all",
    "LegendreEdge",
    "LegendreKANLayer",
    "LegendreKAN",
    "project_to_legendre",
    "closure_residual",
    "jacobian_scale",
    "dirichlet_recombination_basis",
    "constant_dirichlet_lift",
    "upwind_trace",
    "sipg_penalty",
    "Mesh1D",
    "assemble_advection_diffusion",
    "compute_indicators",
    "dorfler_mark",
    "refine_mesh",
    "solve_poisson_single_cell",
    "solve_burgers_single_cell",
    "solve_advection_diffusion_multi_cell",
]
