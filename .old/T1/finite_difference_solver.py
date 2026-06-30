"""
Finite Difference Solver for 1D Diffusion Equation and Validation Functions
"""

import numpy as np
import os


class FiniteDifferenceSolver:
    """Finite difference solver for 1D diffusion equation"""
    
    def __init__(self, nx, nt, x_range=(0, 2), t_range=(0, 0.1), nu=0.3):
        self.nx = nx
        self.nt = nt
        self.nu = nu
        
        self.dx = (x_range[1] - x_range[0]) / nx
        self.dt = (t_range[1] - t_range[0]) / nt
        
        self.x = np.linspace(x_range[0], x_range[1], nx + 1)
        self.t = np.linspace(t_range[0], t_range[1], nt + 1)
        
        # Check stability condition (CFL condition)
        self.r = self.nu * self.dt / (self.dx ** 2)
        if self.r > 0.5:
            print(f"⚠️ Warning: CFL condition violated! r = {self.r:.4f} > 0.5")
            print(f"   Solution may be unstable. Consider reducing dt or increasing nx.")
        else:
            print(f"✓ FD solver stable: r = {self.r:.4f} ≤ 0.5")
        
        # Initialize solution array
        self.u = np.zeros((nt + 1, nx + 1))
        self.initialized = False
    
    def set_initial_condition(self, u0_func):
        """Set initial condition using a function"""
        self.u[0, :] = u0_func(self.x)
        self.initialized = True
    
    def set_initial_condition_from_array(self, u0_array):
        """Set initial condition from array"""
        if len(u0_array) != len(self.x):
            # Interpolate if sizes don't match
            x_old = np.linspace(0, 2, len(u0_array))
            self.u[0, :] = np.interp(self.x, x_old, u0_array)
        else:
            self.u[0, :] = u0_array
        self.initialized = True
    
    def solve(self):
        """Solve the diffusion equation using explicit finite differences"""
        if not self.initialized:
            raise ValueError("Initial condition not set. Call set_initial_condition first.")
        
        # Time stepping
        for n in range(self.nt):
            for i in range(1, self.nx):  # Interior points
                self.u[n+1, i] = (self.u[n, i] + 
                                  self.r * (self.u[n, i+1] - 2*self.u[n, i] + self.u[n, i-1]))
            
            # Boundary conditions (assuming homogeneous Neumann: du/dx = 0)
            self.u[n+1, 0] = self.u[n+1, 1]    # Left boundary
            self.u[n+1, -1] = self.u[n+1, -2]  # Right boundary
        
        return self.u
    
    def get_solution_at_time(self, time_step):
        """Get solution at specific time step"""
        if time_step >= len(self.u):
            return self.u[-1, :]  # Return final solution if beyond range
        return self.u[time_step, :]


def calculate_validation_errors(pinn_solution, fd_solution):
    """
    Calculate validation errors between PINN and finite difference solutions
    
    Args:
        pinn_solution: PINN solution array
        fd_solution: Finite difference solution array  
    
    Returns:
        Dictionary with mean_error, rmse, and max_absolute_error
    """
    if len(pinn_solution) != len(fd_solution):
        # Interpolate FD solution to match PINN grid if needed
        x_fd = np.linspace(0, 2, len(fd_solution))
        x_pinn = np.linspace(0, 2, len(pinn_solution))
        fd_solution = np.interp(x_pinn, x_fd, fd_solution)
    
    diff = pinn_solution - fd_solution
    
    return {
        'mean_error': np.mean(np.abs(diff)),           # Previously L1
        'rmse': np.sqrt(np.mean(diff**2)),            # Previously L2  
        'max_absolute_error': np.max(np.abs(diff))     # Previously Linf
    }


def load_solution_arrays(results_dir):
    """
    Load saved 2D solution arrays
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary containing pinn_solutions, fd_solutions, x_coordinates, time_values
    """
    arrays_path = os.path.join(results_dir, "solution_arrays.npz")
    if not os.path.exists(arrays_path):
        raise FileNotFoundError(f"Solution arrays file {arrays_path} not found")
    
    data = np.load(arrays_path)
    return {
        'pinn_solutions': data['pinn_solutions'],
        'fd_solutions': data['fd_solutions'],
        'x_coordinates': data['x_coordinates'],
        'time_values': data['time_values']
    }
