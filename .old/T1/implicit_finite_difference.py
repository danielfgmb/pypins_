"""
Implicit Finite Difference Solver for 1D Diffusion Equation
This serves as the "source of truth" for validation with independent time stepping
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class ImplicitFiniteDifference:
    """
    Implicit finite difference solver for 1D diffusion equation using backward Euler method
    
    du/dt = nu * d²u/dx²
    
    This method is unconditionally stable and provides high accuracy for validation.
    """
    
    def __init__(self, nx, nt_independent, x_range=(0, 2), t_range=(0, 0.1), nu=0.3):
        """
        Initialize the implicit finite difference solver
        
        Args:
            nx: Number of spatial points
            nt_independent: Independent number of time steps (not tied to PINN)
            x_range: Spatial domain (x_min, x_max)
            t_range: Time domain (t_min, t_max)
            nu: Diffusion coefficient
        """
        self.nx = nx
        self.nt = nt_independent
        self.nu = nu
        
        # Spatial and temporal discretization
        self.dx = (x_range[1] - x_range[0]) / nx
        self.dt = (t_range[1] - t_range[0]) / nt_independent
        
        # Coordinate arrays
        self.x = np.linspace(x_range[0], x_range[1], nx + 1)
        self.t = np.linspace(t_range[0], t_range[1], nt_independent + 1)
        
        # Stability parameter (not restricted for implicit method)
        self.r = self.nu * self.dt / (self.dx ** 2)
        
        print(f"✓ Implicit FD solver initialized:")
        print(f"   Spatial points: {nx + 1}")
        print(f"   Time steps: {nt_independent + 1}")
        print(f"   dx: {self.dx:.6f}")
        print(f"   dt: {self.dt:.6f}")
        print(f"   Stability parameter r: {self.r:.6f} (unconditionally stable)")
        
        # Solution storage
        self.u = np.zeros((nt_independent + 1, nx + 1))
        self.initialized = False
        
        # Precompute the system matrix (tridiagonal)
        self._setup_system_matrix()
    
    def _setup_system_matrix(self):
        """Set up the tridiagonal system matrix for the implicit scheme"""
        n = self.nx - 1  # Interior points only
        
        # Tridiagonal matrix coefficients for implicit backward Euler
        # (I - r*A)*u^{n+1} = u^n
        # where A is the second difference matrix
        
        main_diag = np.ones(n) * (1 + 2 * self.r)  # Main diagonal
        off_diag = np.ones(n - 1) * (-self.r)      # Off diagonals
        
        # Create tridiagonal matrix
        self.A_matrix = diags([off_diag, main_diag, off_diag], [-1, 0, 1], 
                             shape=(n, n), format='csr')
        
        print(f"✓ System matrix setup complete (size: {n} × {n})")
    
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
        """
        Solve the diffusion equation using implicit backward Euler method
        
        This is unconditionally stable and highly accurate, making it perfect
        as a "source of truth" for validation.
        """
        if not self.initialized:
            raise ValueError("Initial condition not set. Call set_initial_condition first.")
        
        print("🧮 Solving diffusion equation with implicit method...")
        
        # Time stepping loop
        for n in range(self.nt):
            # Current solution (interior points only)
            u_current = self.u[n, 1:-1]  # Exclude boundaries
            
            # Right-hand side: current solution + boundary contributions
            rhs = u_current.copy()
            
            # Apply boundary conditions to RHS
            # Homogeneous Neumann boundary conditions: du/dx = 0
            # This means u[0] = u[1] and u[-1] = u[-2]
            # These contribute to the RHS for the first and last interior points
            rhs[0] += self.r * self.u[n, 0]    # Left boundary contribution
            rhs[-1] += self.r * self.u[n, -1]  # Right boundary contribution
            
            # Solve the linear system: A * u_new = rhs
            u_new_interior = spsolve(self.A_matrix, rhs)
            
            # Update solution (interior points)
            self.u[n+1, 1:-1] = u_new_interior
            
            # Apply boundary conditions (Neumann: du/dx = 0)
            self.u[n+1, 0] = self.u[n+1, 1]    # Left boundary
            self.u[n+1, -1] = self.u[n+1, -2]  # Right boundary
            
            # Progress indicator
            if (n + 1) % max(1, self.nt // 10) == 0:
                progress = (n + 1) / self.nt * 100
                print(f"   Progress: {progress:.0f}%")
        
        print(f"✓ Implicit solution completed ({self.nt} time steps)")
        return self.u
    
    def get_solution_at_time(self, time_value):
        """
        Get solution at a specific time value (interpolated if necessary)
        
        Args:
            time_value: Time at which to get the solution
            
        Returns:
            Solution array at the requested time
        """
        if time_value <= 0:
            return self.u[0, :]
        elif time_value >= self.t[-1]:
            return self.u[-1, :]
        else:
            # Linear interpolation between time steps
            t_idx = np.searchsorted(self.t, time_value)
            if t_idx == 0:
                return self.u[0, :]
            
            # Interpolation weights
            t_lower, t_upper = self.t[t_idx-1], self.t[t_idx]
            w_upper = (time_value - t_lower) / (t_upper - t_lower)
            w_lower = 1 - w_upper
            
            return w_lower * self.u[t_idx-1, :] + w_upper * self.u[t_idx, :]
    
    def get_solution_at_index(self, time_index):
        """Get solution at specific time index"""
        if time_index >= len(self.u):
            return self.u[-1, :]
        return self.u[time_index, :]
    
    def plot_solution_evolution(self, n_curves=10, save_path=None):
        """Plot the evolution of the solution"""
        plt.figure(figsize=(10, 6))
        
        time_indices = np.linspace(0, len(self.t)-1, n_curves).astype(int)
        
        for i, t_idx in enumerate(time_indices):
            alpha = 0.3 + 0.7 * (i / (len(time_indices) - 1))  # Fade from light to dark
            plt.plot(self.x, self.u[t_idx, :], alpha=alpha, 
                    label=f't={self.t[t_idx]:.4f}' if i % 3 == 0 else '')
        
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title('Implicit FD Solution Evolution (Source of Truth)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"✓ Solution plot saved to {save_path}")
        
        plt.show()
    
    def get_solution_info(self):
        """Get information about the solution"""
        return {
            'nx': self.nx,
            'nt': self.nt,
            'dx': self.dx,
            'dt': self.dt,
            'nu': self.nu,
            'r': self.r,
            'x_range': (self.x[0], self.x[-1]),
            't_range': (self.t[0], self.t[-1]),
            'method': 'Implicit Backward Euler',
            'stability': 'Unconditionally Stable'
        }


def create_test_initial_condition(x):
    """Create the same initial condition as used in PINN"""
    u0 = np.ones_like(x)
    # Step function: u = 2 for 0.25 ≤ x ≤ 0.5, u = 1 elsewhere
    mask = (x >= 0.5) & (x <= 1.0)  # Adjusted for the diffusion problem
    u0[mask] = 2.0
    return u0


if __name__ == "__main__":
    # Test the implicit solver
    print("Testing Implicit Finite Difference Solver")
    print("=" * 50)
    
    # Create solver with high resolution for accuracy
    solver = ImplicitFiniteDifference(
        nx=1000,           # High spatial resolution
        nt_independent=500, # Independent time stepping
        x_range=(0, 2),
        t_range=(0, 0.1),
        nu=0.3
    )
    
    # Set initial condition
    solver.set_initial_condition(create_test_initial_condition)
    
    # Solve
    solution = solver.solve()
    
    # Plot results
    solver.plot_solution_evolution()
    
    # Print solution info
    info = solver.get_solution_info()
    print("\n📊 Solution Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
