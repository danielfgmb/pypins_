#%%

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import os
import json
from datetime import datetime
import pandas as pd
from finite_difference_solver import calculate_validation_errors, load_solution_arrays
from implicit_finite_difference import ImplicitFiniteDifference, create_test_initial_condition
from utils import (create_experiment_folders, save_hyperparameters, save_solution_dataframe,
                   save_solution_arrays, calculate_and_save_validation_summary, create_validation_plots,
                   save_training_summary, print_final_results)

# Enable optimizations
torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
torch.set_float32_matmul_precision('medium')  # For faster matmul on modern hardware

# Memory optimization
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # MPS specific optimizations
    torch.mps.empty_cache()

# Device selection - Use MPS on Mac if available, otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Device selected: {device}")

# Uncomment the line below to force CPU usage (for debugging)
device = torch.device("cpu")



# Create experiment folders
exec_results_dir, exec_models_dir, time_models_dir = create_experiment_folders()



class TD_PINNS_Implicit(nn.Module):
    def __init__(self, x_dim=1, hid=32, u_dim=1, nu=0.3, dt=0.01, device=device):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
                nn.Linear(x_dim,hid), # entra a la red la IC + x_dim
                nn.Tanh(),
                nn.Linear(hid,hid),
                nn.Tanh(),
                nn.Linear(hid,hid),
                nn.Tanh(),
                nn.Linear(hid,u_dim)  # Sin Tanh final para permitir valores > 1
            )
        
        self.nu = nu          # <-- FALTABA
        self.dt = dt          # <-- FALTABA
            
        # Xavier initialization
        for module in self.net:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight.data, gain=1.)
                torch.nn.init.zeros_(module.bias.data)
        
        # Move model to device
        self.to(device)
        
        # Note: torch.compile disabled for PINN due to higher-order gradient requirements
        # PINNs need create_graph=True for second derivatives, which conflicts with compiled models
        # Uncomment below for non-PINN applications:
        # try:
        #     self.net = torch.compile(self.net, mode='default')
        #     print("✓ Model compiled with torch.compile for faster execution")
        # except:
        #     print("⚠ torch.compile not available, continuing without compilation")
        print("ℹ torch.compile disabled for PINN compatibility with higher-order gradients")
    
        
    def forward(self, x):
        return self.net(x)
    

    def loss_initial(self, X_train, u0_real):
        u0 = self.forward(X_train)
        loss_IC = torch.nn.functional.mse_loss(u0, u0_real)
        return loss_IC
    
    def loss(self, X_train, u0):
        u1 = self.forward(X_train)
        
        # Compute first derivative
        du1_dx = torch.autograd.grad(
            u1, X_train,
            grad_outputs=torch.ones_like(u1),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Compute second derivative
        du1_dxx = torch.autograd.grad(
            du1_dx, X_train,
            grad_outputs=torch.ones_like(du1_dx),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Compute PDE residual
        pde_residual = self.nu * du1_dxx - (u1 - u0) / self.dt
        loss_PDE = torch.mean(pde_residual**2)
        
        # Numerical stability check
        if torch.isnan(loss_PDE) or loss_PDE > 1e8:
            print(f"Warning: Loss PDE is NaN or too large: {loss_PDE.item()}")
            return torch.tensor(1e6, requires_grad=True, device=self.device)
        
        return loss_PDE

#%% Configuraciones

PINN = TD_PINNS_Implicit(device=device)

nx = 1000        # numero de puntos para generar la condicion inicial

nu = 0.3        # constante difusion

tf = 0.1        # tiempo final de simulacion
nt = 10      # numero de pasos de tiempo (numero de PINNs a la vez)

dt = tf/nt

x = torch.linspace(0, 2, nx+1).to(device)
t = torch.linspace(0, tf, nt+1).to(device)

X, T = torch.meshgrid(x, t)

#%% Boundary and Initial Conditions


x = x.reshape(-1, 1)                 # (N, 1)
y_input = torch.ones(x.shape[0], 1, device=device)  # (N, 1)
y_input[int(0.25*nx):int(0.5*nx)+1, 0] = 2.0

# Fronteras e interior (todas como columnas)
x_lb = x[:1, :]        # (1, 1)
x_ic = x[1:-1, :]      # (N-2, 1)
x_ub = x[-1:, :]       # (1, 1)

lb = y_input[:1, :]    # (1, 1)
ic = y_input[1:-1, :]  # (N-2, 1)
ub = y_input[-1:, :]   # (1, 1)

tol = 1e-3  # Tolerancia menos estricta para que converja más fácil
tol_ic = 1e-3  # Tolerancia menos estricta para que converja más fácil
epochs = 300000  # Menos epochs pero con mejor learning rate

n_prints = 300
n_plots = 10

# Collect hyperparameters
hyperparams = {
    "model_config": {
        "x_dim": 1,
        "hidden_size": 32,
        "u_dim": 1,
        "architecture": "3_layers_tanh"
    },
    "physics_params": {
        "nu": nu,
        "dt": dt,
        "tf": tf
    },
    "training_config": {
        "nx": nx,
        "nt": nt,
        "epochs": epochs,
        "tolerance": tol,
        "optimizer": "AdamW",
        "initial_learning_rate": 1e-2,
        "min_learning_rate": 1e-8,
        "weight_decay": 1e-4,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience_initial": 50,
        "scheduler_patience_timestep": 20,
        "scheduler_factor": 0.5,
        "instability_lr_reduction_factor": 0.1
    },
    "computation_config": {
        "device": str(device),
        "mixed_precision": torch.cuda.is_available(),
        "n_prints": n_prints,
        "n_plots": n_plots
    },
    "experiment_info": {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "description": "Time-dependent PINN for implicit PDE with diffusion"
    }
}

# Save hyperparameters
save_hyperparameters(exec_results_dir, hyperparams)

# Initialize implicit finite difference solver for validation (source of truth)
print("\n🔬 Setting up implicit finite difference solver for validation...")
fd_solver = ImplicitFiniteDifference(
    nx=nx*2,              # Higher resolution for better accuracy
    nt_independent=200,   # Independent time stepping (not tied to PINN)
    x_range=(0, 2), 
    t_range=(0, tf), 
    nu=nu
)

# Set initial condition for FD solver (same as PINN)
u0_numpy = y_input.cpu().numpy().flatten()
fd_solver.set_initial_condition_from_array(u0_numpy)

# Solve using implicit finite differences (unconditionally stable)
print("🧮 Solving reference solution with implicit finite differences...")
fd_solution = fd_solver.solve()
print(f"✓ Implicit finite difference reference solution computed")

# Initialize validation error tracking
validation_errors = {
    'time_steps': [],
    'time_values': [],
    'mean_errors': [],
    'rmse_errors': [],
    'max_absolute_errors': [],
    'training_losses': []
}

# Initialize solution data for DataFrame
solution_data = {
    'time_values': [],
    'pinn_solutions': [],
    'fd_solutions': []
}

# Initialize 2D numpy arrays for plotting data (time x space)
n_plot_points = len(range(0, nt+1, max(1, nt // n_plots)))  # Number of time points that will be plotted
x_plot = np.linspace(0, 2, nx+1)  # Spatial coordinates
t_plot_values = []  # Will store actual time values for plotting

# Initialize 2D arrays: rows = time steps, columns = spatial points
pinn_solutions_2d = np.zeros((n_plot_points + 1, len(x_plot)))  # +1 for initial condition
fd_solutions_2d = np.zeros((n_plot_points + 1, len(x_plot)))
plot_time_index = 0  # Counter for plotting time steps
#%%

# se declara una lista de PINNs donde se ira guardando cada PINN
# y otra lista donde se guardan los tiempos de entrenamiento de estos
lista_PINNs = []
train_time = []
ii = 0

# empieza a correr el tiempo de entrenamiento
TIC = time.time()


#%% Preentrenamiento



# empieza un contador de tiempo para cada PINN_j

PINN_old = TD_PINNS_Implicit(nu=nu, dt=dt, device=device)

tic = time.time()

optimizer = torch.optim.AdamW(PINN_old.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5, 
                                                      min_lr=1e-8)

def check_and_reduce_lr_on_instability(optimizer, scheduler, loss_value, min_lr=1e-8):
    """Reduce learning rate immediately if training becomes unstable"""
    current_lr = optimizer.param_groups[0]['lr']
    
    # Check for instability: NaN, infinite, or very large loss
    if not torch.isfinite(torch.tensor(loss_value)) or loss_value > 1e3:
        if current_lr > min_lr:
            new_lr = max(current_lr * 0.1, min_lr)  # Reduce by factor of 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"⚠️ Instability detected! Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
            return True
    return False

# Mixed precision training setup
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()  # Use AMP only with CUDA for now

x_train = torch.vstack([x_lb,x_ic,x_ub])
u0_train = torch.vstack([lb,ic,ub])

x_train.requires_grad_(True)

print("Iniciando pre-entrenamiento de la condición inicial...")
for i in range(epochs):
    optimizer.zero_grad()
    
    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast():
            loss = PINN_old.loss_initial(x_train, u0_train)
        scaler.scale(loss).backward()
    else:
        loss = PINN_old.loss_initial(x_train, u0_train)
        loss.backward()
    
    # Early stopping check
    loss_value = loss.item()
    if loss_value <= tol_ic:
        print(f"¡Convergencia lograda en epoch {i}! Loss final: {loss_value:.2e}")
        break
    
    # Check for instability and reduce LR if needed
    # instability_detected = check_and_reduce_lr_on_instability(optimizer, scheduler, loss_value)
    
    if use_amp and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
        
    # Step scheduler (reduce LR on plateau)
    scheduler.step(loss_value)
    
    # Periodic memory cleanup
    if i % 1000 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    if i % (epochs // n_prints) == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {i}: Loss = {loss_value:.7f}, LR = {current_lr:.2e}")
        
print(f"Pre-entrenamiento completado. Loss final: {loss_value:.2e}")

# Use no_grad for evaluation
with torch.no_grad():
    u0 = PINN_old.forward(x_train)
    u0_ = u0.clone()
    
    # Save initial condition (t=0) in 2D arrays
    u0_full = PINN_old.forward(x.requires_grad_(False))  # Get solution on full x domain
    pinn_solutions_2d[plot_time_index, :] = u0_full[:,0].cpu().numpy()
    
    # Get FD solution and interpolate to PINN grid
    fd_sol_t0 = fd_solver.get_solution_at_time(0.0)
    fd_x_coords = np.linspace(0, 2, len(fd_sol_t0))
    pinn_x_coords = np.linspace(0, 2, len(pinn_solutions_2d[0]))
    fd_sol_interp = np.interp(pinn_x_coords, fd_x_coords, fd_sol_t0)
    fd_solutions_2d[plot_time_index, :] = fd_sol_interp
    
    t_plot_values.append(0.0)  # t=0 for initial condition
    plot_time_index += 1



for j in range(len(t)):
    print("------------------------------\nTime # ", j)

    x_train = torch.vstack([x_ub,x_ic,x_lb])
    u0_train = torch.vstack([ub,ic,lb])

    x_train.requires_grad_(True)

    PINN_j = TD_PINNS_Implicit(nu=nu, dt=dt, device=device)
    
    # Transferred learning

    PINN_j.load_state_dict(PINN_old.state_dict())

    # empieza un contador de tiempo para cada PINN_j
    tic = time.time()

    optimizer = torch.optim.AdamW(PINN_j.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler_j = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, 
                                                            min_lr=1e-8)
    scaler_j = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        if use_amp and scaler_j is not None:
            with torch.cuda.amp.autocast():
                loss = PINN_j.loss(x_train, u0_train)
            scaler_j.scale(loss).backward()
        else:
            loss = PINN_j.loss(x_train, u0_train)
            loss.backward()
        
        # Check convergence
        loss_value = loss.item()
        if loss_value <= tol:
            print(f"epoch until tolerance: {i}")
            break
        
        # Check for instability and reduce LR if needed
        # instability_detected = check_and_reduce_lr_on_instability(optimizer, scheduler_j, loss_value)
        
        if use_amp and scaler_j is not None:
            scaler_j.step(optimizer)
            scaler_j.update()
        else:
            optimizer.step()
            
        # Step scheduler (reduce LR on plateau)
        scheduler_j.step(loss_value)
        
        # Memory cleanup every 500 steps
        if i % 500 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        if i % (epochs // n_prints) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"training: {loss_value:.7f}, LR = {current_lr:.2e}")

    
    
    # termina el contador de tiempo del PINN_j  
    toc = time.time()
    elapsed_time = toc-tic
    lista_PINNs.append(PINN_j)
    PINN_old = PINN_j
    train_time.append(elapsed_time)
    print("Elapsed time: ", elapsed_time, " seconds")

    #% Plot with optimized evaluation
    with torch.no_grad():
        u1 = PINN_j.forward(x_train.detach())  # Detach input to avoid gradients
        u1_ = u1.clone()
        ic = u1_[1:-1]
        
    # Update training data for next iteration (outside no_grad context)
    x_train = x_train.detach()

    if j % (nt // n_plots) == 0:
        # Get PINN solution on full domain for 2D array storage
        with torch.no_grad():
            u_full = PINN_j.forward(x.requires_grad_(False))
            pinn_solution_full = u_full[:,0].cpu().numpy()
        
        # Calculate validation errors against implicit finite difference solution
        time_value = float(t[j]) if j < len(t) else float(j * dt)
        pinn_solution_numpy = u1_[:,0].cpu().numpy()
        fd_solution_at_time = fd_solver.get_solution_at_time(time_value)  # Use actual time value
        
        # Calculate validation errors
        errors = calculate_validation_errors(pinn_solution_numpy, fd_solution_at_time)
        
        # Store in 2D arrays for plotting
        if plot_time_index < pinn_solutions_2d.shape[0]:
            pinn_solutions_2d[plot_time_index, :] = pinn_solution_full
            
            # Get FD solution and interpolate to PINN grid
            fd_sol_t = fd_solver.get_solution_at_time(time_value)
            fd_x_coords = np.linspace(0, 2, len(fd_sol_t))
            pinn_x_coords = np.linspace(0, 2, len(pinn_solutions_2d[0]))
            fd_sol_interp = np.interp(pinn_x_coords, fd_x_coords, fd_sol_t)
            fd_solutions_2d[plot_time_index, :] = fd_sol_interp
            
            t_plot_values.append(time_value)
            plot_time_index += 1
        
        # Store validation errors (no printing during cycle)
        validation_errors['time_steps'].append(j)
        validation_errors['time_values'].append(time_value)
        validation_errors['mean_errors'].append(float(errors['mean_error']))
        validation_errors['rmse_errors'].append(float(errors['rmse']))
        validation_errors['max_absolute_errors'].append(float(errors['max_absolute_error']))
        validation_errors['training_losses'].append(float(loss_value))
        
        # Store solution data for DataFrame
        solution_data['time_values'].append(time_value)
        solution_data['pinn_solutions'].append(pinn_solution_numpy.copy())
        solution_data['fd_solutions'].append(fd_solution_at_time.copy())
        
        # Save model at this time step
        time_step_str = f"time_step_{j:04d}"
        model_path = os.path.join(time_models_dir, f"{time_step_str}.pt")
        
        # Save model state and metadata (including validation errors)
        model_save_dict = {
            'model_state_dict': PINN_j.state_dict(),
            'time_step': j,
            'time_value': time_value,
            'training_loss': loss_value,
            'training_epochs': i,
            'elapsed_time': elapsed_time,
            'validation_errors': {
                'mean_error': float(errors['mean_error']),
                'rmse': float(errors['rmse']),
                'max_absolute_error': float(errors['max_absolute_error'])
            },
            'hyperparameters': hyperparams
        }
        torch.save(model_save_dict, model_path)
        print(f"✓ Saved model at time step {j} to {model_path}")

TOC = time.time()
TOTAL_TIME = TOC - TIC

# Trim arrays to actual used size
actual_plot_points = len(t_plot_values)
pinn_solutions_2d = pinn_solutions_2d[:actual_plot_points, :]
fd_solutions_2d = fd_solutions_2d[:actual_plot_points, :]

# Save 2D numpy arrays
save_solution_arrays(pinn_solutions_2d, fd_solutions_2d, x_plot, t_plot_values, exec_results_dir)

# Clear the current plot and recreate it from saved arrays
plt.clf()
plt.figure(figsize=(10, 6))

# Plot from saved 2D arrays
for i in range(len(t_plot_values)):
    plt.plot(x_plot, pinn_solutions_2d[i, :], label=f"t={t_plot_values[i]:.3f}" if i % 2 == 0 else "")

plt.xticks(np.linspace(0, 2, 11))
plt.yticks(np.linspace(1, 2, 6))
plt.xlabel("x", fontsize=15)
plt.ylabel("u(x,t)", fontsize=15)
plt.legend(["Condición Inicial", "Solución"])
plt.title("Implicit PDE - From Saved Arrays", fontsize=15)

# Save plot to results folder
plot_path = os.path.join(exec_results_dir, "solution_evolution.svg")
plt.savefig(plot_path)
plt.savefig("DiffMM.svg")  # Keep original save for backward compatibility
print(f"✓ Saved solution plot (from arrays) to {plot_path}")

# Save DataFrame and calculate validation statistics
save_solution_dataframe(solution_data, exec_results_dir)
validation_stats = calculate_and_save_validation_summary(validation_errors, exec_results_dir)

# Save training summary
loss_value_final = loss_value if 'loss_value' in locals() else None
training_summary = save_training_summary(TOTAL_TIME, t, time_models_dir, loss_value_final, 
                                        tol, epochs, validation_stats, fd_solver, exec_results_dir, train_time)

# Create validation error plots
create_validation_plots(validation_errors, exec_results_dir)

# Save final trained model
final_model_path = os.path.join(exec_models_dir, "final_model.pt")
if 'PINN_j' in locals():
    final_model_dict = {
        'model_state_dict': PINN_j.state_dict(),
        'final_time': tf,
        'training_complete': True,
        'validation_errors': validation_errors,
        'pinn_solutions_2d': pinn_solutions_2d,
        'fd_solutions_2d': fd_solutions_2d,
        'x_coordinates': x_plot,
        'time_values': np.array(t_plot_values),
        'hyperparameters': hyperparams,
        'training_summary': training_summary
    }
    torch.save(final_model_dict, final_model_path)
    print(f"✓ Saved final model to {final_model_path}")

# Print final results
print("Total runtime: ", TOTAL_TIME, " seconds")
print_final_results(exec_results_dir, exec_models_dir, time_models_dir, 
                   pinn_solutions_2d, t_plot_values, tf, validation_stats)

    












# %%
