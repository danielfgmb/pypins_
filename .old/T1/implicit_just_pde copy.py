#%%

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

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

nx = 10000        # numero de puntos para generar la condicion inicial

nu = 0.3        # constante difusion

tf = 0.1        # tiempo final de simulacion
Nf = 100        # numero de puntos de colocacion para entrenamiento
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
epochs = 300000  # Menos epochs pero con mejor learning rate

n_prints = 300
n_plots = 10
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

optimizer = torch.optim.AdamW(PINN_old.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.8)

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
    if loss_value <= tol:
        print(f"¡Convergencia lograda en epoch {i}! Loss final: {loss_value:.2e}")
        break
    
    if use_amp and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
        
    scheduler.step(loss)
    
    # Periodic memory cleanup
    if i % 1000 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    if i % (epochs // n_prints) == 0:
        print(f"Epoch {i}: Loss = {loss_value:.7f}")
        
print(f"Pre-entrenamiento completado. Loss final: {loss_value:.2e}")

# Use no_grad for evaluation
with torch.no_grad():
    u0 = PINN_old.forward(x_train)
    u0_ = u0.clone()
    plt.plot(x[:,0].cpu().numpy(), u0_[:,0].cpu().numpy())



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

    optimizer = torch.optim.AdamW(PINN_j.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_j = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.9)
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
        
        if use_amp and scaler_j is not None:
            scaler_j.step(optimizer)
            scaler_j.update()
        else:
            optimizer.step()
            
        scheduler_j.step(loss_value)
        
        # Memory cleanup every 500 steps
        if i % 500 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        if i % (epochs // n_prints) == 0:
            print(f"training: {loss_value:.7f}")

    
    
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
        plt.plot(x[:,0].cpu().numpy(), u1_[:,0].cpu().numpy())

TOC = time.time()
TOTAL_TIME = TOC - TIC

plt.xticks(np.linspace(0, 2, 11))
plt.yticks(np.linspace(1, 2, 6))
plt.xlabel("x", fontsize=15)
plt.ylabel("u(x,t)", fontsize=15)
plt.legend(["Condición Inicial", "Solución"])
plt.title("Implicit PDE", fontsize=15)
plt.savefig("DiffMM.svg")

# termina el temporizador y se estima el tiempo de entrenamiento total
print("Total runtime: ", TOTAL_TIME, " seconds")

    












# %%
