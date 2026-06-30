#%%

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube


class Block1(nn.Module):

    def __init__(self, x_dim=1, hid=64, u_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim,hid),
            nn.Tanh(),
            nn.Linear(hid,hid),
            nn.Tanh(),
            nn.Linear(hid,u_dim)
        )
        
        # Xavier initialization
        for module in self.net:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight.data, gain=1.)
                torch.nn.init.zeros_(module.bias.data)
    
    def forward(self, x):
        return self.net(x)
    
class Block2(nn.Module):

    def __init__(self, x_dim=1, hid=64, u_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+u_dim,hid), # entra a la red la IC + x_dim
            nn.Tanh(),
            nn.Linear(hid,hid),
            nn.Tanh(),
            nn.Linear(hid,u_dim)
        )
        
        # Xavier initialization
        for module in self.net:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight.data, gain=1.)
                torch.nn.init.zeros_(module.bias.data)
    
    def forward(self, x, u1):

        return self.net(torch.hstack((x,u1))) # htack([...],[...]) lo deja como 1 solo vector []
    
class TD_PINNS_Implicit(nn.Module):
    def __init__(self, x_dim=1, hid=64, u_dim=1, nu=0.3, dt=0.01):
        super().__init__()
        self.block1 = Block1(x_dim, hid, u_dim)
        self.block2 = Block2(x_dim, hid, u_dim)
        self.nu = nu  # diffusion parameter
        self.dt = dt  # time step
        
    def forward(self, x):
        u1 = self.block1(x)
        u2 = self.block2(x, u1)
        return u1, u2
    
    def loss(self, X_train, Y_train):
        """
        Compute combined loss: IC loss + PDE loss
        X_train: spatial points [N, 1] 
        Y_train: target values [N, 1]
        """
        u1, u2 = self.forward(X_train)
        
        # Initial condition loss
        loss_IC = torch.nn.functional.mse_loss(u1, Y_train)
        
        # PDE residual loss
        du2_dx = torch.autograd.grad(u2, X_train,
                                   torch.ones_like(u2),
                                   retain_graph=True,
                                   create_graph=True)[0]
        du2_dxx = torch.autograd.grad(du2_dx, X_train,
                                    torch.ones_like(du2_dx),
                                    retain_graph=True,
                                    create_graph=True)[0]
        
        pde_residual = self.nu * du2_dxx - (u2 - u1) / self.dt
        loss_PDE = torch.mean(pde_residual**2)
        
        return loss_IC + loss_PDE
    
    def gradiente(self, x):
        """
        Compute second derivative (Laplacian) for advancement
        """
        u1, u2 = self.forward(x)
        du2_dx = torch.autograd.grad(u2, x,
                                   torch.ones_like(u2),
                                   retain_graph=True,
                                   create_graph=True)[0]
        du2_dxx = torch.autograd.grad(du2_dx, x,
                                    torch.ones_like(du2_dx),
                                    retain_graph=True,
                                    create_graph=True)[0]
        return du2_dxx

#%%

# Ecuación de difusión

PINN = TD_PINNS_Implicit()

nx = 100        # numero de puntos para generar la condicion inicial

nu = 0.3

tf = 0.1        # tiempo final de simulacion
Nf = 100        # numero de puntos de colocacion para entrenamiento
nt = 1200       # numero de pasos de tiempo (numero de PINNs a la vez)

dt = tf/nt

x = torch.linspace(0, 2, nx+1, requires_grad=True, dtype=torch.float32).reshape(-1, 1)

IC_np = np.zeros((nx+1)) 
IC_np[20:30] = 1

IC = torch.tensor(IC_np, dtype=torch.float32).reshape(-1, 1)

u1, u2 = PINN(x)

loss_IC = torch.nn.functional.mse_loss(u1, IC)

du2_dx =  torch.autograd.grad(u2, x,
                torch.ones([x.shape[0], 1]),
                retain_graph=True,
                create_graph=True)[0]
du2_dxx = torch.autograd.grad(du2_dx, x,
            torch.ones([x.shape[0], 1]),
            retain_graph=True,
            create_graph=True)[0]

pde_residual = nu*du2_dxx - (u2-u1)/dt
loss_PDE = torch.mean(pde_residual**2)

loss = loss_IC + loss_PDE

loss.backward()

#%%

nu = 0.3        # parametro nu que acompania a la ecuacion diferencial

epochs = 100000 # numero de epocas de entrenamiento por si no se cumple epsilon
n_prints = 10   # numero de impresiones del loss en cada PINN
n_plots = 1200     # numero de graficas para ver evolucion de la solucion
tol = 1e-2      # epsilon o tolerancia permitida

# create tensors for the domains

nx = 100        # numero de puntos para generar la condicion inicial
nt = 1200       # numero de pasos de tiempo (numero de PINNs a la vez)
tf = 0.1        # tiempo final de simulacion
Nf = 100        # numero de puntos de colocacion para entrenamiento

dt = tf/nt
print(dt)
x = torch.linspace(0, 2, nx+1, dtype=torch.float32)
t = torch.linspace(0, tf, nt+1, dtype=torch.float32)

print("tamaÃ±o de x: ", x.size(),
      "tamaÃ±o de t: ", t.size())

X, T = torch.meshgrid(x, t)

lb = x[0]   # lower bound o cuando x = 0
ub = x[-1]  # upper bound o cuando x = 2

x1 = X[:, 0][:, None].detach() # vector de x en tiempo t = 0


#%% Boundary and Initial Conditions
# Initial condition
x_init = x[:, None].clone().detach().requires_grad_(True) # -> [[]]
y_init = torch.ones(x.shape[0], 1, dtype=torch.float32)
print("tamaÃ±o de y_init: ", y_init.size())
y_init[int(0.25*nx):(int(0.5*nx)+1), 0] = 2.

# se declara una lista de PINNs donde se ira guardando cada PINN
# y otra lista donde se guardan los tiempos de entrenamiento de estos
lista_PINNs = []
train_time = []
ii = 0

# empieza a correr el tiempo de entrenamiento
TIC = time.time()

for j in range(len(t)):
    print("------------------------------\nTime # ", j)
    # Boundary condition
    # f(x=0,t) = 0
    bottom_x = X[0, j][None, None].clone().detach().requires_grad_(True)
    bottom_y = torch.zeros(bottom_x.shape[0], 1, dtype=torch.float32)
    
    # f(x=1,t) = 0
    top_x = X[-1, j][None, None].clone().detach().requires_grad_(True)
    top_y = torch.zeros(bottom_x.shape[0], 1, dtype=torch.float32)
    
    bottom_x.size(), bottom_y.size(),
    top_x.size(), top_y.size()
    
    # se agrupan las condiciones iniciales y condiciones de borde
    X_train = torch.vstack([x_init, bottom_x, top_x])
    Y_train = torch.vstack([y_init, bottom_y, top_y])
        
    # Create the j-th model for the j-th time step
    
    PINN_j = TD_PINNS_Implicit(nu=nu, dt=dt)
    
    # Train model
    # si el modelo no es el primer PINN, pues hereda los parametros del modelo
    # anterior (Transfered Learning)
    if j != 0:
        PINN_j.load_state_dict(lista_PINNs[j-1].state_dict())
        X_train = torch.vstack([X_train_apres, bottom_x, top_x])
        Y_train = torch.vstack([y_train_apres, bottom_y, top_y])
    
    optimizer = torch.optim.Adam(PINN_j.parameters(), lr=1e-3, amsgrad=False)
    
    # empieza un contador de tiempo para cada PINN_j
    tic = time.time()
    
    for i in range(epochs):
        loss = PINN_j.loss(X_train, Y_train)
        optimizer.zero_grad()
        loss.backward()
        if loss.item() <= tol:
            print("epoch until tolerance: ", i)
            break
        optimizer.step()
        
        if i%(epochs/n_prints) == 0:
            print("training: ", round(loss.item(), 7))
    
    # termina el contador de tiempo del PINN_j  
    toc = time.time()
    elapsed_time = toc-tic
    lista_PINNs.append(PINN_j)
    train_time.append(elapsed_time)
    print("Elapsed time: ", elapsed_time, " seconds")
    
    # Advance in time with EULER METHOD
    # apres es equivalente a la solucion "despues" o avanzada
    
    # Latin Hypercube Sampling for collocation points
    lhs_sampler = LatinHypercube(d=1)
    lhs_samples = lhs_sampler.random(n=Nf)
    X_train_apres = torch.tensor(lb + (ub - lb) * lhs_samples, dtype=torch.float32, requires_grad=True)
    
    u1, u2 = PINN_j(X_train_apres)
    u1 = u1.detach()
    f_x = PINN_j.gradiente(X_train_apres).detach()
    y_train_apres = u2.detach() + dt*nu*f_x # en esta linea se avanza la solucion
    
    
    #% Plot
    if j%(nt/n_plots) == 0:
        ii += 1
        print (ii)
        u1, u2 = PINN_j(X[:, j][:, None])
        y1 = u2.detach()
        plt.plot(x1, y1)

TOC = time.time()
TOTAL_TIME = TOC - TIC

plt.xticks(np.linspace(0, 2, 11))
plt.yticks(np.linspace(1, 2, 6))
plt.xlabel("x", fontsize=15)
plt.ylabel("u(x,t)", fontsize=15)
plt.legend(["CondiciÃ³n Inicial", "SoluciÃ³n en t = 0.1"])
plt.title("MÃ©todo GS RÃ©plica 4", fontsize=15)
plt.savefig("DiffMM.svg")

# termina el temporizador y se estima el tiempo de entrenamiento total
print("Total runtime: ", TOTAL_TIME, " seconds")

    


# %%
