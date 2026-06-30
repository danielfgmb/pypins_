import torch
import torch.nn as nn

from logger import Logger


class TD_PINNS(nn.Module):
    def __init__(self, device, x_dim=1, hid=64, u_dim=1, nu=0.3, dt=0.01):

        super().__init__()
        self.device = device
        self.net = nn.Sequential(
                nn.Linear(x_dim,hid), # entra a la red la IC + x_dim
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
    
        
    def forward(self, x):
        # return self.net(x.view(-1, 1))
        return self.net(x)
    

    def loss_initial(self, X_train, u0_real, x_bc=None, bc=None):
        #X_train = torch.tensor(X_train, requires_grad=True, device=self.device, dtype=torch.float32)
        u0 = self.forward(X_train)
        #u0_real = torch.tensor(u0_real[:,None], requires_grad=True, device=self.device, dtype=torch.float32)
        loss_IC = torch.nn.functional.mse_loss(u0, u0_real)

        residual = u0**2 - u0_real**2

        #if x_bc !=None and bc != None:
        #x_bc = torch.as_tensor(x_bc,  device= self.device, dtype=torch.float32).requires_grad_(True)
        #bc = torch.as_tensor(bc, device= self.device, dtype=torch.float32)[:,None]
        bc_pred = self.forward(x_bc)

        loss_bc = torch.nn.functional.mse_loss(bc_pred,bc)

        return 10 * loss_IC + 10 * loss_bc, loss_IC, loss_bc, residual
    

    def loss_initial_2(self,X_train,du_dt,du_dtt):
        pass

    def loss(self, X_train, u0, x_bc=None, bc=None):
        #X_train = torch.as_tensor(X_train, dtype=torch.float32).requires_grad_(True)
        #u0      = torch.as_tensor(u0,      dtype=torch.float32, device=self.device).view(-1, 1)
        #X_train = X_train.clone()
        #X_train.requires_grad=True


        u1 = self.forward(X_train)
        
        
        # print(f" shapes u0 {u0.shape} u2 {u1.shape}")

        loss_pde, pde_residual = self.loss_pde(X_train, u0, u1)

        #if x_bc !=None and bc != None:
        #if x_bc !=None and bc != None:
        #x_bc = torch.as_tensor(x_bc,  device= self.device, dtype=torch.float32).requires_grad_(True)
        #bc = torch.as_tensor(bc, device= self.device, dtype=torch.float32)[:,None]
        bc_pred = self.forward(x_bc)

        loss_bc = torch.nn.functional.mse_loss(bc_pred,bc)
        #else:
        #    loss_bc = 0

        return 1/10 * loss_pde + 10*loss_bc, loss_pde, loss_bc, pde_residual #+ loss_pde


    
    
    def loss_pde(self, X_train, u0, u1):
    
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
        pde_residual =  (u1 - u0) / self.dt - self.nu * du1_dxx 
        loss_PDE = torch.mean(pde_residual**2)

        
        # Numerical stability check
        if torch.isnan(loss_PDE) or loss_PDE > 1e8:
            print(f"Warning: Loss PDE is NaN or too large: {loss_PDE.item()}")
            # return torch.tensor(1e6, requires_grad=True, device=self.device)
        
        return loss_PDE, pde_residual
    
    def grad(self, X):

        X.requires_grad_(True)

        u = self.forward(X)

        du1_dx = torch.autograd.grad(
            u, X,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        du1_dxx = torch.autograd.grad(
            du1_dx, X,
            grad_outputs=torch.ones_like(du1_dx),
            retain_graph=True,
            create_graph=True
        )[0]

        return du1_dx, du1_dxx



        
