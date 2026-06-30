import torch
import torch.nn as nn

from logger import Logger


class TD_PINNS(nn.Module):
    def __init__(self, device, x_dim=2, hid=64, u_dim=1, nu=0.3, dt=0.01):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.net = nn.Sequential(
                nn.Linear(x_dim, hid),
                nn.Tanh(),
                nn.Linear(hid, hid),
                nn.Tanh(),
                nn.Linear(hid, hid),
                nn.Tanh(),
                nn.Linear(hid, u_dim)
            )
        
        self.nu = nu
        self.dt = dt
        
        for module in self.net:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight.data, gain=1.)
                torch.nn.init.zeros_(module.bias.data)
        
        self.to(device)
    
        
    def forward(self, x):
        return self.net(x)
    

    def loss_initial(self, X_train, u0_real, x_bc=None, bc=None):
        u0 = self.forward(X_train)
        loss_IC = torch.nn.functional.mse_loss(u0, u0_real)

        residual = u0**2 - u0_real**2

        bc_pred = self.forward(x_bc)

        loss_bc = torch.nn.functional.mse_loss(bc_pred,bc)

        return 10 * loss_IC + 10 * loss_bc, loss_IC, loss_bc, residual
    

    def loss_initial_2(self,X_train,du_dt,du_dtt):
        pass

    def loss(self, X_train, u0, x_bc=None, bc=None):
        u1 = self.forward(X_train)

        loss_pde, pde_residual = self.loss_pde(X_train, u0, u1)

        bc_pred = self.forward(x_bc)

        loss_bc = torch.nn.functional.mse_loss(bc_pred,bc)

        return 1/10 * loss_pde + 10*loss_bc, loss_pde, loss_bc, pde_residual

    
    def loss_pde(self, X_train, u0, u1):
        grad_u1 = torch.autograd.grad(
            u1, X_train,
            grad_outputs=torch.ones_like(u1),
            retain_graph=True,
            create_graph=True
        )[0]
        
        du1_dx = grad_u1[:, 0:1]
        du1_dy = grad_u1[:, 1:2]

        du1_dxx = torch.autograd.grad(
            du1_dx, X_train,
            grad_outputs=torch.ones_like(du1_dx),
            retain_graph=True,
            create_graph=True
        )[0][:, 0:1]

        du1_dyy = torch.autograd.grad(
            du1_dy, X_train,
            grad_outputs=torch.ones_like(du1_dy),
            retain_graph=True,
            create_graph=True
        )[0][:, 1:2]

        laplacian = du1_dxx + du1_dyy

        convection = u1 * (du1_dx + du1_dy)

        pde_residual = (u1 - u0) / self.dt + convection - self.nu * laplacian
        loss_PDE = torch.mean(pde_residual**2)

        if torch.isnan(loss_PDE) or loss_PDE > 1e8:
            print(f"Warning: Loss PDE is NaN or too large: {loss_PDE.item()}")
        
        return loss_PDE, pde_residual
    
    def grad(self, X):
        X.requires_grad_(True)

        u = self.forward(X)

        grad_u = torch.autograd.grad(
            u, X,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        du_dx = grad_u[:, 0:1]
        du_dy = grad_u[:, 1:2]

        du_dxx = torch.autograd.grad(
            du_dx, X,
            grad_outputs=torch.ones_like(du_dx),
            retain_graph=True,
            create_graph=True
        )[0][:, 0:1]

        du_dyy = torch.autograd.grad(
            du_dy, X,
            grad_outputs=torch.ones_like(du_dy),
            retain_graph=True,
            create_graph=True
        )[0][:, 1:2]

        laplacian = du_dxx + du_dyy

        return grad_u, laplacian
