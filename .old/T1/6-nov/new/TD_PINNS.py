import torch
import torch.nn as nn


class TD_PINNS(nn.Module):
    def __init__(self, device, x_dim=1, hid=32, u_dim=1, nu=0.3, dt=0.01):
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
    
        
    def forward(self, x):
        # Ensure input is properly shaped without breaking gradients
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # More gradient-friendly than .view()
        return self.net(x)
    

    def loss_initial(self, X_train, u0_real, x_bc=None, bc=None):
        # Ensure inputs are properly formatted tensors while preserving gradients
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.as_tensor(X_train, device=self.device, dtype=torch.float32)
        X_train = X_train.requires_grad_(True)
        
        if not isinstance(u0_real, torch.Tensor):
            u0_real = torch.as_tensor(u0_real, device=self.device, dtype=torch.float32)
        if u0_real.dim() == 1:
            u0_real = u0_real.unsqueeze(-1)
        
        u0 = self.forward(X_train)
        loss_IC = torch.nn.functional.mse_loss(u0, u0_real)

        if x_bc is not None and bc is not None:
            if not isinstance(x_bc, torch.Tensor):
                x_bc = torch.as_tensor(x_bc, device=self.device, dtype=torch.float32)
            x_bc = x_bc.requires_grad_(True)
            
            if not isinstance(bc, torch.Tensor):
                bc = torch.as_tensor(bc, device=self.device, dtype=torch.float32)
            if bc.dim() == 1:
                bc = bc.unsqueeze(-1)
            
            bc_pred = self.forward(x_bc)
            loss_bc = torch.nn.functional.mse_loss(bc_pred, bc)
        else:
            loss_bc = torch.tensor(0.0, device=self.device, requires_grad=True)

        return 1 * loss_IC + 5 * loss_bc
    

    def loss(self, X_train, u0, x_bc=None, bc=None):
        # Ensure inputs are properly formatted tensors while preserving gradients
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.as_tensor(X_train, device=self.device, dtype=torch.float32)
        X_train = X_train.requires_grad_(True)
        
        if not isinstance(u0, torch.Tensor):
            u0 = torch.as_tensor(u0, device=self.device, dtype=torch.float32)
        if u0.dim() == 1:
            u0 = u0.unsqueeze(-1)

        u1 = self.forward(X_train)
        loss_pde = self.loss_pde(X_train, u0, u1)

        if x_bc is not None and bc is not None:
            if not isinstance(x_bc, torch.Tensor):
                x_bc = torch.as_tensor(x_bc, device=self.device, dtype=torch.float32)
            x_bc = x_bc.requires_grad_(True)
            
            if not isinstance(bc, torch.Tensor):
                bc = torch.as_tensor(bc, device=self.device, dtype=torch.float32)
            if bc.dim() == 1:
                bc = bc.unsqueeze(-1)
            
            bc_pred = self.forward(x_bc)
            loss_bc = torch.nn.functional.mse_loss(bc_pred, bc)
        else:
            loss_bc = torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss_pde + 5 * loss_bc  # CRITICAL FIX: Include PDE loss!


    
    
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
        pde_residual = self.nu * du1_dxx - (u1 - u0) / self.dt
        loss_PDE = torch.mean(pde_residual**2)
        
        # Numerical stability check
        if torch.isnan(loss_PDE) or loss_PDE > 1e8:
            print(f"Warning: Loss PDE is NaN or too large: {loss_PDE.item()}")
            return torch.tensor(1e6, requires_grad=True, device=self.device)
        
        return loss_PDE
    
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



        
