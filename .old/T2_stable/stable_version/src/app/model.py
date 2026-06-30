import torch
import torch.nn as nn


# ── Built-in PDE loss functions ────────────────────────────────────────────

def burgers_1d_loss(model, X, u0, u1):
    # 1-D viscous Burgers: du/dt + u*du/dx = nu*d2u/dx2
    du_dx = torch.autograd.grad(
        u1, X, grad_outputs=torch.ones_like(u1),
        retain_graph=True, create_graph=True)[0]
    du_dxx = torch.autograd.grad(
        du_dx, X, grad_outputs=torch.ones_like(du_dx),
        retain_graph=True, create_graph=True)[0]
    res = (u1 - u0) / model.dt + u1 * du_dx - model.nu * du_dxx
    return torch.mean(res**2), res


def burgers_2d_loss(model, X, u0, u1):
    # 2-D viscous Burgers: du/dt + u*(du/dx + du/dy) = nu*(d2u/dx2 + d2u/dy2)
    g = torch.autograd.grad(
        u1, X, grad_outputs=torch.ones_like(u1),
        retain_graph=True, create_graph=True)[0]
    du_dx, du_dy = g[:, 0:1], g[:, 1:2]
    du_dxx = torch.autograd.grad(
        du_dx, X, grad_outputs=torch.ones_like(du_dx),
        retain_graph=True, create_graph=True)[0][:, 0:1]
    du_dyy = torch.autograd.grad(
        du_dy, X, grad_outputs=torch.ones_like(du_dy),
        retain_graph=True, create_graph=True)[0][:, 1:2]
    res = (u1 - u0) / model.dt + u1 * (du_dx + du_dy) - model.nu * (du_dxx + du_dyy)
    return torch.mean(res**2), res


# ── Network ────────────────────────────────────────────────────────────────

class TD_PINNS(nn.Module):
    # Time-Decomposition PINN.
    # x_dim       : spatial dimension (1 or 2)
    # pde_loss_fn : callable(model, X, u0, u1) -> (scalar_loss, residual)
    #               Defaults to the built-in Burgers loss for x_dim.

    def __init__(self, device, x_dim=1, hid=64, u_dim=1,
                 nu=0.3, dt=0.01, pde_loss_fn=None):
        super().__init__()
        self.device = device
        self.x_dim  = x_dim
        self.nu, self.dt = nu, dt
        self.net = nn.Sequential(
            nn.Linear(x_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid),   nn.Tanh(),
            nn.Linear(hid, hid),   nn.Tanh(),
            nn.Linear(hid, u_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.)
                nn.init.zeros_(m.bias)
        self.pde_loss_fn = pde_loss_fn or (
            burgers_1d_loss if x_dim == 1 else burgers_2d_loss
        )
        self.to(device)

    def forward(self, x):
        return self.net(x)

    def loss_initial(self, X, u0_real, x_bc, bc):
        u0       = self.forward(X)
        loss_IC  = nn.functional.mse_loss(u0, u0_real)
        residual = u0**2 - u0_real**2
        loss_bc  = nn.functional.mse_loss(self.forward(x_bc), bc)
        return 10 * loss_IC + 10 * loss_bc, loss_IC, loss_bc, residual

    def loss(self, X, u0, x_bc, bc):
        u1 = self.forward(X)
        loss_pde, residual = self.pde_loss_fn(self, X, u0, u1)
        loss_bc = nn.functional.mse_loss(self.forward(x_bc), bc)
        return 0.1 * loss_pde + 10 * loss_bc, loss_pde, loss_bc, residual

    def grad(self, X):
        # Compute spatial gradients for adaptive collocation strategies.
        # 1-D: returns (du_dx [N,1], du_dxx [N,1])
        # 2-D: returns (grad_u [N,2], laplacian [N,1])
        X = X.detach().requires_grad_(True)
        u = self.forward(X)
        if self.x_dim == 1:
            du_dx = torch.autograd.grad(
                u, X, grad_outputs=torch.ones_like(u),
                retain_graph=True, create_graph=True)[0]
            du_dxx = torch.autograd.grad(
                du_dx, X, grad_outputs=torch.ones_like(du_dx),
                retain_graph=True, create_graph=True)[0]
            return du_dx, du_dxx
        else:
            g = torch.autograd.grad(
                u, X, grad_outputs=torch.ones_like(u),
                retain_graph=True, create_graph=True)[0]
            du_dx, du_dy = g[:, 0:1], g[:, 1:2]
            du_dxx = torch.autograd.grad(
                du_dx, X, grad_outputs=torch.ones_like(du_dx),
                retain_graph=True, create_graph=True)[0][:, 0:1]
            du_dyy = torch.autograd.grad(
                du_dy, X, grad_outputs=torch.ones_like(du_dy),
                retain_graph=True, create_graph=True)[0][:, 1:2]
            return g, du_dxx + du_dyy
