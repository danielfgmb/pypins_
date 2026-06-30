import torch
import matplotlib.pyplot as plt
import numpy as np
import time

# Time-Discrete PINNS
class TD_PINNS():

    def set_environment(space_dimensions=1, output_dimensions=1, constants=1):

        TD_PINNS.FCN = FCN(space_dimensions+1, output_dimensions) # el +1 es por que los inputs son el tiempop más el espacio

        TD_PINNS.x = [ i for i in range(space_dimensions) ]

        TD_PINNS.output = [ i for i in range(output_dimensions) ]

        TD_PINNS.constants = [ i for i in range(constants) ]

        TD_PINNS.derivatives = []

        TD_PINNS.time_derivatives = []

    def time_derivative(idx_output):
        
        TD_PINNS.time_derivatives.append(idx_output)

    def derivative(idx_output,idx_x):

        TD_PINNS.derivatives.append((idx_output,idx_x))

    # el
    





class FCN(torch.nn.Module):
    def __init__(self, inputs_dim=1, outputs_dim=1, hidden_layers=[10,10,10]):
        super().__init__()
        self.activation = torch.nn.Tanh()
        self.loss_function = torch.nn.MSELoss(reduction="mean")
        self.layers = [inputs_dim] + hidden_layers + [outputs_dim]
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(hidden_layers)-1)])
        for i in range(len(self.layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.)
            torch.nn.init.zeros_(self.linears[i].bias.data)
        
    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def grad(self, x):
        g = x.clone()
        g.requires_grad=True
        f = self.forward(g)
        f_x = torch.autograd.grad(f, g,
                                  torch.ones([g.shape[0], 1]),
                                  retain_graph=True,
                                  create_graph=True)[0]
        f_xx = torch.autograd.grad(f_x, g,
                                  torch.ones([g.shape[0], 1]),
                                  retain_graph=True,
                                  create_graph=True)[0]
        return f_x, f_xx
    
    def lossBC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)
        return loss_BC
    
    def loss(self, x_BC, y_BC):
        return self.lossBC(x_BC, y_BC)




