import torch
import numpy as np
import matplotlib.pyplot as plt

class _Expr:
    def eval(self, fcn, x_in, constants):
        raise NotImplementedError

    def __add__(self, other): return _BinOp(self, _to_expr(other), op=lambda a,b: a+b)
    def __radd__(self, other): return _BinOp(_to_expr(other), self, op=lambda a,b: a+b)
    def __sub__(self, other): return _BinOp(self, _to_expr(other), op=lambda a,b: a-b)
    def __rsub__(self, other): return _BinOp(_to_expr(other), self, op=lambda a,b: a-b)
    def __mul__(self, other): return _BinOp(self, _to_expr(other), op=lambda a,b: a*b)
    def __rmul__(self, other): return _BinOp(_to_expr(other), self, op=lambda a,b: a*b)
    def __truediv__(self, other): return _BinOp(self, _to_expr(other), op=lambda a,b: a/b)
    def __rtruediv__(self, other): return _BinOp(_to_expr(other), self, op=lambda a,b: a/b)
    def __neg__(self): return _UnOp(self, op=lambda a: -a)

def _to_expr(x):
    if isinstance(x, _Expr): return x
    if isinstance(x, (int,float,np.floating)): return _Scalar(float(x))
    if torch.is_tensor(x): return _TensorConst(x)
    raise TypeError(f"Unsupported type in expression: {type(x)}")

class _Scalar(_Expr):
    def __init__(self, v): self.v = float(v)
    def eval(self, fcn, x_in, constants):
        return torch.full((x_in.shape[0],), self.v, dtype=x_in.dtype, device=x_in.device)

class _TensorConst(_Expr):
    def __init__(self, t): self.t = t
    def eval(self, fcn, x_in, constants):
        t = self.t
        if not torch.is_tensor(t): t = torch.as_tensor(t, dtype=x_in.dtype, device=x_in.device)
        return t.expand(x_in.shape[0]) if t.ndim == 0 else t

class _ConstRef(_Expr):
    def __init__(self, idx): self.idx = idx
    def eval(self, fcn, x_in, constants):
        c = constants[self.idx]
        return c.expand(x_in.shape[0])

class _OutputRef(_Expr):
    def __init__(self, idx): self.idx = idx
    def eval(self, fcn, x_in, constants):
        y = fcn(x_in)                  # (N, out_dim)
        return y[:, self.idx]          # (N,)

class _BinOp(_Expr):
    def __init__(self, a, b, op): self.a, self.b, self.op = a, b, op
    def eval(self, fcn, x_in, constants):
        return self.op(self.a.eval(fcn, x_in, constants),
                       self.b.eval(fcn, x_in, constants))

class _UnOp(_Expr):
    def __init__(self, a, op): self.a, self.op = a, op
    def eval(self, fcn, x_in, constants):
        return self.op(self.a.eval(fcn, x_in, constants))

class _TimeDeriv(_Expr):
    pass

class _SpaceDeriv(_Expr):
    def __init__(self, inner, x_idx): self.inner, self.x_idx = inner, x_idx
    def eval(self, fcn, x_in, constants):
        x_in = x_in.clone().detach().requires_grad_(True)
        val = _ensure_column(self.inner.eval(fcn, x_in, constants))
        grad = torch.autograd.grad(val, x_in,
                                   grad_outputs=torch.ones_like(val),
                                   retain_graph=True,
                                   create_graph=True)[0]
        return grad[:, 1 + self.x_idx]

def _ensure_column(v):
    if v.ndim == 1: return v.unsqueeze(1)
    return v


class TD_PINNS:
    FCN = None
    x = None           
    output = None        
    constants = None     
    derivatives = None   
    time_derivatives = None
    equations = None     

    device = torch.device("cpu")

    @staticmethod
    def set_environment(space_dimensions=1, output_dimensions=1, constants=1,
                        hidden_layers=(64,64,64), device=None):
        if device is not None:
            TD_PINNS.device = torch.device(device)

        TD_PINNS.FCN = FCN(inputs_dim=space_dimensions+1,
                           outputs_dim=output_dimensions,
                           hidden_layers=list(hidden_layers)).to(TD_PINNS.device)

        TD_PINNS.x = [i for i in range(space_dimensions)]
        TD_PINNS.output = [ _OutputRef(i) for i in range(output_dimensions) ]

        TD_PINNS._constants_param = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=TD_PINNS.device))
             for _ in range(constants)]
        )
        TD_PINNS.constants = [ _ConstRef(i) for i in range(constants) ]
        TD_PINNS.constant = TD_PINNS.constants

        TD_PINNS.derivatives = []
        TD_PINNS.time_derivatives = []
        TD_PINNS.equations = []

        TD_PINNS._optimizer = torch.optim.Adam(
            list(TD_PINNS.FCN.parameters()) + list(TD_PINNS._constants_param.parameters()),
            lr=1e-3
        )

    @staticmethod
    def time_derivative(output_or_expr):
        pass

    @staticmethod
    def derivative(output_or_expr, idx_x):

        node = output_or_expr if isinstance(output_or_expr, _Expr) else TD_PINNS.output[output_or_expr]
        TD_PINNS.derivatives.append((node, idx_x))
        return _SpaceDeriv(node, idx_x)

    @staticmethod
    def set_governing_equations(expr_list):
        TD_PINNS.equations = expr_list

    @staticmethod
    def evaluate(t, x):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=TD_PINNS.device)
        if not torch.is_tensor(x):
            x = torch.tensor([x], dtype=torch.float32, device=TD_PINNS.device)
        if x.ndim == 1: x = x.unsqueeze(0)  # (1, D)
        inp = torch.cat([t.view(-1,1).expand(x.shape[0],1), x], dim=1)
        with torch.no_grad():
            y = TD_PINNS.FCN(inp)
        return y.cpu().numpy().squeeze()

    @staticmethod
    def evaluate_expr(expr, t_batch, x_batch):
        if not torch.is_tensor(t_batch):
            t_batch = torch.as_tensor(t_batch, dtype=torch.float32, device=TD_PINNS.device)
        if not torch.is_tensor(x_batch):
            x_batch = torch.as_tensor(x_batch, dtype=torch.float32, device=TD_PINNS.device)
        if t_batch.ndim == 0: t_batch = t_batch.unsqueeze(0)
        if x_batch.ndim == 1: x_batch = x_batch.unsqueeze(0)
        x_in = torch.cat([t_batch.view(-1,1), x_batch], dim=1).to(TD_PINNS.device)
        return expr.eval(TD_PINNS.FCN, x_in, TD_PINNS._constants_param)


    @staticmethod
    def train(collocation=None, epochs=2000, print_every=200):
        if collocation is None:
            
            D = len(TD_PINNS.x)
            tt = torch.rand(256, 1, device=TD_PINNS.device)
            xx = torch.rand(256, D, device=TD_PINNS.device)
            collocation = torch.cat([tt, xx], dim=1)

        if not TD_PINNS.equations:
            print("No governing equations set; nothing to train.")
            return

        mse = torch.nn.MSELoss(reduction="mean")

        for it in range(1, epochs+1):
            TD_PINNS._optimizer.zero_grad()
            x_in = collocation.clone().detach().to(TD_PINNS.device)
            # Sum residuals' MSE to zero
            loss = 0.0
            for expr in TD_PINNS.equations:
                r = expr.eval(TD_PINNS.FCN, x_in, TD_PINNS._constants_param)  # (N,)
                loss = loss + mse(r, torch.zeros_like(r))
            loss.backward()
            TD_PINNS._optimizer.step()

            if it % print_every == 0 or it == 1 or it == epochs:
                with torch.no_grad():
                    consts = [float(c.detach().cpu()) for c in TD_PINNS._constants_param]
                print(f"[{it:05d}] loss={float(loss):.6e}   constants={consts}")

class FCN(torch.nn.Module):
    def __init__(self, inputs_dim=1, outputs_dim=1, hidden_layers=[64,64,64]):
        super().__init__()
        self.activation = torch.nn.Tanh()
        self.loss_function = torch.nn.MSELoss(reduction="mean")
        layers = [inputs_dim] + list(hidden_layers) + [outputs_dim]
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        for lin in self.linears:
            torch.nn.init.xavier_normal_(lin.weight, gain=1.0)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, x):
        if not torch.is_tensor(x): x = torch.from_numpy(x)
        a = x.float()
        for i, lin in enumerate(self.linears):
            a = lin(a)
            if i < len(self.linears)-1:
                a = self.activation(a)
        return a
    
    def gradiente(self, x):
        g = x.clone().detach().requires_grad_(True)
        f = self.forward(g)                     # (N, out_dim)  <-- assumes out_dim == 1 here
        f = f[:, 0:1]
        f_x = torch.autograd.grad(f, g, torch.ones_like(f), retain_graph=True, create_graph=True)[0]
        f_xx = torch.autograd.grad(f_x, g, torch.ones_like(f_x), retain_graph=True, create_graph=True)[0]
        return f_xx

    def lossBC(self, x_BC, y_BC): 
        return self.loss_function(self.forward(x_BC), y_BC)

    def loss(self, x_BC, y_BC):
        return self.lossBC(x_BC, y_BC)
