import torch
from TD_PINNS import TD_PINNS
import time
import numpy as np

# Assumes boundaries x[0] and x[-1]

class TrainerDiffusion1D:

    def __init__(self, ):
        self.device = self.optimize_device()
        self.analytic_solution_func = None


    def set_simulation_parameters(self, nu=0.1, t0=0.01, dt=0.01, nt=10, x_bc=torch.Tensor([-3,3])):
        self.nu = nu
        self.t0 = t0
        self.dt = dt
        self.nt = nt

        self.x_bc = x_bc


    def set_training_parameters(self, tol=1e-3, tol_ic=None, max_epochs=300000, n_prints_per_training=3000, PINNs_trained_per_plot=1, lr=1e-2,min_lr=1e-8):

        self.tol = tol
        if tol_ic:
            self.tol_ic = tol_ic
        else:
            self.tol_ic = tol
        self.max_epochs = max_epochs
        self.n_prints_per_training = n_prints_per_training
        self.PINNs_trained_per_plot = PINNs_trained_per_plot
        self.lr = lr
        self.min_lr = min_lr


    def set_point_collocation_distribution(self,func_distribution):
        self.func_distribution = func_distribution # func that receives, func(PINN, t)
    

    def set_initial_condition(self,x_initial,u_initial=None, bc_initial=None,func=None):
        
        if (u_initial or bc_initial) and func:
            raise("O es función o es cond inicial")

        self.x_initial = x_initial
        if u_initial and bc_initial:
            self.bc_initial = np.asarray(bc_initial, dtype=np.float32)
            self.u_initial = np.asarray(u_initial, dtype=np.float32)
        elif func:
            self.u_initial = np.asarray(func(t=self.t0,x=x_initial), dtype=np.float32) # esta función sería dependiente del tiempo y x donde x siempre es un tensor
            self.bc_initial = np.asarray(func(t=self.t0,x=self.x_bc), dtype=np.float32)

    def set_boundary_function(self,func):

        self.boundary_func = func  # esta función sería dependiente del tiempo y x, donde x siempre es un tensor


    
    def set_device(self,device_name):
        self.device = torch.device(device_name)
        return self.device
        

    def optimize_device(self):
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

        self.device = device

        return device
    


    def train_td_implicit(self,):

        self.PINNS_list = []

        self.PINNS_times = []

        self.PINNS_x = []

        PINN_initial = self.train_iteration(self.x_initial, self.u_initial, initial_condition=True, tol=self.tol_ic, x_bc=self.x_bc, bc=self.bc_initial)

        self.PINNS_list.append(PINN_initial)

        PINN_old = PINN_initial

        for i in range(1,self.nt): # para que inicie en t= dt*1

            t = self.t0 + i * self.dt

            start = time.perf_counter()



            x_i = self.func_distribution(PINN_old,t)
            # Ensure x_i is float32 tensor
            if isinstance(x_i, torch.Tensor):
                x_i = x_i.to(dtype=torch.float32)
            else:
                x_i = torch.as_tensor(x_i, dtype=torch.float32)

            u_0 = PINN_old.forward(torch.as_tensor(x_i, dtype=torch.float32).requires_grad_(False)).detach().cpu().numpy().astype(np.float32)

            # debug plots removed to avoid blocking execution

            bc_values = self.boundary_func(t=t,x=self.x_bc)
            if isinstance(bc_values, np.ndarray):
                bc_values = bc_values.astype(np.float32)
            PINN_i = self.train_iteration(x_i, u_0, previous_PINN=PINN_old, x_bc=self.x_bc, bc=bc_values)
            
            end = time.perf_counter()
            elapsed = end - start

            self.PINNS_list.append(PINN_i)
            self.PINNS_times.append(elapsed)

            PINN_old = PINN_i


    def recreate_solution(self,x0,xf,nx):
        
        x = torch.linspace(x0, xf, steps=nx, device=self.device)

        t = torch.linspace(self.t0, self.t0+self.dt*self.nt, steps=self.nt, device=self.device)

        sol = np.zeros([nx,self.nt])

        for i in range(len(self.PINNS_list)):
            PINN = self.PINNS_list[i]
            u_i = PINN.forward(x.requires_grad_(False)).cpu().numpy()
            sol[:,i] = u_i.cpu().numpy()

        return x, t, sol, self.PINNS_times




    def train_iteration(self, x_train, u_train, previous_PINN = None, initial_condition = False, tol=1e-3, x_bc=None, bc=None):

        if initial_condition:
            # we compare with the same time
            u1_train = np.asarray(u_train, dtype=np.float32)
        else:
            # we compare with the previous time
            u0_train = np.asarray(u_train, dtype=np.float32)

        PINN = TD_PINNS(nu=self.nu, dt=self.dt, device=self.device)


        optimizer = torch.optim.AdamW(PINN.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5, 
                                                            min_lr=self.min_lr)

        # Mixed precision training setup
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        use_amp = torch.cuda.is_available()  # Use AMP only with CUDA for now


        if previous_PINN:
            PINN.load_state_dict(previous_PINN.state_dict())

        for i in range(self.max_epochs):
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    if initial_condition:
                        loss = PINN.loss_initial(x_train, u1_train, x_bc, bc)
                    else:
                        loss = PINN.loss(x_train, u0_train, x_bc, bc)
                scaler.scale(loss).backward()
            else:
                if initial_condition:
                    loss = PINN.loss_initial(x_train, u1_train, x_bc, bc)
                else:
                    loss = PINN.loss(x_train, u0_train, x_bc, bc)
                loss.backward()
            
            # Check convergence
            loss_value = loss.item()
            if loss_value <= tol:
                print(f"training: {loss_value:.7f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
                print(f"epoch until tolerance: {i}")
                
                break
            
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # Step scheduler (reduce LR on plateau)
            scheduler.step(loss_value)
            
            # Memory cleanup every 500 steps
            if i % 500 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            if i % (self.max_epochs // self.n_prints_per_training) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"training: {loss_value:.7f}, LR = {current_lr:.2e}")

            # debug
            #print(f"training: {loss_value:.7f}, LR = {current_lr:.2e}")
            #print(x_train, u1_train, x_bc, bc)

        # termina el contador de tiempo del PINN  
        toc = time.time()

        return PINN

"""
        


x = torch.linspace(0, 2, nx+1).to(device)
t = torch.linspace(0, tf, nt+1).to(device)


"""

