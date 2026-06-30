import torch
from TD_PINNS import TD_PINNS
import time
import numpy as np
from logger import Logger

# Assumes boundaries x[0] and x[-1]

METHOD = "residual"

logger = Logger()

class TrainerDiffusion1D:

    def __init__(self, ):
        self.device = self.optimize_device()
        self.analytic_solution_func = None


    def set_simulation_parameters(self, nu=5.0, t0=0.01, dt=0.001, nt=10, x_bc=torch.Tensor([-3,3])):
        self.nu = nu
        self.t0 = t0
        self.dt = dt
        self.nt = nt
        self.x_test= torch.linspace(x_bc[0], x_bc[1], 1000)
        self.x_bc = x_bc


    def set_training_parameters(self, tol=1e-2, tol_ic=None, max_epochs=300000, n_prints_per_training=3000, PINNs_trained_per_plot=1, lr=1e-2,min_lr=1e-8):

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
            self.bc_initial = bc_initial
            self.u_initial = u_initial
        elif func:
            self.u_initial = func(t=self.t0,x=x_initial) # esta función sería dependiente del tiempo y x donde x siempre es un tensor
            self.bc_initial = func(t=self.t0,x=self.x_bc)

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
    
    def train_td_implicit_2(self,):

        self.PINNS_list = []

        self.PINNS_times = []

        self.PINNS_x = []

        PINN_old = None

        for i in range(1,self.nt): # para que inicie en t= dt*1

            print("TRAIN ITERATION =========================")



            t = self.t0 + i * self.dt

            print(f"iteration i {i} t {t} ")

            start = time.perf_counter()

            if i == 1:
                u_0 = self.u_initial
                x_i = self.x_initial.reshape(-1, 1)
            else:
                u_0 = PINN_old.forward(x_i)
                x_i = self.func_distribution(PINN_old,t,pde_residual).reshape(-1, 1)

            PINN_i, pde_residual = self.train_iteration(x_i, u_0, tol=self.tol, previous_PINN=PINN_old, x_bc=self.x_bc, bc=self.boundary_func(t=t,x=self.x_bc))
            
            end = time.perf_counter()
            elapsed = end - start

            self.PINNS_list.append(PINN_i)
            self.PINNS_times.append(elapsed)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(x_i.detach().numpy(), PINN_i.forward(x_i).detach().numpy(), 'o-', markersize=2)
            plt.plot(x_i.detach().numpy(), self.boundary_func(t,x_i.detach().numpy()))
            plt.scatter(self.x_bc.detach().numpy(),self.boundary_func(t=t,x=self.x_bc))
            plt.title(f'u(x, t={t:.3f})')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.grid(True)
            plt.show()

            plt.figure()
            plt.plot(x_i, np.zeros(x_i.size()), 'o-', markersize=2)
            plt.title(f'Point Distribution')
            plt.xlabel('x')
            plt.grid(True)
            plt.show()

            logger.add_data_series([METHOD] *1000, [t] * 1000, self.boundary_func(t=t,x=self.x_test).detach().numpy(), PINN_i.forward(self.x_test.reshape(-1, 1)).detach().numpy(),[self.dt] * 1000 ,[self.nu]*1000,[elapsed] * 1000)
            #logger.add_collocation_data([METHOD]*1000,x_i.detach().numpy(),t)

            

            #print(self.func_distribution(PINN_old,t,pde_residual).detach().numpy(),self.x_bc.detach().numpy(),self.boundary_func(t=t,x=self.x_bc))

            PINN_old = PINN_i

            print("END ITERATION =========================")


    


    def train_td_implicit(self,):

        self.PINNS_list = []

        self.PINNS_times = []

        self.PINNS_x = []

        PINN_initial, pde_residual = self.train_iteration(self.x_initial, self.u_initial, initial_condition=True, tol=self.tol_ic, x_bc=self.x_bc, bc=self.bc_initial)

        self.PINNS_list.append(PINN_initial)

        PINN_old = PINN_initial

        for i in range(1,self.nt): # para que inicie en t= dt*1

            t = self.t0 + i * self.dt

            start = time.perf_counter()


            x_i = self.func_distribution(PINN_old,t,pde_residual).reshape(-1, 1)

            u_0 = PINN_old.forward(x_i)

            # debug
            # debug
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(x_i.detach().numpy(), u_0.detach().numpy(), 'o-', markersize=2)
            plt.plot(x_i.detach().numpy(), self.boundary_func(t-self.dt,x_i.detach().numpy()))
            plt.title(f'u₀(x, t={t-self.dt:.3f})')
            plt.xlabel('x')
            plt.ylabel('u₀')
            plt.grid(True)
            plt.show()

            plt.figure()
            plt.plot(x_i, np.zeros(x_i.size()), 'o-', markersize=2)
            plt.title(f'Point Distribution')
            plt.xlabel('x')
            plt.grid(True)
            plt.show()

            

            PINN_i, pde_residual = self.train_iteration(x_i, u_0, tol=self.tol, previous_PINN=PINN_old, x_bc=self.x_bc, bc=self.boundary_func(t=t,x=self.x_bc))
            
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

        x_train = x_train.reshape(-1, 1)   
        u_train = u_train.reshape(-1,1)
        x_bc = x_bc.reshape(-1, 1)
        bc = bc.reshape(-1, 1)

        x_train.requires_grad_(True)
        x_bc.requires_grad_(True)

        if initial_condition:
            # we compare with the same time
            u1_train = u_train
        else:
            # we compare with the previous time
            u0_train = u_train

        # print(f"train_iterationm nu {self.nu} dt {self.dt}")

        PINN = TD_PINNS(nu=self.nu, dt=self.dt, device=self.device)


        optimizer = torch.optim.AdamW(PINN.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=300, factor=0.5, 
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
                        loss, loss_pde, loss_bc, pde_residual = PINN.loss_initial(x_train, u1_train, x_bc, bc)
                    else:
                        loss, loss_pde, loss_bc, pde_residual = PINN.loss(x_train, u0_train, x_bc, bc)
                scaler.scale(loss).backward(retain_graph=True)
            else:
                if initial_condition:
                    
                    loss, loss_pde, loss_bc, pde_residual = PINN.loss_initial(x_train, u1_train, x_bc, bc)
                else:
                    loss, loss_pde, loss_bc, pde_residual = PINN.loss(x_train, u0_train, x_bc, bc)
                loss.backward(retain_graph=True)
            
            # Check convergence
            loss_value = loss.item()
            if loss_value <= tol:
                print(f"training: {loss_value:.7f}, LossPDE: {loss_pde}, LossBC: {loss_bc},LR = {optimizer.param_groups[0]['lr']:.2e}")
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
                print(f"training: {loss_value:.7f}, LossPDE: {loss_pde}, LossBC: {loss_bc},LR = {optimizer.param_groups[0]['lr']:.2e}")


            # debug
            #print(f"training: {loss_value:.7f}, LR = {current_lr:.2e}")
            #print(x_train, u1_train, x_bc, bc)

        # termina el contador de tiempo del PINN  
        toc = time.time()

        # print("lol")

        return PINN, pde_residual

"""
        


x = torch.linspace(0, 2, nx+1).to(device)
t = torch.linspace(0, tf, nt+1).to(device)


"""

