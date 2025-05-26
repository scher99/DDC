#!/usr/bin/env python3.8


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Define the Physical System (Double Mass-Spring-Damper)
class DoubleMassSpringDamper:
    def __init__(self, m1, m2, k1, k2, b1, b2):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.b1 = b1
        self.b2 = b2

    def equations(self, t, y, F_ext):
        """Defines the system of ODEs."""
        x1, v1, x2, v2 = y
        dx1_dt = v1
        dv1_dt = (F_ext - self.k1 * x1 - self.b1 * v1 + self.k2 * (x2 - x1) + self.b2 * (v2 - v1)) / self.m1
        dx2_dt = v2
        dv2_dt = (-self.k2 * (x2 - x1) - self.b2 * (v2 - v1)) / self.m2
        return [dx1_dt, dv1_dt, dx2_dt, dv2_dt]

    def simulate(self, t_span, initial_state, t_eval, F_ext_func):
        """Simulates the system using solve_ivp."""
        def ode_system(t, y):
            return self.equations(t, y, F_ext_func(t))

        sol = solve_ivp(ode_system, t_span, initial_state, t_eval=t_eval, dense_output=True, method='DOP853')
        return sol.y.T


# 2. Define the Physics-Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, system):
        super(PINN, self).__init__()
        self.system = system
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])  # Input: (time), state, force
        self.linears.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Output: x1, v1, x2, v2

        self.activation = nn.Tanh()

    def forward(self, dt, state, force):
        """Forward pass of the neural network."""
        # state: [batch, 4] current state
        # force: [batch, 1] control input
        # dt:    [batch, 1] time step
        device = next(self.parameters()).device  # Get model's device
        dt = dt.to(device)
        state = state.to(device)
        force = force.to(device)
        x = torch.cat((dt, state, force), dim=1)
        # x = torch.cat((state, force), dim=1)  # Concatenate (time), state, and force
        for linear in self.linears:
            x = self.activation(linear(x))
        return self.output_layer(x)

    def physics_loss(self, t, state, force, state_pred, dt):
        """Calculates the physics-informed loss using the governing equations and auto-differentiation."""
        state  = state.to(state_pred.device)  # Ensure state is on the same device
        dt = dt.to(state_pred.device)  # Ensure dt is on the same device
        force = force.to(state_pred.device)

        x1, v1, x2, v2 = torch.split(state, 1, dim=1)

        x1_pred, v1_pred, x2_pred, v2_pred = torch.split(state_pred, 1, dim=1)

        # # Compute derivatives using automatic differentiation
        # a1_t = torch.autograd.grad(v1_pred, t, grad_outputs=torch.ones_like(v1_pred), create_graph=True, retain_graph=True)[0]
        # a2_t = torch.autograd.grad(v2_pred, t, grad_outputs=torch.ones_like(v2_pred), create_graph=True, retain_graph=True)[0]
        # Compute derivatives using finite differentiation
        a1_t = (v1_pred - v1) / dt
        a2_t = (v2_pred - v2) / dt
        v1_t = (x1_pred - x1) / dt
        v2_t = (x2_pred - x2) / dt

        # Formulate the residual equations
        residual_1 = self.system.m1 * a1_t - (force - self.system.k1 * x1 - self.system.b1 * v1 + self.system.k2 * (x2 - x1) + self.system.b2 * (v2 - v1))
        residual_2 = self.system.m2 * a2_t - (-self.system.k2 * (x2 - x1) - self.system.b2 * (v2 - v1))
        residual_3 = v1_t - v1
        residual_4 = v2_t - v2
        # Physics loss is the mean squared error of the residuals
        loss = torch.mean(residual_1**2) + torch.mean(residual_2**2) + torch.mean(residual_3**2) + torch.mean(residual_4**2)
        return loss

    def data_loss(self, t, state, force, state_target, state_pred):
        state_target = state_target.to(state_pred.device)  # Ensure target is on the same device
        """Calculates the data loss based on the difference between predicted and observed data."""
        loss = torch.mean((state_pred - state_target)**2)
        return loss


# 3. Training the PINN
def train_pinn(pinn, optimizer, t_data, state_data, force_data, state_target_data, \
                epochs, data_loss_weight, physics_loss_weight, dt):
    """Trains the PINN model."""
    pinn.train()
    losses, p_losses, d_losses = [], [], []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Predict the state using the PINN
        state_pred = pinn.forward(dt, state_data, force_data)  # Pass state and force

        data_loss = pinn.data_loss(t_data, state_data, force_data, state_target_data, state_pred)
        physics_loss = pinn.physics_loss(t_data, state_data, force_data, state_pred, dt)

        total_loss = data_loss_weight * data_loss + physics_loss_weight * physics_loss

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        p_losses.append(physics_loss.item())
        d_losses.append(data_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4e}, Data Loss: {data_loss.item():.4e}, Physics Loss: {physics_loss.item():.4e}')
    return losses, p_losses, d_losses 

def inference(system, plot_history=False):
    # Set an initial state
    initial_state_test = torch.tensor(state_data_np[0], dtype=torch.float32).reshape(1,-1)
    num_steps = 100
    predicted_trajectory = []
    current_state = initial_state_test

    # Ground truth for extrapolation
    t_test_np = np.linspace(t_span[0], t_span[0] + 5, num_steps + 1)  # Extrapolate 5 seconds
    t_extrap_span = (t_span[0], t_span[0] + 5)
    t_extrap_eval = t_test_np[1:]
    dt_value = t_test_np[1] - t_test_np[0]

    extrap_solution = system.simulate(t_extrap_span, initial_state_test.numpy().flatten(), t_extrap_eval, F_ext_func)
    x1_true_np = extrap_solution[:, 0]
    v1_true_np = extrap_solution[:, 1]
    x2_true_np = extrap_solution[:, 2]
    v2_true_np = extrap_solution[:, 3]

    pinn.eval()
    with torch.no_grad():
        current_state = initial_state_test
        for i in range(num_steps):
            time_value = torch.tensor(t_test_np[i], dtype=torch.float32).reshape(1, -1).requires_grad_(True)
            dt = torch.tensor(dt_value, dtype=torch.float32).reshape(1,-1)
            force_value = torch.tensor(F_ext_func(t_test_np[i]), dtype=torch.float32).reshape(1, -1)

            next_state_predicted = pinn(dt, current_state, force_value)
            predicted_trajectory.append(next_state_predicted.cpu().numpy().flatten())
            current_state = next_state_predicted

    predicted_trajectory = np.array(predicted_trajectory)

    # Plot Results
    if(plot_history):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(t_eval, x1_data_np, label='Training Data (x1)', color='blue')
        plt.plot(t_test_np[1:], predicted_trajectory[:, 0], label='PINN Prediction (x1)', color='red')
        plt.plot(t_test_np[1:], x1_true_np, label='Extrapolation Truth (x1)', color='green')
        plt.xlabel('Time')
        plt.ylabel('x1')
        plt.title('Mass 1 Displacement (x1)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(t_eval, v1_data_np, label='Training Data (v1)', color='blue')
        plt.plot(t_test_np[1:], predicted_trajectory[:, 1], label='PINN Prediction (v1)', color='red')
        plt.plot(t_test_np[1:], v1_true_np, label='Extrapolation Truth (v1)', color='green')
        plt.xlabel('Time')
        plt.ylabel('v1')
        plt.title('Mass 1 Velocity (v1)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(t_eval, x2_data_np, label='Training Data (x2)', color='blue')
        plt.plot(t_test_np[1:], predicted_trajectory[:, 2], label='PINN Prediction (x2)', color='red')
        plt.plot(t_test_np[1:], x2_true_np, label='Extrapolation Truth (x2)', color='green')
        plt.xlabel('Time')
        plt.ylabel('x2')
        plt.title('Mass 2 Displacement (x2)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(t_eval, v2_data_np, label='Training Data (v2)', color='blue')
        plt.plot(t_test_np[1:], predicted_trajectory[:, 3], label='PINN Prediction (v2)', color='red')
        plt.plot(t_test_np[1:], v2_true_np, label='Extrapolation Truth (v2)', color='green')
        plt.xlabel('Time')
        plt.ylabel('v2')
        plt.title('Mass 2 Velocity (v2)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show(block=False)

# 4. Example Usage
if __name__ == "__main__":
    # Check if GPU is available
    if(torch.cuda.is_available()):  # Check if GPU is available
        print(torch.cuda.device_count())  # Number of GPUs
        print(torch.cuda.get_device_name(0))  # GPU name
        print(torch.cuda.memory_allocated(0) / 1e6, "MB")  # Memory used
        print(torch.cuda.memory_reserved(0) / 1e6, "MB")  # Memory reserved
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f'No GPU available, using the {device} instead.')

    # System parameters
    m1_true = 1.0
    m2_true = 1.0
    k1_true = 10.0
    k2_true = 5.0
    b1_true = 0.5
    b2_true = 0.3
    system = DoubleMassSpringDamper(m1_true, m2_true, k1_true, k2_true, b1_true, b2_true)

    # Define physical parameters (what is assumed known) or what is learnable
    if(True):
        m1 = m1_true       # mass 1
    else:
        m1 = nn.Parameter(torch.tensor(2.0, device=device))
    if(True):
        m2 = m2_true       # mass 2
    else:
        m2 = nn.Parameter(torch.tensor(2.0, device=device))
    if(True):
        k1 = k1_true       # spring stiffness
    else:
        k1 = nn.Parameter(torch.tensor(1.0, device=device))
    if(True):
        b1 = b1_true       # damping coefficient
    else:
        b1 = nn.Parameter(torch.tensor(1.5, device=device))
    if(True):
        k2 = k2_true       # spring stiffness
    else:
        k2 = nn.Parameter(torch.tensor(1.0, device=device))
    if(True):
        b2 = b2_true       # damping coefficient
    else:
        b2 = nn.Parameter(torch.tensor(1.5, device=device))

    # External force function (example: sinusoidal)
    def F_ext_func(t):
        return 2 * np.sin(2 * np.pi * t)

    # Simulation parameters
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 1000, endpoint=True)
    dt = torch.tensor( t_eval[1] - t_eval[0], dtype=torch.float32).reshape(1,-1)
    
    plot_history  = True

    for i in range(2):
        print(f"Training datapoint iteration {i+1}")
        # initial_state = [0.5, 0.0, -0.2, 0.0]
        initial_state = list(10 * np.random.rand(4))  # Random initial state

        # Generate training data by simulating the system
        solution = system.simulate(t_span, initial_state, t_eval, F_ext_func)
        x1_data_np = solution[:, 0]
        v1_data_np = solution[:, 1]
        x2_data_np = solution[:, 2]
        v2_data_np = solution[:, 3]

        # Create state vectors [x1, v1, x2, v2]
        # state_data_np = np.stack([x1_data_np, v1_data_np, x2_data_np, v2_data_np], axis=1)
        state_data_np = np.stack([x1_data_np[:-1], v1_data_np[:-1], x2_data_np[:-1], v2_data_np[:-1]], axis=1) # Current state
        state_next_data_np = np.stack([x1_data_np[1:], v1_data_np[1:], x2_data_np[1:], v2_data_np[1:]], axis=1) # Next state

        # Create force data (external force at each time step)
        force_data_np = np.array([F_ext_func(t) for t in t_eval[:-1]]).reshape(-1, 1)

        # Convert data to torch tensors
        t_data = torch.tensor(t_eval[:-1], dtype=torch.float32).reshape(-1, 1)
        t_data.requires_grad_(True)  # Enable calculating gradients with respect to time
        # create a dt vector in the size of t_eval
        dt = dt*torch.ones_like(t_data)
        state_data = torch.tensor(state_data_np, dtype=torch.float32)
        state_target_data = torch.tensor(state_next_data_np, dtype=torch.float32)
        force_data = torch.tensor(force_data_np, dtype=torch.float32)
        # state_target_data = state_data.clone().detach()  # Train to predict the *same* state

        # PINN parameters
        input_dim = 6 # 6  # Input: (time), state (x1, v1, x2, v2), force
        output_dim = 4  # Output: state (x1, v1, x2, v2)
        hidden_dim = 100 # 64
        num_layers = 5
        learning_rate = 0.0001
        epochs = 10000
        data_loss_weight = 1.0
        physics_loss_weight = .2

        # Create the PINN model
        pinn = PINN(input_dim, output_dim, hidden_dim, num_layers, system).to(device)

        # Load the model's parameters and the learned physical parameters
        try:
            checkpoint = torch.load('pinn_model_state_force.pth')
            pinn.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint['m1'] is not None:
                m1 = checkpoint['m1']
            if checkpoint['m2'] is not None:
                m2 = checkpoint['m2']
            if checkpoint['k1'] is not None:
                k1 = checkpoint['k1']
            if checkpoint['b1'] is not None:
                b1 = checkpoint['b1']
            if checkpoint['k2'] is not None:
                k2 = checkpoint['k2']
            if checkpoint['b2'] is not None:
                b2 = checkpoint['b2']
            print("Model and parameters loaded.")
        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")

        # learnable parameters list
        learnable_params, learnable_params_s = [], []
        if(not isinstance(m1, float)):
            learnable_params.append(m1)
            learnable_params_s.append("m1")
        if(not isinstance(m2, float)):  
            learnable_params.append(m2)
            learnable_params_s.append("m2")
        if(not isinstance(k1, float)):
            learnable_params.append(k1)
            learnable_params_s.append("k1")
        if(not isinstance(b1, float)):
            learnable_params.append(b1)
            learnable_params_s.append("b1")
        if(not isinstance(k2, float)):
            learnable_params.append(k2)
            learnable_params_s.append("k2")
        if(not isinstance(b2, float)):
            learnable_params.append(b2)
            learnable_params_s.append("b2")
        print(f"learnable_params used: {learnable_params_s}={learnable_params}")

        # Count the number of learnable parameters in the model
        total_params = sum(p.numel() for p in pinn.parameters() if p.requires_grad)

        # Add the number of learnable physical parameters
        total_params += sum(p.numel() for p in learnable_params)

        print(f"Total number of learnable parameters: {total_params}")

        # Optimizer
        optimizer = optim.Adam(list(pinn.parameters()) + learnable_params, lr=learning_rate)
        
        # Train the PINN
        loss_history, physics_loss_history, data_loss_history = \
            train_pinn(pinn, optimizer, t_data, state_data, force_data, state_target_data, \
                       epochs, data_loss_weight, physics_loss_weight, dt)

        # Plot Training Loss History
        if(plot_history):
            plt.figure(figsize=(8,5))
            plt.semilogy(loss_history, label='Total Loss')
            plt.semilogy(physics_loss_history, label='Physics Loss')
            plt.semilogy(data_loss_history, label='Data Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Training Loss History iteration:{i+1}")
            plt.grid(True)
            plt.show(block=False)

        # Save the model's parameters and the learned physical parameters
        torch.save({
            'model_state_dict': pinn.state_dict(),
            'm1': m1 if isinstance(m1, nn.Parameter) else None,
            'm2': m2 if isinstance(m2, nn.Parameter) else None,
            'k1': k1 if isinstance(k1, nn.Parameter) else None,
            'b1': b1 if isinstance(b1, nn.Parameter) else None,
            'k2': k2 if isinstance(k2, nn.Parameter) else None,
            'b2': b2 if isinstance(b2, nn.Parameter) else None,
        }, 'pinn_model_state_force.pth')
        print("Model and parameters saved.")

        # # Testing and Iterative Prediction
        # inference(system)

    inference(system, plot_history=True)
    plt.show(block=True)
    