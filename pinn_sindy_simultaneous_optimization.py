#!/usr/bin/env python3.10

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. System Definition (Double Mass-Spring-Damper) - Unchanged
def double_mass_spring_damper(t, z, m1, m2, k1, k2, b1, b2):
    """
    Defines the equations of motion for a double mass-spring-damper system.
    z = [x1, x1_dot, x2, x2_dot]
    """
    x1, x1_dot, x2, x2_dot = z
    u_val = np.sin(2 * np.pi * t)  # this acts as a force that we know
    x1_ddot = ( -b1*x1_dot - k1*x1 + k2*(x2-x1) + b2*(x2_dot - x1_dot) + u_val) / m1
    x2_ddot = ( -b2*(x2_dot - x1_dot) - k2*(x2 - x1) ) / m2
    return [x1_dot, x1_ddot, x2_dot, x2_ddot]

# Define forcing input function (for mass 1)
def force(t):
    # Example: a sinusoidal force
    return 1.0 * torch.sin(2.0 * np.pi * t)

# 2. Data Generation - Unchanged
def generate_data(m1, m2, k1, k2, b1, b2, initial_state, t_span, t_eval, noise_std):
    """Generates training data by solving the ODE and adding noise."""
    sol = solve_ivp(double_mass_spring_damper, t_span, initial_state, t_eval=t_eval,
                    args=(m1, m2, k1, k2, b1, b2), rtol=1e-8, atol=1e-8)
    x_data = sol.y.T + np.random.normal(0, noise_std, sol.y.T.shape)  # Add noise
    t_data = sol.t
    return t_data, x_data

# 3. Neural Network (PINN) - Modified to include Xi as a parameter
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_functions):  # Added num_functions
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.activation = nn.Tanh() #or nn.SiLU()
        # Initialize weights (optional but often helpful)
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize SINDy coefficient matrix (Xi) - now a trainable parameter
        # TODO: initialize this parameter matrix with what we think we know
        self.Xi = nn.Parameter(torch.randn(num_functions, output_dim) * 0.01) # Small random initialization

    def forward(self, t, u):
        # t the time vector shape [N, 1]
        # u the force input shape [N, 1]
        # x = t
        x = torch.cat((t, u), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


# 4. SINDy Library (Candidate Functions) - Unchanged
def sindy_library(x, u, poly_order=2, include_sine=False):
    """
    Generates the SINDy library of candidate functions.
    x:  A tensor of shape (n_samples, state_dim)
    u:  the input forces
    poly_order:  supports only 2 at this point
    """
    generalized_x = torch.cat((x, u), dim=1)
    n_samples = x.shape[0]
    state_dim = generalized_x.shape[1]

    library = [torch.ones(n_samples, 1)]  # Always include a constant term

    # Polynomial terms
    for i in range(state_dim):
        library.append(generalized_x[:, i:i+1]) # Add each state variable (x1, x1_dot, x2, x2_dot, u)

    if poly_order > 1:
        for i in range(state_dim):
            library.append(generalized_x[:, i:i+1]**2)  #Quadratic terms

        if state_dim > 1:
            for i in range(state_dim):
                for j in range(i + 1, state_dim):
                    library.append(generalized_x[:, i:i+1] * generalized_x[:, j:j+1])  # Cross terms

    if include_sine:
        for i in range(state_dim):
            library.append(torch.sin(generalized_x[:, i:i+1])) #Sin terms

    return torch.cat(library, dim=1) # Concatenate the library functions into a matrix


# 5. Loss Functions - Modified to use the NN's internal Xi
def compute_losses(net, t_data, x_data, u_data, t_collocation, x_collocation,
                   u_collocation, poly_order = 2, include_sine = False):
    """Computes the data fidelity, SINDy residual (physics), and sparsity losses."""

    # data loss - only at data points
    x_nn_collocation = net(t_collocation, u_collocation)  # PINN prediction at data points
    L_data = torch.mean((x_nn_collocation - x_collocation)**2)

    # physics loss - at all possible points
    x_nn = net(t_data, u_data) #PINN predictions at collocation points
    dx_nn_dt = torch.autograd.grad(x_nn, t_data,
                                    grad_outputs=torch.ones_like(x_nn),
                                    create_graph=True, retain_graph=True)[0]
    
    # send x_data or x_nn to the sindy_library? need to explore this
    theta = sindy_library(x_nn, u_data, \
                          poly_order=poly_order, include_sine=include_sine)
    # Compute the physical (SINDy) residual
    L_sindy_res = torch.mean((dx_nn_dt - theta @ net.Xi)**2) # Uses net.Xi

    # L_sparsity = torch.sum(torch.abs(net.Xi))  #L1 regularization
    # Use a smooth L1 approximation (Huber loss) for better gradient behavior
    L_sparsity = torch.sum(nn.functional.huber_loss(net.Xi, torch.zeros_like(net.Xi), reduction='sum', delta=1.0))

    return L_data, L_sindy_res, L_sparsity


# 6. Training Loop (Simultaneous Optimization) - Simplified
def train(net, t_data, x_data, u_data, t_collocation, x_collocation, u_collocation, optimizer, num_epochs,
          w_data, w_res, w_sparse, poly_order = 2, include_sine = False):
    """Trains the PINN and SINDy coefficients simultaneously."""
    net.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        L_data, L_sindy_res, L_sparsity = \
            compute_losses(net, t_data, x_data, u_data, \
                           t_collocation, x_collocation, u_collocation, poly_order, include_sine)
        L_total = w_data * L_data + w_res * L_sindy_res + w_sparse * L_sparsity
        L_total.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Data Loss: {L_data.item():.4f}, '
                  f'Residual Loss: {L_sindy_res.item():.4f}, Sparsity: {L_sparsity.item():.4f}')


# 7. Main Training Loop & Visualization
if __name__ == '__main__':
    # System Parameters (Adjust these to change the system) 
    dt = 0.01
    m1, m2 = 1.0, 1.0
    k1, k2 = 1.0, 1.0
    b1, b2 = 0.1, 0.1
    initial_state = [1.0, 0.0, 0.5, 0.0]  # [x1, x1_dot, x2, x2_dot]
    t_span = [0, 10]
    t_coll = [0, 7]  # Collocation points for physics loss
    t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]/dt))  # Data points
    noise_std = 0.01

    # Generate Data - Unchanged
    t_data, x_data = generate_data(m1, m2, k1, k2, b1, b2, initial_state, t_span, t_eval, noise_std)

    # Convert to PyTorch tensors - Unchanged
    t_data = torch.tensor(t_data[:, None], dtype=torch.float32, requires_grad=True) # [:, None] adds a dimension
    x_data = torch.tensor(x_data, dtype=torch.float32)
    u_data = force(t_data)  # External input (forcing function)
    u_data = torch.tensor(u_data, dtype=torch.float32, requires_grad=True)  # Convert to tensor

    # Collocation Points (for physics loss) - Unchanged
    t_collocation = torch.linspace(t_coll[0], t_coll[1], int(t_coll[1]/dt), requires_grad=True).reshape(-1, 1)
    u_collocation = force(t_collocation)  # External input (forcing function)
    u_collocation = torch.tensor(u_collocation, dtype=torch.float32, requires_grad=True)  # Convert to tensor
    x_collocation = torch.tensor(x_data[:len(t_collocation), :], dtype=torch.float32)  # Convert to tensor

    # PINN Parameters
    input_dim = 2  # Time, external inputs
    output_dim = 4  # [x1, x1_dot, x2, x2_dot]
    hidden_dim = 32
    num_layers = 4
    learning_rate = 1e-4
    num_epochs = 20000 # Training epochs

    # SINDy Parameters
    poly_order = 2 #Order of polynomial functions for SINDy library
    include_sine = False

    # Loss Weights
    w_data = 1.0
    w_res = 0.1 #Physics loss weight
    w_sparse = 0.1  #Sparsity weight

    # Initialize PINN and Optimizer
    num_functions = sindy_library(x_data, u_data, poly_order, include_sine).shape[1]  # Determine size of library
    net = PINN(input_dim, output_dim, hidden_dim, num_layers, num_functions) #Added num_functions argument
    optimizer = optim.Adam(net.parameters(), lr=learning_rate) #Optimizes all parameters including Xi

    # Training Loop
    train(net, t_data, x_data, u_data, t_collocation, x_collocation, u_collocation, optimizer, num_epochs,
               w_data, w_res, w_sparse, poly_order, include_sine)

    # Post-Processing and Visualization
    net.eval()
    with torch.no_grad():
        t_test = torch.linspace(t_span[0], t_span[1], int(t_span[1]/dt)).reshape(-1, 1)
        u_test = force(t_test)  # External input (forcing function)
        u_test = torch.tensor(u_test, dtype=torch.float32, requires_grad=True)  # Convert to tensor

        x_pred = net(t_test, u_test).numpy()

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    titles = ["x1", "x1_dot", "x2", "x2_dot"]

    for i in range(4):
        axs[i].plot(t_data.detach().numpy(), x_data[:, i].numpy(), 'o', label='Data', alpha=0.5)
        axs[i].plot(t_test.detach().numpy(), x_pred[:, i], label='PINN Prediction')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(titles[i])
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    # Print Discovered Equations (Simple Example)
    print("\nDiscovered Equations:")
    feature_names = ["1"]
    feature_names.extend([f"x{i+1}" for i in range(4)])
    feature_names.extend([f"u"])
    feature_names.extend([f"x{i+1}^2" for i in range(4)])
    feature_names.extend([f"u^2"])
    for i in range(5):
        for j in range(i + 1, 5):
            if (j==4):
                feature_names.append(f"x{i+1} * u")
            elif(i==4):
                feature_names.append(f"u * x{j+1}")
            else:
                feature_names.append(f"x{i+1} * x{j+1}")

    equations = []
    for i in range(4):  # For each state variable (x1_dot, x1_ddot, x2_dot, x2_ddot)
        equation = ""
        for j in range(num_functions):  # Iterate through each term in the library
            if abs(net.Xi[j, i]) > 5e-3:  # Only print terms with significant coefficients
                equation += f"{net.Xi[j, i]:.2f} * {feature_names[j]} + "
            equation = equation[:-3] #Remove last " + "
            equations.append(equation)

    state_names = ["x1_dot", "x1_ddot", "x2_dot", "x2_ddot"]
    for i in range(4):
      print(f"{state_names[i]} = {equations[i]}")

    print("\nDiscovered Coefficients (Xi):")
    print(net.Xi.data.numpy())  # Print the learned coefficients