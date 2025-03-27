#!/usr/bin/env python3.10

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso  # For SINDy sparse regression

# 1. System Definition (Double Mass-Spring-Damper)
def double_mass_spring_damper(t, z, m1, m2, k1, k2, b1, b2):
    """
    Defines the equations of motion for a double mass-spring-damper system.
    z = [x1, x1_dot, x2, x2_dot]
    """
    x1, x1_dot, x2, x2_dot = z
    x1_ddot = ( -b1*x1_dot - k1*x1 + k2*(x2-x1) + b2*(x2_dot - x1_dot) ) / m1
    x2_ddot = ( -b2*(x2_dot - x1_dot) - k2*(x2 - x1) ) / m2
    return [x1_dot, x1_ddot, x2_dot, x2_ddot]


# 2. Data Generation
def generate_data(m1, m2, k1, k2, b1, b2, initial_state, t_span, t_eval, noise_std):
    """Generates training data by solving the ODE and adding noise."""
    sol = solve_ivp(double_mass_spring_damper, t_span, initial_state, t_eval=t_eval,
                    args=(m1, m2, k1, k2, b1, b2), rtol=1e-8, atol=1e-8)
    x_data = sol.y.T + np.random.normal(0, noise_std, sol.y.T.shape)  # Add noise
    t_data = sol.t
    return t_data, x_data


# 3. Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
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

    def forward(self, t):
        x = t
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

# 4. SINDy Library (Candidate Functions)
def sindy_library(x, poly_order=2, include_sine=False):
    """
    Generates the SINDy library of candidate functions.
    x:  A tensor of shape (n_samples, state_dim)
    """
    n_samples = x.shape[0]
    state_dim = x.shape[1]

    library = [torch.ones(n_samples, 1)]  # Always include a constant term

    # Polynomial terms
    for i in range(state_dim):
        library.append(x[:, i:i+1]) # Add each state variable (x1, x1_dot, x2, x2_dot)

    if poly_order > 1:
        for i in range(state_dim):
            library.append(x[:, i:i+1]**2)  #Quadratic terms

        if state_dim > 1:
            for i in range(state_dim):
                for j in range(i + 1, state_dim):
                    library.append(x[:, i:i+1] * x[:, j:j+1])  # Cross terms

    if include_sine:
        for i in range(state_dim):
            library.append(torch.sin(x[:, i:i+1])) #Sin terms

    return torch.cat(library, dim=1) # Concatenate the library functions into a matrix


# 5. Loss Functions
def compute_losses(net, t_data, x_data, t_collocation, Xi, poly_order = 2, include_sine = False):
    """Computes the data fidelity, SINDy residual, and sparsity losses."""

    x_nn = net(t_data)  # PINN prediction at data points
    L_data = torch.mean((x_nn - x_data)**2)

    x_nn_collocation = net(t_collocation) #PINN predictions at collocation points
    dx_nn_dt = torch.autograd.grad(x_nn_collocation, t_collocation,
                                    grad_outputs=torch.ones_like(x_nn_collocation),
                                    create_graph=True, retain_graph=True)[0]

    theta = sindy_library(x_nn_collocation, poly_order, include_sine)
    L_sindy_res = torch.mean((dx_nn_dt - theta @ Xi)**2)

    L_sparsity = torch.sum(torch.abs(Xi))

    return L_data, L_sindy_res, L_sparsity


# 6. Optimization Loops (Alternating)
def train_pinn(net, t_data, x_data, t_collocation, Xi, optimizer, num_epochs,
               w_data, w_res, poly_order = 2, include_sine = False):
    """Trains the PINN with a fixed SINDy coefficient matrix Xi."""
    net.train()  # Set network to training mode
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        L_data, L_sindy_res, _ = compute_losses(net, t_data, x_data, t_collocation, Xi, poly_order, include_sine)
        L_total = w_data * L_data + w_res * L_sindy_res
        L_total.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'PINN Epoch {epoch + 1}, Data Loss: {L_data.item():.4f}, '
                  f'Residual Loss: {L_sindy_res.item():.4f}')


def update_sindy_coefficients(net, t_collocation, poly_order = 2, include_sine = False, threshold = 0.1):
    """Updates the SINDy coefficient matrix Xi using sparse regression (LASSO)."""
    net.eval()  # Set network to evaluation mode
    with torch.no_grad():
        x_nn_collocation = net(t_collocation) #PINN predictions at collocation points
        dx_nn_dt = torch.autograd.grad(x_nn_collocation, t_collocation,
                                        grad_outputs=torch.ones_like(x_nn_collocation),
                                        create_graph=False, retain_graph=False)[0] #Gradient computation is slow. Disable graph tracking when not needed

        theta = sindy_library(x_nn_collocation, poly_order, include_sine)
        # Perform sparse regression using scikit-learn's Lasso
        lasso = Lasso(alpha=threshold, fit_intercept=False, max_iter=5000, tol=1e-6, random_state = 0)
        Xi = np.zeros((theta.shape[1], 4)) # 4 = state_dim = [x1, x1_dot, x2, x2_dot]
        for i in range(4):
            lasso.fit(theta.numpy(), dx_nn_dt[:,i].numpy())
            Xi[:, i] = lasso.coef_

        Xi = torch.tensor(Xi, dtype=torch.float32)
    return Xi

# 7. Main Training Loop & Visualization
if __name__ == '__main__':
    # System Parameters (Adjust these to change the system)
    m1, m2 = 1.0, 1.0
    k1, k2 = 1.0, 1.0
    b1, b2 = 0.1, 0.1
    initial_state = [1.0, 0.0, 0.5, 0.0]  # [x1, x1_dot, x2, x2_dot]
    t_span = [0, 20]
    t_eval = np.linspace(t_span[0], t_span[1], 200)  # Data points
    noise_std = 0.01

    # Generate Data
    t_data, x_data = generate_data(m1, m2, k1, k2, b1, b2, initial_state, t_span, t_eval, noise_std)

    # Convert to PyTorch tensors
    t_data = torch.tensor(t_data[:, None], dtype=torch.float32, requires_grad=True) # [:, None] adds a dimension
    x_data = torch.tensor(x_data, dtype=torch.float32)

    # Collocation Points (for physics loss)
    t_collocation = torch.linspace(t_span[0], t_span[1], 200, requires_grad=True).reshape(-1, 1)

    # PINN Parameters
    input_dim = 1  # Time
    output_dim = 4  # [x1, x1_dot, x2, x2_dot]
    hidden_dim = 32
    num_layers = 4
    learning_rate = 1e-3
    num_epochs_pinn = 200 # Epochs for PINN training within each alternating loop

    # SINDy Parameters
    poly_order = 2 #Order of polynomial functions for SINDy library
    include_sine = False
    threshold = 0.01 #Thresholding parameter for SINDy (Lasso alpha)

    # Loss Weights
    w_data = 1.0
    w_res = 0.1 #Physics loss weight
    w_sparse = 0.001 #Sparsity weight.  (This is not used in this version but keep it in the compute losses). Lasso takes care of this.

    # Initialize PINN and Optimizer
    net = PINN(input_dim, output_dim, hidden_dim, num_layers)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Initialize SINDy coefficient matrix (Xi)
    num_functions = sindy_library(x_data, poly_order, include_sine).shape[1]  # Determine size of library
    Xi = torch.randn(num_functions, output_dim, dtype=torch.float32) * 0.01 # Small random initialization. Can also start from zero.

    # Alternating Optimization Loop
    num_alternating_iters = 10
    for iter in range(num_alternating_iters):
        print(f"Alternating Iteration {iter + 1}/{num_alternating_iters}")

        # 1. Train PINN (with fixed SINDy coefficients)
        train_pinn(net, t_data, x_data, t_collocation, Xi, optimizer, num_epochs_pinn,
                   w_data, w_res, poly_order, include_sine)

        # 2. Update SINDy Coefficients (with fixed PINN)
        Xi = update_sindy_coefficients(net, t_collocation, poly_order, include_sine, threshold)
        print("Updated SINDy Coefficients (Xi):\n", Xi.numpy())

    # Post-Processing and Visualization (Example)
    net.eval()
    with torch.no_grad():
        t_test = torch.linspace(t_span[0], t_span[1], 200).reshape(-1, 1)
        x_pred = net(t_test).numpy()

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    titles = ["x1", "x1_dot", "x2", "x2_dot"]

    for i in range(4):
        axs[i].plot(t_data.numpy(), x_data[:, i].numpy(), 'o', label='Data', alpha=0.5)
        axs[i].plot(t_test.numpy(), x_pred[:, i], label='PINN Prediction')
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
    feature_names.extend([f"x{i+1}^2" for i in range(4)])

    equations = []
    for i in range(4):  # For each state variable (x1_dot, x1_ddot, x2_dot, x2_ddot)
      equation = ""
      for j in range(Xi.shape[0]):  # Iterate through each term in the library
        if abs(Xi[j, i]) > 1e-3:  # Only print terms with significant coefficients
          equation += f"{Xi[j, i]:.2f} * {feature_names[j]} + "
      equation = equation[:-3] #Remove last " + "
      equations.append(equation)

    state_names = ["x1_dot", "x1_ddot", "x2_dot", "x2_ddot"]
    for i in range(4):
      print(f"{state_names[i]} = {equations[i]}")