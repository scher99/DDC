import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# Neural network architecture to represent state: [x1, x1_dot, x2, x2_dot]
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
       
        # Construct hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
       
        # Xavier initialization
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)
   
    def forward(self, t, u):
        # t time vector is of shape [N, 1]
        # u the force input shape [N, 1]
        a = torch.cat((t, u), dim=1)
        for i in range(len(self.layers)-1):
            a = self.activation(self.layers[i](a))
        out = self.layers[-1](a)
        return out

# Optionally, generate synthetic data (e.g., from a known numerical solver)
# to provide additional supervision. For demonstration, we numerically integrate the
# ODEs with known initial conditions.
def two_mass_ode(y, t, m1, m2, k1, c1):
    # y = [x1, x1_dot, x2, x2_dot]
    x1, x1_dot, x2, x2_dot = y
    u_val = np.sin(2 * np.pi * t)  # same as forcing above (assuming amplitude 1)
    x1_ddot = (-k1 * (x1 - x2) - c1 * (x1_dot - x2_dot) + u_val) / m1
    x2_ddot = ( k1 * (x1 - x2) + c1 * (x1_dot - x2_dot)) / m2
    return [x1_dot, x1_ddot, x2_dot, x2_ddot]

# Define forcing input function (for mass 1)
def force(t):
    # Example: a sinusoidal force
    return 1.0 * torch.sin(2.0 * np.pi * t)


# Define the physics-informed residuals based on the system dynamics.
def physics_residual(t, u):
    """
    Given a time tensor t and input force u, returns the physics residual loss terms based on the ODEs.
    """
    # Ensure t, u requires grad for autograd differentiation.
    t.requires_grad = True
    u.requires_grad = True
   
    # Forward pass: get the predicted states
    y = model(t, u)
    x1     = y[:, 0:1]
    x1_dot = y[:, 1:2]
    x2     = y[:, 2:3]
    x2_dot = y[:, 3:4]

    # Second derivatives using autograd:
    # Derivative of x1_dot wrt t gives x1_ddot
    x1_dot_grad = torch.autograd.grad(x1_dot, t, grad_outputs=torch.ones_like(x1_dot), \
                                      retain_graph=True, create_graph=True)[0]
    # Similarly, derivative of x2_dot gives x2_ddot
    x2_dot_grad = torch.autograd.grad(x2_dot, t, grad_outputs=torch.ones_like(x2_dot), \
                                      retain_graph=True, create_graph=True)[0]
   
    # Enforce the known forcing function on mass 1
    # u = force(t)
   
    # Physics residuals from the ODEs:
    # For mass 1: m1*x1_ddot = -k1*(x1 - x2) - c1*(x1_dot - x2_dot) + u
    e1 = m1 * x1_dot_grad + k1 * (x1 - x2) + c1 * (x1_dot - x2_dot) - u  # should equal 0
    # For mass 2: m2*x2_ddot =  k1*(x1 - x2) + c1*(x1_dot - x2_dot)
    e2 = m2 * x2_dot_grad - k1 * (x1 - x2) - c1 * (x1_dot - x2_dot)      # should equal 0

    return e1, e2


# Define the loss function:
def loss_function(t_data, u_data, y_data):
    # Physics loss on collocation points
    e1, e2 = physics_residual(t_data, u_data)
    physics_loss = torch.mean(e1**2) + torch.mean(e2**2)

    # Data loss on measurement data
    y_pred = model(t_data, u_data)
    data_loss = torch.mean((y_pred - y_data)**2)

    return physics_loss + data_loss, physics_loss, data_loss

if __name__ == '__main__':
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

    # Define true physical parameters (assumed unknown)
    m1_true = 1.0       # mass 1
    m2_true = 1.5       # mass 2
    k1_true = 2.0       # spring stiffness
    c1_true = 0.5       # damping coefficient

    Tf = 10 # seconds for the test simulation
    Ts = 0.01 # seconds for the test simulation

    # Define physical parameters (what is assumed known) or what is learnable
    if(True):
        m1 = m1_true       # mass 1
    else:
        m1 = nn.Parameter(torch.tensor(2.0, device=device))
    if(True):
        m2 = m2_true       # mass 2
    else:
        m2 = nn.Parameter(torch.tensor(2.0, device=device))
    if(False):
        k1 = k1_true       # spring stiffness
    else:
        k1 = nn.Parameter(torch.tensor(1.0, device=device))
    if(True):
        c1 = c1_true       # damping coefficient
    else:
        c1 = nn.Parameter(torch.tensor(1.5, device=device))

    # learnable parameters list
    learnable_params = []
    if(not isinstance(m1, float)):
        learnable_params.append(m1)
    if(not isinstance(m2, float)):  
        learnable_params.append(m2)
    if(not isinstance(k1, float)):
        learnable_params.append(k1)
    if(not isinstance(c1, float)):
        learnable_params.append(c1)
    print(f"learnable_params used: {learnable_params}")

    # Define the NN model architecture:
    # Input: time t and input u, Output: [x1, x1_dot, x2, x2_dot]
    layers = [2, 50, 100, 100, 50, 4]
    model = PINN(layers).to(device)

    # Load the model's parameters and the learned physical parameters
    try:
        checkpoint = torch.load('pinn_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['m1'] is not None:
            m1 = checkpoint['m1']
        if checkpoint['m2'] is not None:
            m2 = checkpoint['m2']
        if checkpoint['k1'] is not None:
            k1 = checkpoint['k1']
        if checkpoint['c1'] is not None:
            c1 = checkpoint['c1']
        print("Model and parameters loaded.")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")

    # Time domain for simulation
    t_sim = np.linspace(0, Tf, int(Tf/Ts))
    # Initial conditions: [x1, x1_dot, x2, x2_dot]
    y0 = [0.0, 0.0, 0.0, 0.0]
    sol = odeint(two_mass_ode, y0, t_sim, args=(m1_true,  m2_true,  k1_true,  c1_true))

    # Convert simulated data to torch tensors
    t_full = torch.tensor(t_sim, dtype=torch.float32).view(-1,1).to(device)
    u_full = torch.tensor(force(t_full).clone().detach(), dtype=torch.float32).view(-1,1).to(device)
    y_full = torch.tensor(sol,   dtype=torch.float32).to(device)

    # Training parameters
    Tf_train = 7.0
    Ts_train = 0.01
    n_epochs = 10000 #70000
    learning_rate = 1e-5

    # what parameters are learnable
    optimizer = optim.Adam(list(model.parameters()) + learnable_params, lr=learning_rate)

    # Generate collocation points for enforcing the physics constraint in the domain
    # t_colloc = torch.unsqueeze(torch.linspace(0.0, Tf_train, int(Tf_train/Ts_train)), 1).to(device)
    y_train = y_full[:int(Tf_train/Ts_train), :]
    u_train = u_full[:int(Tf_train/Ts_train), :]
    t_train = t_full[:int(Tf_train/Ts_train), :]

    # Training loop history
    loss_history = []
    physics_loss_history = []
    data_loss_history = []

    print("Starting training ...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
    
        total_loss, phys_loss, d_loss = loss_function(t_train, u_train, y_train)
        total_loss.backward()
        optimizer.step()
    
        loss_history.append(total_loss.item())
        physics_loss_history.append(phys_loss.item())
        data_loss_history.append(d_loss.item())
    
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.4e} | Physics Loss: {phys_loss.item():.4e} | Data Loss: {d_loss.item():.4e}")


    # Print the final learned parameter values
    if(not isinstance(m1, float)):
        print(f"Learned parameter: m1 = {m1.item():.4f} (physical value: {m1_true})")
    if(not isinstance(m2, float)):  
        print(f"Learned parameter: m2 = {m2.item():.4f} (physical value: {m2_true})")
    if(not isinstance(k1, float)):
        print(f"Learned parameter: k1 = {k1.item():.4f} (physical value: {k1_true})")
    if(not isinstance(c1, float)):
        print(f"Learned parameter: c1 = {c1.item():.4f} (physical value: {c1_true})")

    # Plot the losses
    plt.figure(figsize=(8,5))
    plt.semilogy(loss_history, label='Total Loss')
    plt.semilogy(physics_loss_history, label='Physics Loss')
    plt.semilogy(data_loss_history, label='Data Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss History")
    plt.show()

    # Plot the predictions versus true simulation:
    model.eval()
    with torch.no_grad():
        y_pred = model(t_full, u_full).cpu().numpy()

    # Save the model's parameters and the learned physical parameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'm1': m1 if isinstance(m1, nn.Parameter) else None,
        'm2': m2 if isinstance(m2, nn.Parameter) else None,
        'k1': k1 if isinstance(k1, nn.Parameter) else None,
        'c1': c1 if isinstance(c1, nn.Parameter) else None,
    }, 'pinn_model.pth')
    print("Model and parameters saved.")
    
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(t_sim, sol[:,0], 'b-', label='x1 True')
    plt.plot(t_sim, y_pred[:,0], 'r--', label='x1 PINN')
    plt.xlabel("Time")
    plt.ylabel("x1")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t_sim, sol[:,2], 'b-', label='x2 True')
    plt.plot(t_sim, y_pred[:,2], 'r--', label='x2 PINN')
    plt.xlabel("Time")
    plt.ylabel("x2")
    plt.legend()

    plt.tight_layout()
    plt.show()

