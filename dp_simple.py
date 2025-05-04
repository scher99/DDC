import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint # pip install torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim, TransferFunction

# ------------------------------------
# 1. System Definition
# ------------------------------------
# Parameters for the two-mass-spring-damper system
m1 = 1.0
m2 = 1.5
k1 = 2.0
k2 = 3.0
c1 = 0.5
c2 = 0.8

# Store parameters in a dictionary for easy access
system_params = {'m1': m1, 'm2': m2, 'k1': k1, 'k2': k2, 'c1': c1, 'c2': c2}

# ------------------------------------
# 2. Controller Parameters (Learnable)
# ------------------------------------
# We learn log(K), log(z), log(p) to ensure positivity
# Initialize them (these are just starting points)
log_params = nn.ParameterDict({
    'logK': nn.Parameter(torch.log(torch.tensor(70.0))),
    'logz': nn.Parameter(torch.log(torch.tensor(0.7))),
    'logp': nn.Parameter(torch.log(torch.tensor(19.0))) # Initial guess: p > z (lead)
})


# ------------------------------------
# 3. Combined ODE System (Plant + Controller)
# ------------------------------------
class ControlledSystem(nn.Module):
    def __init__(self, system_params, log_controller_params_dict, x2_ref_func):
        super().__init__()
        self.p = system_params
        self.log_params = log_controller_params_dict
        self.x2_ref_func = x2_ref_func

    # The forward method defines the dZ/dt function required by the ODE solver
    def forward(self, t, z):
        # Unpack combined state z = [x1, v1, x2, v2, xc]
        x1, v1, x2, v2, xc = z

        # Get current controller parameters (ensure positivity)
        K = torch.exp(self.log_params['logK'])
        z_ctrl = torch.exp(self.log_params['logz']) # to make it z and p
        p_ctrl = torch.exp(self.log_params['logp'])

        # Calculate error and control input u
        x2_ref = self.x2_ref_func(t)
        e = x2_ref - x2
        # u = K * (z_ctrl - p_ctrl) * xc + K * e # Original state-space form
        # Ensure controller state calculation uses the correct pole `p_ctrl`
        # xc be the state of the controller. A common realization is:
        # d(xc)/dt = -p * xc + e(t)
        # u(t) = K * (z - p) * xc(t) + K * e(t)
        # Note: We need p > 0 for stability. 
        # We also typically want z > 0. p > z gives lead , 
        # z > p gives lag characteristics. The optimization will find the best K, z, p.
        d_xc = -p_ctrl * xc + e
        # Recalculate u based on the definition U(s) = C(s)E(s) -> (s+p)U = K(s+z)E
        # Using the state-space form derived earlier:
        u = K * (z_ctrl - p_ctrl) * xc + K * e

        # System dynamics
        dx1_dt = v1
        dv1_dt = (u - self.p['k1']*x1 - self.p['c1']*v1 \
                  - self.p['k2']*(x1 - x2) - self.p['c2']*(v1 - v2)) / self.p['m1']
        dx2_dt = v2
        dv2_dt = (self.p['k2']*(x1 - x2) + self.p['c2']*(v1 - v2)) / self.p['m2']

        # Combined state derivative
        dz_dt = torch.stack([dx1_dt, dv1_dt, dx2_dt, dv2_dt, d_xc])

        return dz_dt

# ------------------------------------
# 4. Define Target Response and Loss
# ------------------------------------
# Target 2nd order system parameters
target_wn = 10.0  # Desired natural frequency (rad/s)
target_zeta = 0.7 # Desired damping ratio (critically damped = 1, underdamped < 1)
target_gain = 1.0 # Desired steady-state gain (usually 1 for position control)

# Create the target transfer function (using scipy.signal)
target_tf = TransferFunction([target_gain * target_wn**2],
                             [1, 2*target_zeta*target_wn, target_wn**2])

# Simulation time
t_start = 0.0
t_end = 4.0
n_points = 400+1
# Time vector for torchdiffeq solver (needs to be a Tensor)
t_eval = torch.linspace(t_start, t_end, n_points, dtype=torch.float32) # Keep dtype consistent if needed

# Time vector specifically for scipy.lsim (use numpy for robustness)
# Generate this directly with numpy to ensure exact equal spacing for lsim
t_ideal_np = np.linspace(t_start, t_end, n_points, dtype=np.float64) # Use float64 for better precision in lsim if needed

# Reference input (step function)
target_pos = 1.0
def x2_reference(t):
    # Simple step function - can be made more complex
    # Need to handle tensor inputs from the solver
    if isinstance(t, torch.Tensor):
        # Apply thresholding carefully for tensors
        return torch.where(t >= 0.0,
                           torch.tensor(target_pos, dtype=t.dtype, device=t.device),
                           torch.tensor(0.0, dtype=t.dtype, device=t.device))
    else: # Handle scalar float input during target generation
        return target_pos if t >= 0.0 else 0.0

# Generate the ideal target trajectory using scipy.signal.lsim
# Use the numpy-generated time vector t_ideal_np here
u_ideal_np = np.array([x2_reference(t) for t in t_ideal_np]) # Step input for target sim
_, x2_ideal_np, _ = lsim(target_tf, U=u_ideal_np, T=t_ideal_np)

# Convert the ideal trajectory back to a torch tensor for the loss calculation
# Ensure it matches the dtype and device of the simulation output later
x2_ideal = torch.tensor(x2_ideal_np, dtype=t_eval.dtype) # Match t_eval's dtype

# Loss function: MSE between simulated x2 and ideal x2
def calculate_loss(simulated_z):
    x2_simulated = simulated_z[:, 2] # Extract x2 trajectory
    # Ensure x2_ideal is on the same device as x2_simulated for loss calculation
    loss = torch.mean((x2_simulated - x2_ideal.to(x2_simulated.device))**2)
    return loss

# ------------------------------------
# 5. Training Setup
# ------------------------------------
# Initial state (all zeros)
z0 = torch.zeros(5, dtype=torch.float32) # [x1, v1, x2, v2, xc]

# Instantiate the combined system model
controlled_ode = ControlledSystem(system_params, log_params, x2_reference)

# Optimizer
# Need to ensure parameters are properly registered if using nn.ModuleDict or similar
# Here, log_params directly holds Parameter objects, so we can pass them like this:
optimizer = optim.Adam(controlled_ode.parameters(), lr=1e-2) # Might need tuning

# Training loop
n_epochs = 500
print("Starting training...")

history = {'loss': [], 'K': [], 'z': [], 'p': []}

# Store initial trajectory
with torch.no_grad():
    z_init = odeint(controlled_ode, z0, t_eval, method='rk4', options=dict(step_size=0.1))
    x2_init = z_init[:, 2]

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Simulate the system with current parameters
    # Using adjoint method requires the function to accept t as a scalar potentially
    # Ensure x2_reference can handle this if necessary (already handled above)
    z_sim = odeint(controlled_ode, z0, t_eval, method='rk4', options=dict(step_size=0.1)) # 'rk4' or 'dopri5'

    # Calculate loss
    loss = calculate_loss(z_sim)

    # Backpropagation
    loss.backward()

    # Gradient clipping (optional, but can help stability)
    # torch.nn.utils.clip_grad_norm_(log_params.values(), 1.0)

    # Optimizer step
    optimizer.step()

    # Log results
    current_K = torch.exp(log_params['logK']).item()
    current_z = torch.exp(log_params['logz']).item()
    current_p = torch.exp(log_params['logp']).item()

    history['loss'].append(loss.item())
    history['K'].append(current_K)
    history['z'].append(current_z)
    history['p'].append(current_p)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, "
              f"K: {current_K:.3f}, z: {current_z:.3f}, p: {current_p:.3f}")

print("Training finished.")

# ------------------------------------
# 6. Evaluation and Plotting
# ------------------------------------
print("\nFinal Learned Parameters:")
final_K = torch.exp(log_params['logK']).item()
final_z = torch.exp(log_params['logz']).item()
final_p = torch.exp(log_params['logp']).item()
print(f"K = {final_K:.4f}")
print(f"z = {final_z:.4f}")
print(f"p = {final_p:.4f}")
print(f"Compensator C(s) = {final_K:.3f} * (s + {final_z:.3f}) / (s + {final_p:.3f})")


# Simulate final result
with torch.no_grad():
    z_final = odeint(controlled_ode, z0, t_eval, method='rk4', options=dict(step_size=0.1))
    x1_final = z_final[:, 0].cpu().numpy()
    v1_final = z_final[:, 1].cpu().numpy()
    x2_final = z_final[:, 2].cpu().numpy()
    v2_final = z_final[:, 3].cpu().numpy()
    xc_final = z_final[:, 4].cpu().numpy()

# Calculate final control input u(t)
t_np = t_eval.cpu().numpy()
x2_ref_np = np.array([x2_reference(t) for t in t_np])
e_final = x2_ref_np - x2_final
u_final = final_K * (final_z - final_p) * xc_final + final_K * e_final

# Plotting
plt.figure(figsize=(12, 10))

# Plot x2 response
plt.subplot(3, 1, 1)
plt.plot(t_np, x2_ideal.cpu().numpy(), 'k--', label='Ideal Target Response (2nd Order)')
plt.plot(t_np, x2_init.cpu().numpy(), 'r:', label=f'Initial x2 (K={history["K"][0]:.2f}, z={history["z"][0]:.2f}, p={history["p"][0]:.2f})')
plt.plot(t_np, x2_final, 'b-', label=f'Final Learned x2 (K={final_K:.2f}, z={final_z:.2f}, p={final_p:.2f})')
plt.plot(t_np, x2_ref_np, 'g-.', label='Reference x2_ref')
plt.title('Position of Mass 2 (x2)')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

# Plot control input u
plt.subplot(3, 1, 2)
plt.plot(t_np, u_final, 'm-', label='Control Input u(t)')
plt.title('Control Input (Force on m1)')
plt.xlabel('Time (s)')
plt.ylabel('Force')
plt.legend()
plt.grid(True)

# Plot parameter evolution (optional)
plt.subplot(3, 1, 3)
plt.plot(history['K'], label='K')
plt.plot(history['z'], label='z')
plt.plot(history['p'], label='p')
plt.yscale('log') # Use log scale if parameters vary widely
plt.title('Learned Controller Parameters (log scale)')
plt.xlabel('Epoch')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

# Plot loss curve
plt.figure(figsize=(6, 4))
plt.plot(history['loss'])
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.show()