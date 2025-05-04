import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim, TransferFunction

# --- New Component: Friction NN ---
class FrictionNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        # Input: velocity v2
        # Outputs: raw_kinetic_magnitude, raw_stiction_limit
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(), # Activation function
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # Two raw outputs
        )
        # Initialize weights for potentially better starting point
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, v2):
        # Input v2 should be a tensor, potentially shape [1] or scalar from solver
        v2_input = v2.unsqueeze(-1) if v2.ndim == 0 else v2 # Ensure input has feature dim

        raw_outputs = self.network(v2_input)

        # Ensure outputs are non-negative using ReLU (or softplus/exp)
        # Output 0: Kinetic friction magnitude (always positive)
        # Output 1: Stiction limit (always positive)
        kinetic_mag = torch.relu(raw_outputs[..., 0])
        stiction_limit = torch.relu(raw_outputs[..., 1])

        return kinetic_mag, stiction_limit

# Small velocity threshold for Karnopp model's deadband
KARNOPP_DV = 1e-4 # Adjust if needed

# --- System and Controller Parameters (same as before) ---
m1 = 1.0
m2 = 1.5
k1 = 2.0
k2 = 3.0
c1 = 0.5
c2 = 0.8
system_params = {'m1': m1, 'm2': m2, 'k1': k1, 'k2': k2, 'c1': c1, 'c2': c2}

# reasons for optimizing the log of the parameters:
# 1. Enforcing positivity constraints
# 2. Unconstrained optimization (log is from -inf to +inf so it is usually easier to optimize)
# 3. Numerical stability - parameters can vary over several orders of magnitude during optimization. Working in log-space can compress this range, potentially leading to smoother gradients
log_params = nn.ParameterDict({
    'logK': nn.Parameter(torch.log(torch.tensor(1.0))),
    'logz': nn.Parameter(torch.log(torch.tensor(1.0))),
    'logp': nn.Parameter(torch.log(torch.tensor(10.0)))
})

# --- Instantiate the Friction NN ---
friction_model = FrictionNN(hidden_dim=24) # Can adjust hidden_dim

# --- Modified ControlledSystem ---
class ControlledSystem(nn.Module):
    def __init__(self, system_params, log_controller_params_dict, friction_nn, x2_ref_func, karnopp_dv):
        super().__init__()
        self.p = system_params
        self.log_params = log_controller_params_dict
        self.friction_nn = friction_nn # Store the friction NN
        self.x2_ref_func = x2_ref_func
        self.karnopp_dv = karnopp_dv # Store the velocity threshold

    def forward(self, t, z):
        # Unpack combined state z = [x1, v1, x2, v2, xc]
        x1, v1, x2, v2, xc = z

        # --- Controller Calculation (same as before) ---
        K = torch.exp(self.log_params['logK'])
        z_ctrl = torch.exp(self.log_params['logz'])
        p_ctrl = torch.exp(self.log_params['logp'])
        x2_ref = self.x2_ref_func(t)
        e = x2_ref - x2
        d_xc = -p_ctrl * xc + e
        u = K * (z_ctrl - p_ctrl) * xc + K * e

        # --- System Dynamics ---
        dx1_dt = v1
        dv1_dt = (u - self.p['k1']*x1 - self.p['c1']*v1 \
                  - self.p['k2']*(x1 - x2) - self.p['c2']*(v1 - v2)) / self.p['m1']
        dx2_dt = v2

        # --- Friction Calculation for m2 ---
        # Forces acting on m2 *excluding* the unknown friction
        F_net_m2_no_friction = self.p['k2']*(x1 - x2) + self.p['c2']*(v1 - v2)

        # Get friction components from NN based on v2
        kinetic_mag_est, stiction_limit_est = self.friction_nn(v2)

        # Apply Karnopp logic
        abs_v2 = torch.abs(v2)
        is_static = abs_v2 < self.karnopp_dv

        # Calculate static friction force (clamp net force by stiction limit)
        F_static = -torch.clamp(F_net_m2_no_friction, -stiction_limit_est, stiction_limit_est)

        # Calculate kinetic friction force (magnitude from NN, opposes velocity)
        # Add a small epsilon to sign(v2) to avoid NaN gradients at v2=0 if sign is used directly
        # Or better: use the kinetic_mag_est directly as force magnitude opposing velocity
        F_kinetic = -kinetic_mag_est * torch.sign(v2) # Opposes velocity direction

        # Choose friction based on static/kinetic condition
        # Use torch.where for compatibility with potential batching/vectorization by solver
        F_friction_m2 = torch.where(is_static, F_static, F_kinetic)

        # --- Dynamics for m2 including estimated friction ---
        dv2_dt = (F_net_m2_no_friction + F_friction_m2) / self.p['m2'] # Added friction term

        # --- Combined state derivative ---
        dz_dt = torch.stack([dx1_dt, dv1_dt, dx2_dt, dv2_dt, d_xc])

        return dz_dt
    

# --- Target Response, Loss Function (same as before) ---
target_wn = 2.0
target_zeta = 0.7
target_gain = 1.0
target_tf = TransferFunction([target_gain * target_wn**2],
                             [1, 2*target_zeta*target_wn, target_wn**2])
t_start = 0.0
t_end = 15.0
n_points = 200
t_eval = torch.linspace(t_start, t_end, n_points, dtype=torch.float32)
t_ideal_np = np.linspace(t_start, t_end, n_points, dtype=np.float64)
target_pos = 1.0

# step function for the target response
def x2_reference(t):
    if isinstance(t, torch.Tensor):
        return torch.where(t >= 0.0,
                           torch.tensor(target_pos, dtype=t.dtype, device=t.device),
                           torch.tensor(0.0, dtype=t.dtype, device=t.device))
    else:
        return target_pos if t >= 0.0 else 0.0
    
u_ideal_np = np.array([x2_reference(t) for t in t_ideal_np])
_, x2_ideal_np, _ = lsim(target_tf, U=u_ideal_np, T=t_ideal_np)
x2_ideal = torch.tensor(x2_ideal_np, dtype=t_eval.dtype)

def calculate_loss(simulated_z):
    x2_simulated = simulated_z[:, 2]
    loss = torch.mean((x2_simulated - x2_ideal.to(x2_simulated.device))**2)
    return loss

# --- Training Setup ---
z0 = torch.zeros(5, dtype=torch.float32) # [x1, v1, x2, v2, xc]

# Instantiate the combined system model WITH the friction NN
controlled_ode = ControlledSystem(system_params, log_params, friction_model, x2_reference, KARNOPP_DV)

# --- Optimizer: Combine parameters ---
# Create parameter groups if different learning rates are desired
optimizer = optim.Adam([
    {'params': controlled_ode.log_params.values(), 'lr': 1e-2}, # Controller params
    {'params': controlled_ode.friction_nn.parameters(), 'lr': 1e-3} # NN params (often needs smaller LR)
], lr=1e-3) # Default LR if not specified in group

# --- Training Loop (mostly the same, just log friction params if desired) ---
n_epochs = 300 # May need more epochs
print("Starting training with learnable friction...")

history = {'loss': [], 'K': [], 'z': [], 'p': []} # Add friction NN stats later if needed

# Store initial trajectory
with torch.no_grad():
    # Ensure model is in eval mode if dropout/batchnorm were used (not the case here)
    # controlled_ode.eval()
    z_init = odeint(controlled_ode, z0, t_eval, method='rk4', options=dict(step_size=0.1))
    # controlled_ode.train() # Set back to train mode
    x2_init = z_init[:, 2]


for epoch in range(n_epochs):
    optimizer.zero_grad()
    # Ensure model is in train mode
    controlled_ode.train()

    # Simulate the system
    z_sim = odeint(controlled_ode, z0, t_eval, method='rk4', options=dict(step_size=0.1)) # Adjoint is usually better: 'dopri5', adjoint_options={"step_size": 0.1}? Try rk4 first.

    # Calculate loss
    loss = calculate_loss(z_sim)

    # Backpropagation
    loss.backward()

    # Gradient clipping (can be crucial when NNs are involved)
    torch.nn.utils.clip_grad_norm_(controlled_ode.parameters(), 1.0)

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
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.5f}, "
              f"K: {current_K:.3f}, z: {current_z:.3f}, p: {current_p:.3f}")
        # Optional: print some stats about friction NN weights/outputs if needed

print("Training finished.")

# --- Evaluation and Plotting (add friction plot) ---
print("\nFinal Learned Parameters:")
final_K = torch.exp(log_params['logK']).item()
final_z = torch.exp(log_params['logz']).item()
final_p = torch.exp(log_params['logp']).item()
print(f"Controller: K = {final_K:.4f}, z = {final_z:.4f}, p = {final_p:.4f}")
print(f"Compensator C(s) = {final_K:.3f} * (s + {final_z:.3f}) / (s + {final_p:.3f})")
# Note: Inspecting the learned friction NN requires evaluating it over a range of velocities

# Simulate final result
# controlled_ode.eval() # Set to eval mode for final simulation
with torch.no_grad():
    z_final = odeint(controlled_ode, z0, t_eval, method='rk4', options=dict(step_size=0.1))
    x1_final = z_final[:, 0].cpu().numpy()
    v1_final = z_final[:, 1].cpu().numpy()
    x2_final = z_final[:, 2].cpu().numpy()
    v2_final = z_final[:, 3].cpu().numpy()
    xc_final = z_final[:, 4].cpu().numpy()

# Calculate final control input u(t) and learned friction F_fric(t)
t_np = t_eval.cpu().numpy()
x2_ref_np = np.array([x2_reference(t) for t in t_np])
e_final = x2_ref_np - x2_final
u_final = final_K * (final_z - final_p) * xc_final + final_K * e_final

# Recalculate friction force along the final trajectory
F_fric_final_np = []
F_kinetic_final_np = []
F_stiction_limit_final_np = []
controlled_ode.eval() # Ensure NN is in eval mode
with torch.no_grad():
    for i in range(len(t_np)):
        # Get state at this time step (need v2 and forces acting on m2)
        x1_t, v1_t, x2_t, v2_t, xc_t = z_final[i]

        # Forces excluding friction
        F_net_m2_no_friction_t = system_params['k2']*(x1_t - x2_t) + system_params['c2']*(v1_t - v2_t)

        # Get friction components from NN
        kinetic_mag_est_t, stiction_limit_est_t = controlled_ode.friction_nn(v2_t)
        F_kinetic_final_np.append(kinetic_mag_est_t.item())
        F_stiction_limit_final_np.append(stiction_limit_est_t.item())


        # Apply Karnopp logic
        abs_v2_t = torch.abs(v2_t)
        is_static_t = abs_v2_t < controlled_ode.karnopp_dv

        F_static_t = -torch.clamp(F_net_m2_no_friction_t, -stiction_limit_est_t, stiction_limit_est_t)
        F_kinetic_t = -kinetic_mag_est_t * torch.sign(v2_t)
        F_friction_m2_t = torch.where(is_static_t, F_static_t, F_kinetic_t)
        F_fric_final_np.append(F_friction_m2_t.item())

F_fric_final_np = np.array(F_fric_final_np)
F_kinetic_final_np = np.array(F_kinetic_final_np)
F_stiction_limit_final_np = np.array(F_stiction_limit_final_np)


# Plotting
plt.figure(figsize=(12, 14)) # Increased height

# Plot x2 response
plt.subplot(4, 1, 1) # Changed to 4 rows
plt.plot(t_np, x2_ideal.cpu().numpy(), 'k--', label='Ideal Target Response')
plt.plot(t_np, x2_init.cpu().numpy(), 'r:', label=f'Initial x2')
plt.plot(t_np, x2_final, 'b-', label=f'Final Learned x2')
plt.plot(t_np, x2_ref_np, 'g-.', label='Reference x2_ref')
plt.title('Position of Mass 2 (x2)')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

# Plot control input u
plt.subplot(4, 1, 2)
plt.plot(t_np, u_final, 'm-', label='Control Input u(t)')
plt.title('Control Input (Force on m1)')
plt.xlabel('Time (s)')
plt.ylabel('Force')
plt.legend()
plt.grid(True)

# --- New Plot: Learned Friction Force ---
plt.subplot(4, 1, 3)
plt.plot(t_np, F_fric_final_np, 'c-', label='Estimated Friction Force on m2')
# Optionally plot components if insightful
# plt.plot(t_np, F_kinetic_final_np * -np.sign(v2_final), 'y--', label='Kinetic Friction Component')
# plt.plot(t_np, F_stiction_limit_final_np, 'p:', label='Learned Stiction Limit')
plt.title('Estimated Friction Force on m2')
plt.xlabel('Time (s)')
plt.ylabel('Force')
plt.legend()
plt.grid(True)

# Plot parameter evolution
plt.subplot(4, 1, 4)
plt.plot(history['K'], label='K')
plt.plot(history['z'], label='z')
plt.plot(history['p'], label='p')
plt.yscale('log')
plt.title('Learned Controller Parameters (log scale)')
plt.xlabel('Epoch')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot loss curve (same as before)
plt.figure(figsize=(6, 4))
plt.plot(history['loss'])
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.show()

# Optional: Plot learned friction curve (Friction vs Velocity)
plt.figure(figsize=(7, 5))
v_range = torch.linspace(min(v2_final) - 0.1, max(v2_final) + 0.1, 200)
controlled_ode.eval()
with torch.no_grad():
    kinetic_mag_curve, stiction_limit_curve = controlled_ode.friction_nn(v_range)
    kinetic_force_curve = -kinetic_mag_curve * torch.sign(v_range)
    # Note: Stiction limit from NN is plotted, but actual static friction depends on applied force

plt.plot(v_range.numpy(), kinetic_force_curve.numpy(), label='Learned Kinetic Friction $F_{kin}(v_2)$')
# Indicate static region conceptually
stiction_limit_avg = np.mean(F_stiction_limit_final_np) # Use average learned limit
plt.plot([0, 0], [-stiction_limit_avg, stiction_limit_avg], 'r--', linewidth=2, label=f'Learned Stiction Range (avg limit $\\approx$ {stiction_limit_avg:.2f})')
plt.hlines(0, v_range.numpy().min(), v_range.numpy().max(), colors='k', linestyles='dotted')
plt.vlines(0, kinetic_force_curve.min(), kinetic_force_curve.max(), colors='k', linestyles='dotted')
plt.xlabel('Velocity of Mass 2 (v2)')
plt.ylabel('Estimated Friction Force')
plt.title('Learned Friction Characteristic (NN Approx.)')
plt.legend()
plt.grid(True)
plt.show()