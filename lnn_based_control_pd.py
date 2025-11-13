import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Assume all previous code from the LNN example is here ---
# Including:
# - params_true
# - LNN class definition
# - lnn_model (INSTANCE of LNN, assumed to be TRAINED)
# - true_ode_lnn (original true ODE for data generation)
# - Training data generation for LNN (Y_train, q_train, q_dot_train, a_train, target_EL1, target_EL2)
# - LNN training loop (optimizer, loss_fn, epochs etc.)
# --- We need a trained lnn_model for this controller ---

# For demonstration, let's quickly re-run a minimal LNN training if you don't have a saved model.
# In a real scenario, you'd load a pre-trained lnn_model.
# THIS IS A PLACEHOLDER - USE YOUR ACTUALLY TRAINED LNN MODEL
# Re-pasting minimal LNN training setup for self-containment of this example:
print("Setting up and (re)training a placeholder LNN model for the controller example...")
# --- 1. Define System Parameters ---
params_true = {
    'm1': 1.0, 'm2': 1.5,
    'k1': 1.0, 'k2': 0.8,
    'c1': 0.1, 'c2': 0.15,
    'mu1_coulomb': 0.2, 'mu2_coulomb': 0.15,
    'g_approx': 1.0
}
# --- 2. Define the "True" System ODEs ---
def true_ode_lnn_orig(t, y_arr, p): # Renamed to avoid conflict with controlled version
    x1, v1, x2, v2 = y_arr
    f_spring1 = -p['k1'] * x1; f_damper1 = -p['c1'] * v1
    f_spring2_on1 = p['k2'] * (x2 - x1); f_damper2_on1 = p['c2'] * (v2 - v1)
    f_friction1 = -p['mu1_coulomb'] * p['g_approx'] * np.sign(v1) if v1 != 0 else 0
    f_spring2_on2 = -p['k2'] * (x2 - x1); f_damper2_on2 = -p['c2'] * (v2 - v1)
    f_friction2 = -p['mu2_coulomb'] * p['g_approx'] * np.sign(v2) if v2 != 0 else 0
    a1 = (f_spring1 + f_damper1 + f_spring2_on1 + f_damper2_on1 + f_friction1) / p['m1']
    a2 = (f_spring2_on2 + f_damper2_on2 + f_friction2) / p['m2']
    return [v1, a1, v2, a2]
# --- 3. Generate Training Data ---
t_span_train = [0, 10]; n_points_train = 500 # Shorter for quick demo
t_eval_train = np.linspace(t_span_train[0], t_span_train[1], n_points_train)
y0_train = [1.0, 0.0, 0.5, 0.0]
sol_train = solve_ivp(true_ode_lnn_orig, t_span_train, y0_train, args=(params_true,), dense_output=True, t_eval=t_eval_train)
Y_train_np = sol_train.y.T; t_train_np = sol_train.t
dYdt_train_np = np.array([true_ode_lnn_orig(t_train_np[i], Y_train_np[i], params_true) for i in range(len(t_train_np))])
a1_true_np = dYdt_train_np[:, 1]; a2_true_np = dYdt_train_np[:, 3]
x1_d, v1_d, x2_d, v2_d = Y_train_np[:,0], Y_train_np[:,1], Y_train_np[:,2], Y_train_np[:,3]
Q_nc1_known_np = -params_true['c1'] * v1_d - params_true['c2'] * (v1_d - v2_d)
Q_nc2_known_np = -params_true['c2'] * (v2_d - v1_d)
target_EL1_np = params_true['m1'] * a1_true_np - Q_nc1_known_np
target_EL2_np = params_true['m2'] * a2_true_np - Q_nc2_known_np
Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
target_EL1 = torch.tensor(target_EL1_np, dtype=torch.float32).unsqueeze(1)
target_EL2 = torch.tensor(target_EL2_np, dtype=torch.float32).unsqueeze(1)
q_train = Y_train[:, [0, 2]]; q_dot_train = Y_train[:, [1, 3]]; a_train = torch.tensor(dYdt_train_np[:, [1,3]], dtype=torch.float32)
# --- 4. LNN Definition (copied from your LNN example) ---
class LNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim); self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x_state):
        x = torch.tanh(self.fc1(x_state)); x = torch.tanh(self.fc2(x)); L = self.fc3(x)
        return L
    def get_euler_lagrange_terms(self, q_current, q_dot_current, a_current): # From your corrected LNN
        q = q_current.clone().detach().requires_grad_(True)
        q_dot = q_dot_current.clone().detach().requires_grad_(True)
        state_for_L = torch.cat((q, q_dot), dim=1)
        L_val = self.forward(state_for_L)
        grad_L_outputs = torch.ones_like(L_val, requires_grad=False)
        dL_dq_tuple = torch.autograd.grad(outputs=L_val, inputs=q, grad_outputs=grad_L_outputs, create_graph=True, allow_unused=True)
        dL_dq = dL_dq_tuple[0] if dL_dq_tuple[0] is not None else torch.zeros_like(q)
        p_tuple = torch.autograd.grad(outputs=L_val, inputs=q_dot, grad_outputs=grad_L_outputs, create_graph=True, allow_unused=True)
        p = p_tuple[0] if p_tuple[0] is not None else torch.zeros_like(q_dot)
        dp_dt = torch.zeros_like(p)
        for i in range(p.shape[1]):
            p_i = p[:, i]; grad_outputs_p_i_scalar = torch.ones_like(p_i, requires_grad=False)
            dp_i_dq_vec_tuple = torch.autograd.grad(outputs=p_i, inputs=q, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, retain_graph=True, allow_unused=True)
            dp_i_dq_vec = dp_i_dq_vec_tuple[0] if dp_i_dq_vec_tuple[0] is not None else torch.zeros_like(q)
            dp_i_dq_dot_vec_tuple = torch.autograd.grad(outputs=p_i, inputs=q_dot, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, retain_graph=True, allow_unused=True)
            dp_i_dq_dot_vec = dp_i_dq_dot_vec_tuple[0] if dp_i_dq_dot_vec_tuple[0] is not None else torch.zeros_like(q_dot)
            term1 = (dp_i_dq_vec * q_dot_current).sum(dim=1); term2 = (dp_i_dq_dot_vec * a_current).sum(dim=1)
            dp_dt[:, i] = term1 + term2
        EL_terms = dp_dt - dL_dq
        return EL_terms
# --- 5. LNN Training (minimal placeholder) ---
lnn_model = LNN(input_dim=4, hidden_dim=64) # Reduced hidden_dim for speed
optimizer = optim.Adam(lnn_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
epochs_lnn_ctrl = 500 # Reduced epochs for quick demo
lnn_model.train()
for epoch in range(epochs_lnn_ctrl):
    permutation = torch.randperm(q_train.size(0))
    for i in range(0, q_train.size(0), 64):
        optimizer.zero_grad(); indices = permutation[i:i+64]
        batch_q, batch_q_dot, batch_a = q_train[indices], q_dot_train[indices], a_train[indices]
        batch_target_EL1, batch_target_EL2 = target_EL1[indices], target_EL2[indices]
        el_preds = lnn_model.get_euler_lagrange_terms(batch_q, batch_q_dot, batch_a)
        loss1 = loss_fn(el_preds[:, 0].unsqueeze(1), batch_target_EL1)
        loss2 = loss_fn(el_preds[:, 1].unsqueeze(1), batch_target_EL2)
        total_loss = loss1 + loss2; total_loss.backward(); optimizer.step()
    if (epoch + 1) % 100 == 0: print(f"LNN Ctrl Pre-Train Epoch [{epoch+1}/{epochs_lnn_ctrl}], Loss: {total_loss.item():.4f}")
print("LNN placeholder training for controller finished.")
lnn_model.eval() # Set to eval mode

# --- End of Placeholder LNN Setup ---


# 1. Modified True System ODE to include control inputs tau = [tau1, tau2]
def true_ode_controlled(t, y_arr, p, tau_func, t_offset=0.0):
    # y_arr = [x1, v1, x2, v2]
    x1, v1, x2, v2 = y_arr

    # Get current control inputs
    # tau_func might need (t, y_arr) or just y_arr
    # For simplicity, assume tau_func(y_arr) returns [tau1, tau2]
    # If tau_func is state-dependent only, t_offset is not strictly needed here
    # but can be useful if tau_func has explicit time dependency for reference trajectories
    tau1, tau2 = tau_func(t - t_offset, y_arr) # Apply current control

    # Forces on m1
    f_spring1 = -p['k1'] * x1
    f_damper1 = -p['c1'] * v1
    f_spring2_on1 = p['k2'] * (x2 - x1)
    f_damper2_on1 = p['c2'] * (v2 - v1)
    f_friction1 = -p['mu1_coulomb'] * p['g_approx'] * np.sign(v1) if v1 != 0 else 0

    # Forces on m2
    f_spring2_on2 = -p['k2'] * (x2 - x1)
    f_damper2_on2 = -p['c2'] * (v2 - v1)
    f_friction2 = -p['mu2_coulomb'] * p['g_approx'] * np.sign(v2) if v2 != 0 else 0

    # Equations of motion with control inputs
    a1 = (f_spring1 + f_damper1 + f_spring2_on1 + f_damper2_on1 + f_friction1 + tau1) / p['m1']
    a2 = (f_spring2_on2 + f_damper2_on2 + f_friction2 + tau2) / p['m2']

    return [v1, a1, v2, a2]


# 2. Function to get M_nn(q, q_dot) and h_nn(q, q_dot) from the LNN
# h_nn represents C(q,q_dot)q_dot + G(q) terms
# Q_nc_estimated can also be added here if we have a model for it
def get_lnn_dynamics_terms(q_np, q_dot_np, lnn_model_trained):
    # q_np, q_dot_np are 1D numpy arrays e.g. [x1, x2] and [v1, v2]
    with torch.enable_grad(): # Crucial for autograd
        q_th = torch.tensor([q_np], dtype=torch.float32, requires_grad=True)
        q_dot_th = torch.tensor([q_dot_np], dtype=torch.float32, requires_grad=True)
        state_for_L = torch.cat((q_th, q_dot_th), dim=1)

        L_val = lnn_model_trained(state_for_L) # L_nn(q, q_dot)

        grad_L_outputs = torch.ones_like(L_val)
        dL_dq = torch.autograd.grad(L_val, q_th, grad_outputs=grad_L_outputs, create_graph=True)[0]
        # p_generalized = dL/d(q_dot)
        p_generalized = torch.autograd.grad(L_val, q_dot_th, grad_outputs=grad_L_outputs, create_graph=True)[0]

        # Mass matrix M_nn_ij = d(p_i)/d(q_dot_j)
        num_dof = q_dot_th.shape[1]
        M_nn = torch.zeros((num_dof, num_dof), dtype=torch.float32, device=q_th.device)
        # C_terms_nn represents (dp/dq)*q_dot part, i.e., part of h_nn
        C_terms_nn_vec = torch.zeros_like(p_generalized) # Will be (1, num_dof)

        for i in range(num_dof):
            p_i = p_generalized[:, i]
            grad_outputs_p_i_scalar = torch.ones_like(p_i)

            grad_p_i_wrt_q_dot = torch.autograd.grad(p_i, q_dot_th, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, allow_unused=True)[0]
            if grad_p_i_wrt_q_dot is not None:
                M_nn[i, :] = grad_p_i_wrt_q_dot[0, :]
            else: M_nn[i, i] = 1e-6 # Fallback for safety

            grad_p_i_wrt_q = torch.autograd.grad(p_i, q_th, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, allow_unused=True)[0]
            if grad_p_i_wrt_q is not None:
                C_terms_nn_vec[0,i] = (grad_p_i_wrt_q * q_dot_th).sum()

        # h_nn(q, q_dot) = C_terms_nn_vec - dL_dq
        # (from M*a + C_terms - dL/dq = tau => M*a + h = tau)
        h_nn_vec = C_terms_nn_vec - dL_dq # This is (1, num_dof)

    return M_nn.detach().cpu().numpy(), h_nn_vec[0].detach().cpu().numpy() # Return (num_dof,num_dof) and (num_dof,)

# 3. Define the Controller Logic
# Controller parameters
Kp_x2 = 25.0  # Proportional gain for x2 position
Kd_x2 = 10.0  # Derivative gain for x2 velocity
x2_desired_setpoint = 0.0 # Target position for x2

# We also need some behavior for x1, otherwise M_nn might be singular if we only control one DOF
# For simplicity, let's try to damp x1 towards its initial position or zero.
Kp_x1 = 1.0
Kd_x1 = 0.5
x1_desired_setpoint = 0.0 # Or y0_controlled[0] if you want to hold initial x1

# Store control inputs for plotting
control_inputs_history = []

def computed_torque_controller(t, y_arr_ctrl):
    global control_inputs_history
    # y_arr_ctrl = [x1, v1, x2, v2]
    x1, v1, x2, v2 = y_arr_ctrl
    q_current = np.array([x1, x2])
    q_dot_current = np.array([v1, v2])

    # Get learned dynamics M_nn and h_nn
    # Note: lnn_model is the trained Lagrangian NN
    M_nn, h_nn = get_lnn_dynamics_terms(q_current, q_dot_current, lnn_model)

    # Desired accelerations (q_ddot_command)
    # For x2:
    x2_error = x2_desired_setpoint - x2
    x2_dot_error = 0.0 - v2 # Desired velocity is 0
    a2_command = Kp_x2 * x2_error + Kd_x2 * x2_dot_error # PD control for x2

    # For x1 (auxiliary control to keep things well-behaved):
    x1_error = x1_desired_setpoint - x1
    x1_dot_error = 0.0 - v1
    a1_command = Kp_x1 * x1_error + Kd_x1 * x1_dot_error # PD control for x1

    q_ddot_command = np.array([a1_command, a2_command])

    # Computed Torque Law: tau = M_nn * q_ddot_command + h_nn
    # We might also want to estimate and cancel Q_nc if significant
    # Q_nc1_est = -params_true['c1']*v1 - params_true['c2']*(v1-v2) - params_true['mu1_coulomb']*params_true['g_approx']*np.sign(v1)
    # Q_nc2_est = -params_true['c2']*(v2-v1) - params_true['mu2_coulomb']*params_true['g_approx']*np.sign(v2)
    # Q_nc_estimated = np.array([Q_nc1_est, Q_nc2_est])
    # tau = M_nn @ q_ddot_command + h_nn - Q_nc_estimated # With Q_nc compensation

    # For simplicity, let's first try without Q_nc compensation in the controller
    # The LNN's h_nn might have implicitly learned some of these effects if Q_nc wasn't perfectly subtracted during LNN training.
    tau = M_nn @ q_ddot_command + h_nn

    # Limit control inputs to avoid extreme values (optional but good practice)
    tau_max = 50.0
    tau = np.clip(tau, -tau_max, tau_max)
    
    control_inputs_history.append(np.concatenate(([t], tau)))
    return tau[0], tau[1] # tau1, tau2


# 4. Simulate the Controlled System
y0_controlled = [1.0, 0.0, 0.5, 0.0] # Initial conditions for controlled sim
t_span_controlled = [0, 15]
t_eval_controlled = np.linspace(t_span_controlled[0], t_span_controlled[1], 1000)

control_inputs_history = [] # Reset history

# Simulate with the controller
# The lambda function wraps the controller and passes its output to true_ode_controlled
sol_controlled = solve_ivp(
    lambda t, y: true_ode_controlled(t, y, params_true, computed_torque_controller),
    t_span_controlled,
    y0_controlled,
    dense_output=True,
    t_eval=t_eval_controlled,
    method='RK45', # Try 'LSODA' if 'RK45' struggles
    rtol=1e-5, atol=1e-7
)

Y_controlled_np = sol_controlled.y.T
t_controlled_np = sol_controlled.t

if sol_controlled.success:
    print("Controlled simulation successful.")
else:
    print(f"Controlled simulation FAILED: {sol_controlled.message}")

# Plotting results
control_inputs_history_np = np.array(control_inputs_history)

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(t_controlled_np, Y_controlled_np[:, 0], label='$x_1$ (Controlled)')
plt.plot(t_controlled_np, Y_controlled_np[:, 2], label='$x_2$ (Controlled)')
plt.axhline(x2_desired_setpoint, color='r', linestyle='--', label='$x_2$ Desired')
plt.axhline(x1_desired_setpoint, color='g', linestyle='--', label='$x_1$ Desired (aux)')
plt.title('Positions under LNN-based Computed Torque Control')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_controlled_np, Y_controlled_np[:, 1], label='$v_1$ (Controlled)')
plt.plot(t_controlled_np, Y_controlled_np[:, 3], label='$v_2$ (Controlled)')
plt.title('Velocities under Control')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
if control_inputs_history_np.size > 0:
    plt.plot(control_inputs_history_np[:, 0], control_inputs_history_np[:, 1], label='$\\tau_1$ (Control Force on m1)')
    plt.plot(control_inputs_history_np[:, 0], control_inputs_history_np[:, 2], label='$\\tau_2$ (Control Force on m2)')
plt.title('Control Inputs $\\tau$')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# For comparison, simulate uncontrolled system from same y0
sol_uncontrolled = solve_ivp(
    lambda t, y: true_ode_controlled(t, y, params_true, lambda t_sim, y_sim: (0,0)), # Zero control
    t_span_controlled, y0_controlled, dense_output=True, t_eval=t_eval_controlled
)
Y_uncontrolled_np = sol_uncontrolled.y.T

plt.figure(figsize=(10, 6))
plt.plot(t_eval_controlled, Y_uncontrolled_np[:, 2], 'b--', label='$x_2$ (Uncontrolled)')
plt.plot(t_controlled_np, Y_controlled_np[:, 2], 'r-', label='$x_2$ (LNN Controlled)')
plt.axhline(x2_desired_setpoint, color='k', linestyle=':', label='$x_2$ Desired')
plt.title('Comparison: $x_2$ Position With and Without LNN Control')
plt.xlabel('Time (s)')
plt.ylabel('$x_2$ Position (m)')
plt.legend()
plt.grid(True)
plt.show()