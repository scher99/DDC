import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Assuming the LNN model (`lnn_model`) is trained and available
# and other necessary functions like `get_lnn_dynamics_terms`, `true_ode_controlled`
# are defined as in the previous Computed Torque Controller example.

# --- Placeholder LNN Setup (same as previous example, ensure it runs) ---
print("Setting up and (re)training a placeholder LNN model for the controller example...")
# ... (copy the LNN setup and minimal training from the Computed Torque example) ...
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

# --- Functions from Computed Torque Example (true_ode_controlled, get_lnn_dynamics_terms) ---
# 1. Modified True System ODE to include control inputs tau = [tau1, tau2]
def true_ode_controlled(t, y_arr, p, tau_func, t_offset=0.0):
    x1, v1, x2, v2 = y_arr; tau1, tau2 = tau_func(t - t_offset, y_arr)
    f_spring1 = -p['k1'] * x1; f_damper1 = -p['c1'] * v1
    f_spring2_on1 = p['k2'] * (x2 - x1); f_damper2_on1 = p['c2'] * (v2 - v1)
    f_friction1 = -p['mu1_coulomb'] * p['g_approx'] * np.sign(v1) if v1 != 0 else 0
    f_spring2_on2 = -p['k2'] * (x2 - x1); f_damper2_on2 = -p['c2'] * (v2 - v1)
    f_friction2 = -p['mu2_coulomb'] * p['g_approx'] * np.sign(v2) if v2 != 0 else 0
    a1 = (f_spring1 + f_damper1 + f_spring2_on1 + f_damper2_on1 + f_friction1 + tau1) / p['m1']
    a2 = (f_spring2_on2 + f_damper2_on2 + f_friction2 + tau2) / p['m2']
    return [v1, a1, v2, a2]

# 2. Function to get M_nn(q, q_dot) and h_nn(q, q_dot) from the LNN
def get_lnn_dynamics_terms(q_np, q_dot_np, lnn_model_trained):
    with torch.enable_grad():
        q_th = torch.tensor([q_np], dtype=torch.float32, requires_grad=True)
        q_dot_th = torch.tensor([q_dot_np], dtype=torch.float32, requires_grad=True)
        state_for_L = torch.cat((q_th, q_dot_th), dim=1)
        L_val = lnn_model_trained(state_for_L)
        grad_L_outputs = torch.ones_like(L_val)
        dL_dq = torch.autograd.grad(L_val, q_th, grad_outputs=grad_L_outputs, create_graph=True)[0]
        p_generalized = torch.autograd.grad(L_val, q_dot_th, grad_outputs=grad_L_outputs, create_graph=True)[0]
        num_dof = q_dot_th.shape[1]
        M_nn = torch.zeros((num_dof, num_dof), dtype=torch.float32, device=q_th.device)
        C_terms_nn_vec = torch.zeros_like(p_generalized)
        for i in range(num_dof):
            p_i = p_generalized[:, i]; grad_outputs_p_i_scalar = torch.ones_like(p_i)
            grad_p_i_wrt_q_dot = torch.autograd.grad(p_i, q_dot_th, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, allow_unused=True)[0]
            if grad_p_i_wrt_q_dot is not None: M_nn[i, :] = grad_p_i_wrt_q_dot[0, :]
            else: M_nn[i, i] = 1e-6
            grad_p_i_wrt_q = torch.autograd.grad(p_i, q_th, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, allow_unused=True)[0]
            if grad_p_i_wrt_q is not None: C_terms_nn_vec[0,i] = (grad_p_i_wrt_q * q_dot_th).sum()
        h_nn_vec = C_terms_nn_vec - dL_dq
    return M_nn.detach().cpu().numpy(), h_nn_vec[0].detach().cpu().numpy()
# --- End of copied functions ---

# Optimal-Like Controller (LQR-inspired gains)
# Define Q and R matrices for LQR "spirit" - these are tuning parameters
# Penalize deviations in q and q_dot, and control effort tau
# For this example, we'll translate this into Kp_lqr and Kd_lqr gains directly.
# If we had a linear model A, B, we'd solve ARE. Here, we tune Kp, Kd.
# A larger Kp/Kd means more aggressive control (like smaller R in LQR).
# A smaller Kp/Kd means less aggressive (like larger R in LQR).

# Desired equilibrium state (setpoint)
q_ref = np.array([0.0, 0.0])  # Desired x1=0, x2=0
q_dot_ref = np.array([0.0, 0.0]) # Desired v1=0, v2=0

# LQR-inspired gain matrices (diagonal for simplicity, need tuning)
# These are effectively state feedback gains K = [Kp_lqr, Kd_lqr] for x_ddot = -K*error
# We want to control x2 primarily, x1 secondarily.
Kp_lqr_diag = np.array([5.0, 25.0]) # Penalize x1 error, Penalize x2 error more
Kd_lqr_diag = np.array([2.0, 10.0]) # Penalize v1 error, Penalize v2 error more
Kp_lqr = np.diag(Kp_lqr_diag)
Kd_lqr = np.diag(Kd_lqr_diag)

# Store control inputs for plotting
optimal_like_control_history = []

def optimal_like_lqr_controller(t, y_arr_ctrl):
    global optimal_like_control_history
    x1, v1, x2, v2 = y_arr_ctrl
    q_current = np.array([x1, x2])
    q_dot_current = np.array([v1, v2])

    # Get learned dynamics
    M_nn, h_nn = get_lnn_dynamics_terms(q_current, q_dot_current, lnn_model)

    # Calculate errors
    q_error = q_current - q_ref # Note: some formulations use q_ref - q_current
    q_dot_error = q_dot_current - q_dot_ref

    # LQR-like command for acceleration (stabilizing to q_ref, q_dot_ref)
    # q_ddot_command = -Kp_lqr @ q_error - Kd_lqr @ q_dot_error
    # If error is (ref - current): q_ddot_command = Kp_lqr @ (q_ref - q_current) + Kd_lqr @ (q_dot_ref - q_dot_current)
    # Let's use (ref - current) for standard PD error definition
    q_ddot_command = Kp_lqr @ (q_ref - q_current) + Kd_lqr @ (q_dot_ref - q_dot_current)


    # Control Law: tau = M_nn * q_ddot_command + h_nn
    # (Compensates for learned dynamics and imposes LQR-like feedback)
    tau = M_nn @ q_ddot_command + h_nn

    tau_max = 50.0 # Limit control inputs
    tau = np.clip(tau, -tau_max, tau_max)

    optimal_like_control_history.append(np.concatenate(([t], tau)))
    return tau[0], tau[1]

# Simulation
y0_optimal_ctrl = [1.0, 0.0, 0.5, 0.0]
t_span_optimal_ctrl = [0, 20] # Increased time span
t_eval_optimal_ctrl = np.linspace(t_span_optimal_ctrl[0], t_span_optimal_ctrl[1], 1500)

optimal_like_control_history = [] # Reset

sol_optimal_like = solve_ivp(
    lambda t, y: true_ode_controlled(t, y, params_true, optimal_like_lqr_controller),
    t_span_optimal_ctrl,
    y0_optimal_ctrl,
    dense_output=True,
    t_eval=t_eval_optimal_ctrl,
    method='RK45',
    rtol=1e-5, atol=1e-7
)

Y_optimal_like_np = sol_optimal_like.y.T
t_optimal_like_np = sol_optimal_like.t

if sol_optimal_like.success:
    print("Optimal-like LQR simulation successful.")
else:
    print(f"Optimal-like LQR simulation FAILED: {sol_optimal_like.message}")


# Plotting
optimal_like_control_history_np = np.array(optimal_like_control_history)

plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(t_optimal_like_np, Y_optimal_like_np[:, 0], label='$x_1$ (Optimal-Like Ctrl)')
plt.plot(t_optimal_like_np, Y_optimal_like_np[:, 2], label='$x_2$ (Optimal-Like Ctrl)')
plt.axhline(q_ref[1], color='r', linestyle='--', label='$x_2$ Desired (0.0)')
plt.axhline(q_ref[0], color='g', linestyle='--', label='$x_1$ Desired (0.0)')
plt.title('Positions under LNN-based Optimal-Like Control')
plt.xlabel('Time (s)'); plt.ylabel('Position (m)'); plt.legend(); plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_optimal_like_np, Y_optimal_like_np[:, 1], label='$v_1$ (Optimal-Like Ctrl)')
plt.plot(t_optimal_like_np, Y_optimal_like_np[:, 3], label='$v_2$ (Optimal-Like Ctrl)')
plt.title('Velocities under Control')
plt.xlabel('Time (s)'); plt.ylabel('Velocity (m/s)'); plt.legend(); plt.grid(True)

plt.subplot(3, 1, 3)
if optimal_like_control_history_np.size > 0:
    plt.plot(optimal_like_control_history_np[:, 0], optimal_like_control_history_np[:, 1], label='$\\tau_1$')
    plt.plot(optimal_like_control_history_np[:, 0], optimal_like_control_history_np[:, 2], label='$\\tau_2$')
plt.title('Control Inputs $\\tau$')
plt.xlabel('Time (s)'); plt.ylabel('Force (N)'); plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()