# Lagrangian Neural Network (LNN) for a 2DOF System with Friction and Damping
# This code implements a Lagrangian Neural Network (LNN) to learn the dynamics of a 2DOF system
# with friction and damping. The LNN is trained to predict the Euler-Lagrange equations
# of motion, incorporating non-conservative forces like damping and friction.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Define system parameters ---
params_true = {
    'm1': 1.0, 'm2': 1.5,
    'k1': 1.0, 'k2': 0.8,
    'c1': 0.1, 'c2': 0.15,
    'mu1_coulomb': 0.2, 'mu2_coulomb': 0.15
}

# --- Define the "true" system ODEs (with friction and damping) ---
def true_ode_lnn(t, y_arr, p):
    # The state vector is [x1, v1, x2, v2]
    x1, v1, x2, v2 = y_arr
    # Forces on m1
    f_spring1     = -p['k1'] * x1
    f_damper1     = -p['c1'] * v1
    f_spring2_on1 = p['k2'] * (x2 - x1)
    f_damper2_on1 = p['c2'] * (v2 - v1)
    f_friction1   = -p['mu1_coulomb'] * np.sign(v1) if v1 != 0 else 0

    # Forces on m2
    f_spring2_on2 = -p['k2'] * (x2 - x1)
    f_damper2_on2 = -p['c2'] * (v2 - v1)
    f_friction2   = -p['mu2_coulomb'] * np.sign(v2) if v2 != 0 else 0

    a1 = (f_spring1 + f_damper1 + f_spring2_on1 + f_damper2_on1 + f_friction1) / p['m1']
    a2 = (f_spring2_on2 + f_damper2_on2 + f_friction2) / p['m2']
    return [v1, a1, v2, a2]

# --- Generate training data ---
fs = 100 # Sampling frequency
ts = 1 / fs # Sampling time
t_span_train = [0, 20] # Longer time for more diverse data
n_points_train = int((t_span_train[1] - t_span_train[0])*fs)
t_eval_train = np.linspace(t_span_train[0], t_span_train[1], n_points_train)
y0_train = [1.0, 0.0, 0.5, 0.0] # initial state: x1, v1, x2, v2

# gather training data using the true ODE, in practice this would be from real data
sol_train = solve_ivp(true_ode_lnn, t_span_train, y0_train, args=(params_true,), dense_output=True, t_eval=t_eval_train)
Y_train_np = sol_train.y.T # (n_points, 4)
t_train_np = sol_train.t

# Calculate true accelerations (for loss calculation target)
# and known non-conservative forces (damping)
# this is cheating a bit, in practice we would not have this, we would have to differentiate the data
dYdt_train_np = np.array([true_ode_lnn(t_train_np[i], Y_train_np[i], params_true) for i in range(len(t_train_np))])
a1_true_np = dYdt_train_np[:, 1]
a2_true_np = dYdt_train_np[:, 3]

# Known non-conservative forces (damping only for this example)
# We assume we know c1 and c2, but not the friction terms.
# The LNN will try to learn an L_eff that accounts for springs + friction.
x1_d, v1_d, x2_d, v2_d = Y_train_np[:,0], Y_train_np[:,1], Y_train_np[:,2], Y_train_np[:,3]
Q_nc1_known_np = -params_true['c1'] * v1_d - params_true['c2'] * (v1_d - v2_d)
Q_nc2_known_np = -params_true['c2'] * (v2_d - v1_d)

# Target for EL equations: M*a - Q_nc_known
target_EL1_np = params_true['m1'] * a1_true_np - Q_nc1_known_np
target_EL2_np = params_true['m2'] * a2_true_np - Q_nc2_known_np

# Convert to PyTorch tensors
Y_train    = torch.tensor(Y_train_np, dtype=torch.float32)
target_EL1 = torch.tensor(target_EL1_np, dtype=torch.float32).unsqueeze(1)
target_EL2 = torch.tensor(target_EL2_np, dtype=torch.float32).unsqueeze(1)
t_train    = torch.tensor(t_train_np, dtype=torch.float32).unsqueeze(1)

# Extract q, q_dot, and acc for EL computation
q_train     = Y_train[:, [0, 2]] # x1, x2
q_dot_train = Y_train[:, [1, 3]] # v1, v2
# accelerations needed for d/dt (dL/dq_dot)
a_train = torch.tensor(dYdt_train_np[:, [1,3]], dtype=torch.float32) # a1, a2


# --- Define the Lagrangian Neural Network (LNN) ---
class LNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(LNN, self).__init__()
        # input_dim is for [x1, x2, v1, v2] or [q, q_dot]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim) # Output is scalar L

    def forward(self, x_state):
        x = torch.tanh(self.fc1(x_state))
        x = torch.tanh(self.fc2(x))
        L = self.fc3(x) # Learned Lagrangian
        return L

    def get_euler_lagrange_terms(self, q_current, q_dot_current, a_current):
        q = q_current.clone().detach().requires_grad_(True)
        q_dot = q_dot_current.clone().detach().requires_grad_(True)

        # DEBUG PRINTS
        # print(f"Initial q.requires_grad: {q.requires_grad}, q_dot.requires_grad: {q_dot.requires_grad}")
        # for name, param in self.named_parameters():
        #     if not param.requires_grad:
        #         print(f"WARNING: LNN Parameter {name} does not require grad!")

        state_for_L = torch.cat((q, q_dot), dim=1)
        # print(f"state_for_L.requires_grad: {state_for_L.requires_grad}, state_for_L.grad_fn: {state_for_L.grad_fn}")
        
        L_val = self.forward(state_for_L)
        # print(f"L_val.requires_grad: {L_val.requires_grad}, L_val.grad_fn: {L_val.grad_fn}")

        if L_val.grad_fn is None: # This is the critical check
            print("CRITICAL: L_val.grad_fn is None. Gradient tracking to L_val is broken.")
            print(f"  q.is_leaf: {q.is_leaf}, q.requires_grad: {q.requires_grad}, q.grad_fn: {q.grad_fn}")
            print(f"  q_dot.is_leaf: {q_dot.is_leaf}, q_dot.requires_grad: {q_dot.requires_grad}, q_dot.grad_fn: {q_dot.grad_fn}")
            print(f"  state_for_L.is_leaf: {state_for_L.is_leaf}, state_for_L.requires_grad: {state_for_L.requires_grad}, state_for_L.grad_fn: {state_for_L.grad_fn}")
            # Check model parameters' requires_grad status
            # for name, param_val in self.named_parameters():
            #     print(f"  Param {name} requires_grad: {param_val.requires_grad}")


        grad_L_outputs = torch.ones_like(L_val, requires_grad=False)
        
        dL_dq_tuple = torch.autograd.grad(outputs=L_val, inputs=q, grad_outputs=grad_L_outputs, create_graph=True, allow_unused=True)
        if dL_dq_tuple[0] is None:
            raise RuntimeError("dL_dq is None after autograd.grad call. L_val may not depend on q or graph broken.")
        dL_dq = dL_dq_tuple[0]

        p_tuple = torch.autograd.grad(outputs=L_val, inputs=q_dot, grad_outputs=grad_L_outputs, create_graph=True, allow_unused=True)
        if p_tuple[0] is None:
            raise RuntimeError("p (dL/dq_dot) is None. L_val may not depend on q_dot or graph broken.")
        p = p_tuple[0]

        dp_dt = torch.zeros_like(p)
        for i in range(p.shape[1]):
            p_i = p[:, i]
            grad_outputs_p_i_scalar = torch.ones_like(p_i, requires_grad=False)

            dp_i_dq_vec_tuple = torch.autograd.grad(outputs=p_i, inputs=q, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, retain_graph=True, allow_unused=True)
            dp_i_dq_vec = dp_i_dq_vec_tuple[0] if dp_i_dq_vec_tuple[0] is not None else torch.zeros_like(q)
            
            dp_i_dq_dot_vec_tuple = torch.autograd.grad(outputs=p_i, inputs=q_dot, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, retain_graph=True, allow_unused=True)
            dp_i_dq_dot_vec = dp_i_dq_dot_vec_tuple[0] if dp_i_dq_dot_vec_tuple[0] is not None else torch.zeros_like(q_dot)
            
            term1 = (dp_i_dq_vec * q_dot_current).sum(dim=1)
            term2 = (dp_i_dq_dot_vec * a_current).sum(dim=1)
            dp_dt[:, i] = term1 + term2
            
        EL_terms = dp_dt - dL_dq
        return EL_terms


# --- Training the LNN ---
lnn_model = LNN(input_dim=4, hidden_dim=128) # State is [x1,x2,v1,v2] and output is L
optimizer = optim.Adam(lnn_model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

epochs = 3000
batch_size = 128

print("Starting LNN training...")
for epoch in range(epochs):
    permutation = torch.randperm(q_train.size(0))
    epoch_loss = 0
    num_batches = 0

    for i in range(0, q_train.size(0), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_q, batch_q_dot, batch_a = q_train[indices], q_dot_train[indices], a_train[indices]
        batch_target_EL1, batch_target_EL2 = target_EL1[indices], target_EL2[indices]

        # EL_terms from LNN: [EL_for_x1, EL_for_x2]
        el_preds = lnn_model.get_euler_lagrange_terms(batch_q, batch_q_dot, batch_a)

        loss1 = loss_fn(el_preds[:, 0].unsqueeze(1), batch_target_EL1)
        loss2 = loss_fn(el_preds[:, 1].unsqueeze(1), batch_target_EL2)
        total_loss = loss1 + loss2

        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        num_batches +=1

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/num_batches:.6f}")

print("Training finished.")

# --- Evaluate the LNN (Qualitative Check) ---
# Compare EL terms from LNN with targets
lnn_model.eval()
# Even though we are "evaluating", this specific function needs gradients
# So, we re-enable them temporarily.
with torch.enable_grad():
    el_preds_full = lnn_model.get_euler_lagrange_terms(q_train, q_dot_train, a_train)

# Detach el_preds_full from the graph for plotting if it still has one
el_preds_full_np = el_preds_full.detach().numpy()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(t_train_np, target_EL1_np, 'k-', label='Target EL1 ($m_1 a_1 - Q_{nc1,known}$)')
plt.plot(t_train_np, el_preds_full_np[:, 0], 'r--', label='LNN Predicted EL1')
plt.xlabel('Time (s)')
plt.ylabel('Force Term for $x_1$')
plt.legend()
plt.title('Euler-Lagrange Term for $x_1$')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_train_np, target_EL2_np, 'k-', label='Target EL2 ($m_2 a_2 - Q_{nc2,known}$)')
plt.plot(t_train_np, el_preds_full_np[:, 1], 'r--', label='LNN Predicted EL2')
plt.xlabel('Time (s)')
plt.ylabel('Force Term for $x_2$')
plt.legend()
plt.title('Euler-Lagrange Term for $x_2$')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Simulate Dynamics using the Trained LNN (More Complex) ---
# To simulate, we need to solve M(q,q_dot)*q_ddot + C(q,q_dot)*q_dot + G(q) = Q_nc_known for q_ddot
# where M, C, G are derived from the learned L.
# M_ij = d^2 L / (d(q_dot_i) d(q_dot_j))
# The EL equation is: M*q_ddot + (dM/dq * q_dot - 0.5 * d(q_dot^T M q_dot)/dq_dot)*q_dot - dL/dq = Q_nc_known (simplified)
# Or, more directly from EL_term = Q_nc_known:
# (d/dt (dL/d(q_dot))) - dL/dq = Q_nc_known
# This involves solving a system for q_ddot (accelerations) at each step.
# Let's define the ODE function for the LNN.

# def lnn_ode_solver_func(t, y_arr_lnn, lnn_model_trained, p_sys):
#     # y_arr_lnn = [x1, v1, x2, v2]
#     q_np = np.array([[y_arr_lnn[0], y_arr_lnn[2]]]) # [[x1, x2]]
#     q_dot_np = np.array([[y_arr_lnn[1], y_arr_lnn[3]]]) # [[v1, v2]]

#     q_th = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
#     q_dot_th = torch.tensor(q_dot_np, dtype=torch.float32, requires_grad=True)
#     state_for_L = torch.cat((q_th, q_dot_th), dim=1)

#     # We need to find 'a_th' such that EL_terms(q_th, q_dot_th, a_th) = Q_nc_known_at_y
#     # This is an implicit equation for 'a_th'.
#     # For simpler systems or where M is easy to get, one can solve M*a = ...
#     # Here, we approximate M = d^2L / (dv dv) (Hessian of L w.r.t velocities)

#     L_val = lnn_model_trained(state_for_L)
#     dL_dq = torch.autograd.grad(L_val.sum(), q_th, create_graph=True)[0]
#     p_generalized = torch.autograd.grad(L_val.sum(), q_dot_th, create_graph=True)[0] # dL/dv

#     # Mass matrix M_ij = d(p_i)/d(v_j)
#     M = torch.zeros( (1, q_dot_th.shape[1], q_dot_th.shape[1]), dtype=torch.float32) # (batch, N_DOF, N_DOF)
#     for i in range(q_dot_th.shape[1]):
#         grad_p_i_q_dot = torch.autograd.grad(p_generalized[:, i].sum(), q_dot_th, create_graph=True, allow_unused=True)[0]
#         if grad_p_i_q_dot is not None:
#             M[0, i, :] = grad_p_i_q_dot[0, :]
#         else: # If p_i doesn't depend on q_dot_th (e.g. L is linear in q_dot), M might be tricky
#             M[0, i, i] = 1e-6 # Add small regularization if diagonal is zero, or handle properly

#     # Other terms in EL: C_terms = d/dt(dL/dv) - M*a = sum_k (d(dL/dvi)/dqk * vk)
#     # This is the part of dp/dt that doesn't involve 'a'
#     C_terms_vec = torch.zeros_like(p_generalized)
#     for i in range(p_generalized.shape[1]):
#         grad_p_i_q = torch.autograd.grad(p_generalized[:, i].sum(), q_th, create_graph=True, allow_unused=True)[0]
#         if grad_p_i_q is not None:
#             C_terms_vec[0,i] = (grad_p_i_q * q_dot_th).sum()


#     # Target for M*a: Q_nc_known + dL/dq - C_terms_vec
#     # Q_nc_known at current state y_arr_lnn
#     Q_nc1_curr = -p_sys['c1'] * y_arr_lnn[1] - p_sys['c2'] * (y_arr_lnn[1] - y_arr_lnn[3])
#     Q_nc2_curr = -p_sys['c2'] * (y_arr_lnn[3] - y_arr_lnn[1])
#     Q_nc_known_curr = torch.tensor([[Q_nc1_curr, Q_nc2_curr]], dtype=torch.float32)

#     rhs_for_M_a = Q_nc_known_curr + dL_dq - C_terms_vec

#     # Solve M*a = rhs_for_M_a for 'a'
#     # Add regularization to M for stability if needed
#     M_reg = M[0] + torch.eye(M.shape[1]) * 1e-6
#     try:
#         # M_inv = torch.linalg.inv(M_reg) # PyTorch 1.8+
#         # a_pred_th = (M_inv @ rhs_for_M_a.T).T
#         a_pred_th, _ = torch.linalg.solve(M_reg, rhs_for_M_a.T) # More stable
#         a_pred_th = a_pred_th.T

#     except RuntimeError as e:
#         print(f"Matrix inversion/solve failed: {e}. Using zero acceleration.")
#         a_pred_th = torch.zeros_like(q_dot_th)


#     a_pred_np = a_pred_th.detach().numpy().flatten()
#     return [y_arr_lnn[1], a_pred_np[0], y_arr_lnn[3], a_pred_np[1]]
def lnn_ode_solver_func(t, y_arr_lnn, lnn_model_trained, p_sys):
    print(f"t = {t:.3f}, y = {y_arr_lnn}")

    with torch.enable_grad(): # Ensure gradients are enabled for this function's scope
        q_np = np.array([[y_arr_lnn[0], y_arr_lnn[2]]])
        q_dot_np = np.array([[y_arr_lnn[1], y_arr_lnn[3]]])

        q_th = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
        q_dot_th = torch.tensor(q_dot_np, dtype=torch.float32, requires_grad=True)
        state_for_L = torch.cat((q_th, q_dot_th), dim=1)

        L_val = lnn_model_trained(state_for_L)
        
        grad_L_outputs = torch.ones_like(L_val)
        dL_dq = torch.autograd.grad(L_val, q_th, grad_outputs=grad_L_outputs, create_graph=True)[0]
        p_generalized = torch.autograd.grad(L_val, q_dot_th, grad_outputs=grad_L_outputs, create_graph=True)[0]

        M = torch.zeros((1, q_dot_th.shape[1], q_dot_th.shape[1]), dtype=torch.float32)
        C_terms_vec = torch.zeros_like(p_generalized)

        for i in range(q_dot_th.shape[1]):
            p_i = p_generalized[:, i]
            grad_outputs_p_i_scalar = torch.ones_like(p_i)

            grad_p_i_q_dot = torch.autograd.grad(p_i, q_dot_th, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, allow_unused=True)[0]
            if grad_p_i_q_dot is not None:
                M[0, i, :] = grad_p_i_q_dot[0, :]
            else:
                M[0, i, i] = 1e-6 

            grad_p_i_q = torch.autograd.grad(p_i, q_th, grad_outputs=grad_outputs_p_i_scalar, create_graph=True, allow_unused=True)[0] 
            if grad_p_i_q is not None:
                C_terms_vec[0,i] = (grad_p_i_q * q_dot_th).sum() 

        Q_nc1_curr = -p_sys['c1'] * y_arr_lnn[1] - p_sys['c2'] * (y_arr_lnn[1] - y_arr_lnn[3])
        Q_nc2_curr = -p_sys['c2'] * (y_arr_lnn[3] - y_arr_lnn[1])
        Q_nc_known_curr = torch.tensor([[Q_nc1_curr, Q_nc2_curr]], dtype=torch.float32)

        rhs_for_M_a = Q_nc_known_curr + dL_dq - C_terms_vec
        M_reg = M[0] + torch.eye(M.shape[1], device=M.device) * 1e-6 # Added device=M.device
        
        try:
            # For PyTorch 1.9+ torch.linalg.solve is preferred
            if hasattr(torch.linalg, 'solve'):
                 a_pred_th = torch.linalg.solve(M_reg, rhs_for_M_a.T) 
                 a_pred_th = a_pred_th.T
            else: # Fallback for older PyTorch
                 M_inv = torch.inverse(M_reg) 
                 a_pred_th = (M_inv @ rhs_for_M_a.T).T

        except RuntimeError as e:
            print(f"Matrix inversion/solve failed in LNN ODE: {e}. Using zero acceleration.")
            a_pred_th = torch.zeros_like(q_dot_th)

        a_pred_np = a_pred_th.detach().numpy().flatten()
    return [y_arr_lnn[1], a_pred_np[0], y_arr_lnn[3], a_pred_np[1]]

print("\nSimulating with trained LNN...")
# Use a shorter time span for simulation to see initial behavior
t_span_test = [0, 5]
t_eval_test = np.linspace(t_span_test[0], t_span_test[1],500)
y0_test = y0_train # Use same initial conditions

# Simulate true system for comparison on test interval
sol_true_test = solve_ivp(true_ode_lnn, t_span_test, y0_test, args=(params_true,), dense_output=True, t_eval=t_eval_test)
Y_true_test_np = sol_true_test.y.T

# Simulate with LNN
# Wrap lnn_model_trained to avoid issues with pickling if multiprocessing were used by solve_ivp
lnn_model.eval() # Ensure model is in eval mode
sol_lnn_test = solve_ivp(
    lambda t, y: lnn_ode_solver_func(t, y, lnn_model, params_true),
    t_span_test, y0_test, dense_output=True, t_eval=t_eval_test,
    # method='RK45' # Or 'LSODA' for stiff, 'DOP853'
)
Y_lnn_test_np = sol_lnn_test.y.T

plt.figure(figsize=(12, 8))
labels = ['$x_1$', '$v_1$', '$x_2$', '$v_2$']
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t_eval_test, Y_true_test_np[:, i], 'k-', label=f'True {labels[i]}')
    plt.plot(sol_lnn_test.t, Y_lnn_test_np[:, i], 'r--', label=f'LNN Predicted {labels[i]}')
    plt.xlabel('Time (s)')
    plt.ylabel(labels[i])
    plt.legend()
    plt.grid(True)
plt.suptitle('System Simulation: True vs. LNN-derived Dynamics')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()