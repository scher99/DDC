import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Assume all previous code (params, true_ode_lnn, data gen) has run ---
# We need Y_train, target_EL1, target_EL2, params_true, true_ode_lnn

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


# Ensure Y_train and targets are available as tensors
Y_train_np = Y_train.numpy() # If Y_train exists as tensor
target_EL1_np = target_EL1.numpy().flatten() # Flatten to (N,)
target_EL2_np = target_EL2.numpy().flatten() # Flatten to (N,)
target_EL_np = np.stack((target_EL1_np, target_EL2_np), axis=1) # Shape (N, 2)

Y_train_th = torch.tensor(Y_train_np, dtype=torch.float32)
target_EL_th = torch.tensor(target_EL_np, dtype=torch.float32)

# --- 1. Define the "Easier" Neural Network (NN_Forces) ---
class NNForces(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super(NNForces, self).__init__()
        # Input: [x1, v1, x2, v2]
        # Output: [EL1_pred, EL2_pred]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_state):
        x = torch.tanh(self.fc1(x_state))
        x = torch.tanh(self.fc2(x))
        el_terms = self.fc3(x)
        return el_terms

# --- 2. Train the NN_Forces Model ---
nn_forces_model = NNForces(input_dim=4, hidden_dim=128)
optimizer_nf = optim.Adam(nn_forces_model.parameters(), lr=1e-3)
loss_fn_nf = nn.MSELoss()

epochs_nf = 3000 # Might need adjustment
batch_size_nf = 128

print("\nStarting NN_Forces training...")
nn_forces_model.train() # Set to training mode
for epoch in range(epochs_nf):
    permutation = torch.randperm(Y_train_th.size(0))
    epoch_loss = 0
    num_batches = 0

    for i in range(0, Y_train_th.size(0), batch_size_nf):
        optimizer_nf.zero_grad()
        indices = permutation[i:i+batch_size_nf]
        batch_Y = Y_train_th[indices]
        batch_target_EL = target_EL_th[indices]

        el_preds = nn_forces_model(batch_Y) # Direct prediction

        total_loss = loss_fn_nf(el_preds, batch_target_EL)

        total_loss.backward()
        optimizer_nf.step()
        epoch_loss += total_loss.item()
        num_batches +=1

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs_nf}], Loss: {epoch_loss/num_batches:.6f}")

print("NN_Forces training finished.")

# --- 3. Define the "Easier" ODE Function ---
def nn_forces_ode(t, y_arr, model_nf, p):
    # y_arr = [x1, v1, x2, v2]
    x1, v1, x2, v2 = y_arr

    # Prepare input for the NN
    state_th = torch.tensor([[x1, v1, x2, v2]], dtype=torch.float32)

    # Predict EL terms using the NN
    model_nf.eval() # Ensure model is in eval mode
    with torch.no_grad(): # No need for gradients during this prediction
        el_terms_pred = model_nf(state_th).numpy().flatten() # (2,)

    # Calculate known non-conservative forces
    Q_nc1_known = -p['c1'] * v1 - p['c2'] * (v1 - v2)
    Q_nc2_known = -p['c2'] * (v2 - v1)

    # Calculate accelerations: a = (EL_terms_pred + Q_nc_known) / m
    a1 = (el_terms_pred[0] + Q_nc1_known) / p['m1']
    a2 = (el_terms_pred[1] + Q_nc2_known) / p['m2']

    return [v1, a1, v2, a2]

# --- 4. Simulate Using the "Easier" ODE Function ---
print("\nSimulating with trained NN_Forces...")
t_span_test = [0, 40] # Use a longer span now
t_eval_test = np.linspace(t_span_test[0], t_span_test[1], 1000)
y0_test = [1.0, 0.0, 0.5, 0.0] # Use same initial conditions as training

# Simulate true system for comparison
sol_true_test = solve_ivp(true_ode_lnn, t_span_test, y0_test, args=(params_true,), dense_output=True, t_eval=t_eval_test)
Y_true_test_np = sol_true_test.y.T

# Simulate with NN_Forces
sol_nnf_test = solve_ivp(
    lambda t, y: nn_forces_ode(t, y, nn_forces_model, params_true),
    t_span_test, y0_test, dense_output=True, t_eval=t_eval_test,
    method='RK45', rtol=1e-5, atol=1e-7
)
Y_nnf_test_np = sol_nnf_test.y.T

# --- 5. Plot the Results ---
plt.figure(figsize=(12, 8))
labels = ['$x_1$', '$v_1$', '$x_2$', '$v_2$']
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t_eval_test, Y_true_test_np[:, i], 'k-', lw=1.5, label=f'True {labels[i]}')
    if sol_nnf_test.success:
        plt.plot(t_eval_test, Y_nnf_test_np[:, i], 'g--', lw=2, label=f'NN_Forces Pred {labels[i]}')
    else:
        plt.text(0.5, 0.5, 'NN_Forces Sim Failed', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.xlabel('Time (s)')
    plt.ylabel(labels[i])
    plt.legend()
    plt.grid(True)
plt.suptitle('System Simulation: True vs. NN_Forces (Easier LNN-Inspired) Dynamics')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

if not sol_nnf_test.success:
    print(f"NN_Forces simulation failed: {sol_nnf_test.message}")
else:
    print("NN_Forces simulation successful.")