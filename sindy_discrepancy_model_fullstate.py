# Discrepancy Model for a Two-Mass System with Friction
# assuming full state measurement.    

# Simulate the true system to get x(t), v(t)
# Calculate the true accelerations a_true(t) (either by differentiating v(t) or, more accurately, by plugging x(t), v(t) back into the true ODEs).
# Calculate the accelerations predicted by the known model: a_known_model(t) = f_known(x(t), v(t), known_params).
# The discrepancy in acceleration is a_discrepancy(t) = a_true(t) - a_known_model(t)



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# --- System Parameters ---
params = {
    'm1': 1.0,  # mass 1
    'm2': 1.5,  # mass 2
    'k1': 1.0,  # spring constant for m1 to wall
    'k2': 0.8,  # spring constant between m1 and m2
    'c1': 0.1,  # damping coefficient for m1
    'c2': 0.15, # damping coefficient between m1 and m2
    # --- Friction Parameters (These are "unknown" to the discrepancy model initially) ---
    'mu1_coulomb': 0.2, # Coulomb friction coefficient for m1 (acting on m1*g, assume g=1 for simplicity or N)
    'mu2_coulomb': 0.15,# Coulomb friction coefficient for m2
}

# --- The "True" System ODEs (with friction) ---
def true_ode(t, x, p):
    x1, v1, x2, v2 = x
    # Forces on m1
    f_spring1 = -p['k1'] * x1
    f_damper1 = -p['c1'] * v1
    f_spring2_on1 = p['k2'] * (x2 - x1)
    f_damper2_on1 = p['c2'] * (v2 - v1)
    f_friction1 = -p['mu1_coulomb'] * np.sign(v1) if v1 != 0 else 0 # Coulomb

    # Forces on m2
    f_spring2_on2 = -p['k2'] * (x2 - x1)
    f_damper2_on2 = -p['c2'] * (v2 - v1)
    f_friction2 = -p['mu2_coulomb'] * np.sign(v2) if v2 != 0 else 0 # Coulomb

    a1 = (f_spring1 + f_damper1 + f_spring2_on1 + f_damper2_on1 + f_friction1) / p['m1']
    a2 = (f_spring2_on2 + f_damper2_on2 + f_friction2) / p['m2']

    # Return derivatives: [dx1/dt, dv1/dt, dx2/dt, dv2/dt]
    return [v1, a1, v2, a2]

# --- The "Known" Model ODEs (without friction) ---
# We can achieve this by setting friction parameters to 0 or by defining a separate function
def known_model_ode_terms(t, x, p_known):
    x1, v1, x2, v2 = x
    # Forces on m1 (known part)
    f_spring1_known = -p_known['k1'] * x1
    f_damper1_known = -p_known['c1'] * v1
    f_spring2_on1_known = p_known['k2'] * (x2 - x1)
    f_damper2_on1_known = p_known['c2'] * (v2 - v1)
    a1_known_contrib = (f_spring1_known + f_damper1_known + f_spring2_on1_known + f_damper2_on1_known) / p_known['m1']

    # Forces on m2 (known part)
    f_spring2_on2_known = -p_known['k2'] * (x2 - x1)
    f_damper2_on2_known = -p_known['c2'] * (v2 - v1)
    a2_known_contrib = (f_spring2_on2_known + f_damper2_on2_known) / p_known['m2']

    # The v1, v2 terms are definitional and perfectly known
    return [v1, a1_known_contrib, v2, a2_known_contrib]

# if we were using a known_ode function like true_ode,
# we would set params_known['mu1_coulomb'] = 0 and params_known['mu2_coulomb'] = 0.
# Here, known_model_ode_terms simply doesn't include them.
params_known = params.copy()

# --- Simulate True System to Get Data ---
fs = 100  # Sampling frequency
dt = 1 / fs  # Time step
t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 1+(t_span[1]-t_span[0]) * fs) # More points for SINDy
y0 = [1.0, 0, 0.5, 0] # Initial conditions: x1, v1, x2, v2

sol_true = solve_ivp(true_ode, t_span, y0, args=(params,), dense_output=True, t_eval=t_eval)
Y_true = sol_true.y.T # States: [x1, v1, x2, v2]
t_true = sol_true.t

# --- Calculate True Derivatives ---
# True derivatives dY/dt (including v1_dot_true, v2_dot_true)
# in practice, we would use numerical differentiation because we don't have access to the true accelerations directly.
dYdt_true = np.array([true_ode(t_true[i], Y_true[i], params) for i in range(len(t_true))])

# Predictions from the known model using TRUE states
dYdt_known_model_pred = np.array([known_model_ode_terms(t_true[i], Y_true[i], params_known) for i in range(len(t_true))])

# --- Calculate the Discrepancy ---
# The discrepancy is in the accelerations (dYdt[1] and dYdt[3])
# d(x1)/dt = v1 and d(x2)/dt = v2 are by definition and assumed perfectly known.
# So, dYdt_true[i,0] - dYdt_known_model_pred[i,0] should be 0 (it's v1_true - v1_true)
# Same for dYdt_true[i,2] - dYdt_known_model_pred[i,2] (it's v2_true - v2_true)

# Discrepancy in a1 and a2
a1_discrepancy = dYdt_true[:, 1] - dYdt_known_model_pred[:, 1]
a2_discrepancy = dYdt_true[:, 3] - dYdt_known_model_pred[:, 3]

# --- Build Feature Library for SINDy ---
# We want to find terms for the discrepancy in accelerations using states x1,v1,x2,v2
# For friction, we expect terms like sign(v1) and sign(v2).
# Let's create a feature library using the true states Y_true.
X_features = Y_true # Use all states [x1, v1, x2, v2] to build library

# Candidate functions
# For this problem, we are particularly interested in sign functions for Coulomb friction.
# We can also add polynomials.
poly_order = 1 # Simple linear terms + bias
pipeline = Pipeline([('poly', PolynomialFeatures(degree=poly_order, include_bias=True))])
Theta_poly = pipeline.fit_transform(X_features)
feature_names_poly = pipeline.named_steps['poly'].get_feature_names_out(['x1', 'v1', 'x2', 'v2'])

# Add custom features like sign(v1) and sign(v2)
# Note: SINDy libraries like PySINDy handle this more elegantly.
# Here, we'll manually add them.
sign_v1 = np.sign(X_features[:, 1]).reshape(-1, 1)
sign_v2 = np.sign(X_features[:, 3]).reshape(-1, 1)

# Combine polynomial features with custom sign features
Theta = np.concatenate((Theta_poly, sign_v1, sign_v2), axis=1)
feature_names = list(feature_names_poly) + ['sign(v1)', 'sign(v2)']

print(f"Feature names: {feature_names}")

# --- Apply SINDy (Sparse Regression) ---
# We will find coefficients for a1_discrepancy and a2_discrepancy separately.

sindy_alpha = 0.005 # Regularization strength for Lasso - a tuning parameter

# For a1_discrepancy
lasso_a1 = Lasso(alpha=sindy_alpha, fit_intercept=False, max_iter=5000) # fit_intercept=False because bias is in Theta
lasso_a1.fit(Theta, a1_discrepancy)
Xi_a1 = lasso_a1.coef_

# For a2_discrepancy
lasso_a2 = Lasso(alpha=sindy_alpha, fit_intercept=False, max_iter=5000)
lasso_a2.fit(Theta, a2_discrepancy)
Xi_a2 = lasso_a2.coef_

print("\nSINDy Coefficients for a1_discrepancy (friction on m1):")
for name, coef in zip(feature_names, Xi_a1):
    if np.abs(coef) > 1e-4: # Threshold for printing
        print(f"  {name}: {coef:.4f}")

print("\nSINDy Coefficients for a2_discrepancy (friction on m2):")
for name, coef in zip(feature_names, Xi_a2):
    if np.abs(coef) > 1e-4:
        print(f"  {name}: {coef:.4f}")

# Expected values:
# For a1: term 'sign(v1)' should have coefficient -params['mu1_coulomb']/params['m1']
# For a2: term 'sign(v2)' should have coefficient -params['mu2_coulomb']/params['m2']
expected_coef_a1_friction = -params['mu1_coulomb'] / params['m1']
expected_coef_a2_friction = -params['mu2_coulomb'] / params['m2']
print(f"\nExpected coefficient for sign(v1) in a1_disc: {expected_coef_a1_friction:.4f}")
print(f"Expected coefficient for sign(v2) in a2_disc: {expected_coef_a2_friction:.4f}")


# --- Reconstruct Discrepancy and Full Model ---
a1_discrepancy_sindy = Theta @ Xi_a1
a2_discrepancy_sindy = Theta @ Xi_a2

# Define the ODE for the full reconstructed model
def reconstructed_full_ode(t, y, p_known, Xi_a1_sindy, Xi_a2_sindy, feature_func_theta):
    x1, v1, x2, v2 = y

    # Known model part
    dYdt_known_part = known_model_ode_terms(t, y, p_known)
    a1_known_contrib = dYdt_known_part[1]
    a2_known_contrib = dYdt_known_part[3]

    # Discrepancy model part (evaluated using current states)
    current_states_for_features = np.array([[x1, v1, x2, v2]]) # Needs to be 2D for transform
    theta_current_poly = pipeline.transform(current_states_for_features) # Use the same pipeline
    sign_v1_current = np.sign(v1).reshape(-1,1)
    sign_v2_current = np.sign(v2).reshape(-1,1)
    theta_current = np.concatenate((theta_current_poly, sign_v1_current, sign_v2_current), axis=1)


    a1_disc_pred = theta_current @ Xi_a1_sindy
    a2_disc_pred = theta_current @ Xi_a2_sindy

    # Full model accelerations
    a1_full = a1_known_contrib + a1_disc_pred[0] # [0] because theta_current @ Xi is a (1,) array
    a2_full = a2_known_contrib + a2_disc_pred[0]

    return [v1, a1_full, v2, a2_full]

# Simulate the reconstructed full model
sol_reconstructed = solve_ivp(
    reconstructed_full_ode, t_span, y0,
    args=(params_known, Xi_a1, Xi_a2, None), # feature_func_theta not directly used here, embedded
    dense_output=True, t_eval=t_eval
)
Y_reconstructed = sol_reconstructed.y.T
t_reconstructed = sol_reconstructed.t

# Simulate the known model (without any discrepancy learning) for comparison
sol_known_only = solve_ivp(known_model_ode_terms, t_span, y0, args=(params_known,), dense_output=True, t_eval=t_eval)
Y_known_only = sol_known_only.y.T
t_known_only = sol_known_only.t


# --- Plotting ---
plt.figure(figsize=(15, 12))

# Plot 1: Positions
plt.subplot(3, 2, 1)
plt.plot(t_true, Y_true[:, 0], 'k-', label='True $x_1$')
plt.plot(t_known_only, Y_known_only[:, 0], 'b--', label='Known Model $x_1$')
plt.plot(t_reconstructed, Y_reconstructed[:, 0], 'r:', lw=2, label='Reconstructed $x_1$')
plt.xlabel('Time (s)')
plt.ylabel('Position $x_1$ (m)')
plt.legend()
plt.title('Mass 1 Position')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t_true, Y_true[:, 2], 'k-', label='True $x_2$')
plt.plot(t_known_only, Y_known_only[:, 2], 'b--', label='Known Model $x_2$')
plt.plot(t_reconstructed, Y_reconstructed[:, 2], 'r:', lw=2, label='Reconstructed $x_2$')
plt.xlabel('Time (s)')
plt.ylabel('Position $x_2$ (m)')
plt.legend()
plt.title('Mass 2 Position')
plt.grid(True)

# Plot 2: Velocities
plt.subplot(3, 2, 3)
plt.plot(t_true, Y_true[:, 1], 'k-', label='True $v_1$')
plt.plot(t_known_only, Y_known_only[:, 1], 'b--', label='Known Model $v_1$')
plt.plot(t_reconstructed, Y_reconstructed[:, 1], 'r:', lw=2, label='Reconstructed $v_1$')
plt.xlabel('Time (s)')
plt.ylabel('Velocity $v_1$ (m/s)')
plt.legend()
plt.title('Mass 1 Velocity')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t_true, Y_true[:, 3], 'k-', label='True $v_2$')
plt.plot(t_known_only, Y_known_only[:, 3], 'b--', label='Known Model $v_2$')
plt.plot(t_reconstructed, Y_reconstructed[:, 3], 'r:', lw=2, label='Reconstructed $v_2$')
plt.xlabel('Time (s)')
plt.ylabel('Velocity $v_2$ (m/s)')
plt.legend()
plt.title('Mass 2 Velocity')
plt.grid(True)

# Plot 3: Discrepancy in Accelerations
plt.subplot(3, 2, 5)
plt.plot(t_true, a1_discrepancy, 'k-', label='True $a_1$ Discrepancy')
plt.plot(t_true, a1_discrepancy_sindy, 'r--', lw=2, label='SINDy Identified $a_1$ Discrepancy')
plt.xlabel('Time (s)')
plt.ylabel('$\Delta a_1$ (m/s$^2$)')
plt.legend()
plt.title('Discrepancy in Acceleration of Mass 1')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(t_true, a2_discrepancy, 'k-', label='True $a_2$ Discrepancy')
plt.plot(t_true, a2_discrepancy_sindy, 'r--', lw=2, label='SINDy Identified $a_2$ Discrepancy')
plt.xlabel('Time (s)')
plt.ylabel('$\Delta a_2$ (m/s$^2$)')
plt.legend()
plt.title('Discrepancy in Acceleration of Mass 2')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot SINDy Coefficients
fig_coeffs, axes_coeffs = plt.subplots(1, 2, figsize=(12, 5))
coeffs_to_plot_a1 = {name: coef for name, coef in zip(feature_names, Xi_a1) if abs(coef) > 1e-4}
coeffs_to_plot_a2 = {name: coef for name, coef in zip(feature_names, Xi_a2) if abs(coef) > 1e-4}

axes_coeffs[0].bar(coeffs_to_plot_a1.keys(), coeffs_to_plot_a1.values())
axes_coeffs[0].set_title('SINDy Coefficients for $a_1$ Discrepancy')
axes_coeffs[0].set_ylabel('Coefficient Value')
axes_coeffs[0].tick_params(axis='x', rotation=45)
axes_coeffs[0].axhline(expected_coef_a1_friction, color='r', linestyle='--', label=f'Expected: {expected_coef_a1_friction:.2f} (sign(v1))')
axes_coeffs[0].legend()


axes_coeffs[1].bar(coeffs_to_plot_a2.keys(), coeffs_to_plot_a2.values())
axes_coeffs[1].set_title('SINDy Coefficients for $a_2$ Discrepancy')
axes_coeffs[1].set_ylabel('Coefficient Value')
axes_coeffs[1].tick_params(axis='x', rotation=45)
axes_coeffs[1].axhline(expected_coef_a2_friction, color='r', linestyle='--', label=f'Expected: {expected_coef_a2_friction:.2f} (sign(v2))')
axes_coeffs[1].legend()


plt.tight_layout()
plt.show()

