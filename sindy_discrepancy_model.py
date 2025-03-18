#!/usr/bin/env python3.8


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pysindy as ps
from sklearn.linear_model import Lasso


def average_mutual_information(x, tau, n_bins=32):
    """
    Compute the average mutual information (AMI) between x(t) and x(t+tau)
    using a histogram-based estimator.
    
    Parameters:
        x      : 1D numpy array containing the time series.
        tau    : Delay (in number of time steps) between x(t) and x(t+tau).
        n_bins : Number of bins to use in the histogram.
    
    Returns:
        mi : Estimated mutual information (in nats) for the given tau.
    """
    # Create two vectors: one for x(t) and one for x(t+tau)
    x1 = x[:-tau]
    x2 = x[tau:]
    
    # Compute the 2D histogram for the joint distribution of (x(t), x(t+tau))
    hist_2d, x_edges, y_edges = np.histogram2d(x1, x2, bins=n_bins)
    
    # Normalize the joint histogram to obtain a joint probability density
    pxy = hist_2d / np.sum(hist_2d)
    
    # Compute marginal probability distributions by summing over rows and columns
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Compute mutual information using the formula:
    # MI = sum_{i,j} pxy(i,j) * log(pxy(i,j) / (px(i)*py(j)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi


# -------------------------------
# 1. Define and simulate the full system
# -------------------------------
def two_mass_system(t, z, m1, m2, k1, k2, c1, c2):
    """
    Two-mass spring-damper system:
      Mass 1: connected to ground by spring (k1) and damper (c1)
      Mass 1 and mass 2: coupled by spring-damper (k2, c2)
    
    State vector: z = [x1, v1, x2, v2]
    """
    x1, v1, x2, v2 = z
    dx1dt = v1
    dv1dt = (-k1*x1 + k2*(x2 - x1) - c1*v1) / m1
    dx2dt = v2
    dv2dt = (-k2*(x2 - x1) - c2*v2) / m2
    return [dx1dt, dv1dt, dx2dt, dv2dt]

# System parameters
m1, m2 = 1.0, 1.0
k1, k2 = 2.0, 1.0
c1, c2 = 0.1, 0.1

# Initial conditions: [x1, v1, x2, v2]
z0 = [1.0, 0.0, 0.0, 0.0]

# Time span for simulation
t_start, t_end = 0, 20
n_points = 1000
t = np.linspace(t_start, t_end, n_points)
dt = t[1] - t[0]

# Solve the ODE
sol = solve_ivp(two_mass_system, [t_start, t_end], z0, args=(m1, m2, k1, k2, c1, c2), t_eval=t)

# Extract the full state (for simulation/truth) and the measurement
x1 = sol.y[0]   # hidden
v1 = sol.y[1]   # hidden
x2 = sol.y[2]   # measured
v2_true = sol.y[3]  # true velocity of mass 2 (we pretend we don't measure this directly)

# -------------------------------
# 2. Pretend we only measure x2(t)
# -------------------------------
# In a realistic setting you might only have x2. Here we simulate that scenario.
x2_data = x2.reshape(-1, 1)  # shape (n_samples, 1)

x2_data = x2_data #+ np.random.rand(x2_data.shape[0], 1) * 0.01  # add noise to x2

# Compute AMI for a range of delays tau
taus = np.arange(1, 100)  # delays from 1 to 99 time steps
ami_values = [average_mutual_information(x2, tau, n_bins=32) for tau in taus]

# Plot the Average Mutual Information vs. tau
plt.figure(figsize=(8, 4))
plt.plot(taus, ami_values, 'b.-')
plt.xlabel('Delay (tau)')
plt.ylabel('Average Mutual Information (nats)')
plt.title('AMI vs. Delay')
plt.grid(True)
plt.show()

# -------------------------------
# 3. Estimate derivatives from measured data
# -------------------------------
# We compute the first derivative (v2_est) and the second derivative (acceleration)
# using a simple finite difference (here, np.gradient works well for our smooth data).
v2_est = np.gradient(x2_data.squeeze(), dt)
dv2_est = np.gradient(v2_est, dt)

# -------------------------------
# 4. Define a partial (incomplete) physics model for x2
# -------------------------------
# Suppose you know (or assume) that the second mass is only damped:
#    m2 * x2'' = -c2 * x2'
# Since m2 = 1, the partial model predicts:
partial_model_acc = -c2 * v2_est / m2

# -------------------------------
# 5. Compute the discrepancy between the true acceleration and the partial model
# -------------------------------
# The full system actually follows:
#    x2'' = (-k2*(x2-x1) - c2*v2) / m2
# so the discrepancy (due to the unobserved coupling with mass 1) is:
discrepancy = dv2_est - partial_model_acc
# (i.e., discrepancy = measured x2'' - ( -c2*v2_est ) )

# -------------------------------
# 6. Use delay embedding + SINDy to learn a model for the discrepancy
# -------------------------------
# The idea: even though we only measure x2, its history (or delay coordinates) can 
# help us recover the effect of the hidden states.
#
# We will create an augmented (delay embedded) state of x2. For example, we choose
# an embedding with delays corresponding to [x2(t), x2(t - τ), x2(t - 2τ)].
tau = 43  # delay in number of time steps (adjust as needed)
delays = [0, tau, 2*tau, 3*tau]  # delays in indices

def delay_embed(data, delays):
    """
    Create a delay-embedded version of the data.
    data: 2D array of shape (n_samples, n_features)
    delays: list of integer delays (in number of samples)
    Returns an array of shape (n_samples - max(delays), n_features * len(delays))
    """
    n_samples = data.shape[0]
    max_delay = max(delays)
    embed_list = []
    for d in delays:
        embed_list.append(data[d:n_samples - max_delay + d])
    return np.hstack(embed_list)

# Create the delay-embedded feature matrix from x2_data.
X_embedded = delay_embed(x2_data, delays)
# Align the discrepancy target: we discard the first max(delays) samples.
max_delay = max(delays)
d_target = discrepancy[max_delay:]

# -------------------------------
# 7. Build a candidate function library and perform sparse regression
# -------------------------------
# We use a polynomial library (up to degree 3) from PySINDy.
library = ps.PolynomialLibrary(degree=1)
Theta = library.fit_transform(X_embedded)

# Use a sparse regression method (here, Lasso) to find coefficients xi in:
#      discrepancy ≈ Θ(X_embedded) · xi
# You can adjust the regularization parameter (alpha) to control sparsity.
lasso = Lasso(alpha=0.001, fit_intercept=False, max_iter=10000)
lasso.fit(Theta, d_target)
xi = lasso.coef_

# Get the names of the candidate functions from the library.
feature_names = library.get_feature_names()

print("Learned model for the discrepancy (nonzero terms):")
for coef, name in zip(xi, feature_names):
    if np.abs(coef) > 1e-5:
        print(f"  {name:20s} : {coef: .5f}")

# -------------------------------
# 8. Compare the learned discrepancy model with the true discrepancy
# -------------------------------
d_pred = Theta @ xi

plt.figure(figsize=(10, 4))
plt.plot(t[max_delay:], d_target, 'k', label='True discrepancy')
plt.plot(t[max_delay:], d_pred, 'r--', label='Learned discrepancy model')
plt.xlabel('Time')
plt.ylabel('Discrepancy [m/s²]')
plt.title('Comparison of True and Learned Discrepancy')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 9. (Optional) How to build a hybrid model
# -------------------------------
# In a hybrid approach you could combine the known partial physics model with the
# learned discrepancy:
#
#    x2'' = -c2 * x2' + g(delay_embedded(x2))
#
# You would then simulate your system using this model. Note that because g(·) depends
# on delay coordinates, one would have to store the history of x2 during simulation.
#
# This example shows how SINDy can be applied to learn the discrepancy so that the effect
# of the hidden states (x1 and v1) is captured from the measurement of x2 only.

# -------------------------------
# 9. Simulate the hybrid (physics-informed) model
# -------------------------------
# The hybrid model uses:
#    x2'' = - c2 * x2' + g( x2(t), x2(t-τ), x2(t-2τ) )
# where g is the discrepancy function learned from SINDy.
#
# Because g depends on delayed values of x2, we simulate in discrete time (Euler integration)
# and use the known history from the full simulation to seed the delay.
#
# Initialize arrays for the hybrid simulation of x2 and its velocity.
n_steps = len(t)
hybrid_x2 = np.zeros(n_steps)
hybrid_v2 = np.zeros(n_steps)

# Use the true simulation to provide the history for the first max_delay steps.
hybrid_x2[:max_delay+1] = x2[:max_delay+1]
hybrid_v2[:max_delay+1] = v2_est[:max_delay+1]

# Simulate forward using Euler integration.
# At each step n (starting from n = max_delay) we compute the delay-embedded state and predict g.
for n in range(max_delay, n_steps - 1):
    # Indices for the delays: current, τ, and 2τ steps back.
    idx0 = n
    idx1 = n - tau
    idx2 = n - 2*tau
    idx3 = n - 3*tau
    # Form the delay-embedded vector: [x2(n), x2(n-τ), x2(n-2τ)]
    delay_vec = np.array([hybrid_x2[idx0], hybrid_x2[idx1], hybrid_x2[idx2], hybrid_x2[idx3]])
    # Reshape to (1, -1) for library evaluation
    delay_vec_reshaped = delay_vec.reshape(1, -1)
    # Evaluate the candidate library functions on this vector.
    Theta_delay = library.transform(delay_vec_reshaped)
    # Compute the learned discrepancy g(delay_vec)
    g_pred = np.dot(Theta_delay, xi)[0]
    
    # The hybrid acceleration is the sum of the partial physics term and the discrepancy.
    a_hybrid = - c2 * hybrid_v2[n]/m2 + g_pred
    # Euler integration to update velocity and position.
    hybrid_v2[n+1] = hybrid_v2[n] + dt * a_hybrid
    hybrid_x2[n+1] = hybrid_x2[n] + dt * hybrid_v2[n]

# -------------------------------
# 7. Plot and compare results
# -------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, x2, 'k', label='Full simulation (x2 true)')
plt.plot(t, v2_true, 'k--', label='Full simulation (v2 true)')
plt.plot(t, hybrid_x2, 'r', label='Hybrid simulation (x2 hybrid)')
plt.plot(t, hybrid_v2, 'r--', label='Hybrid simulation (v2 hybrid)')
plt.xlabel('Time')
plt.ylabel('x2/v2 (position/velocity)')
plt.title('Comparison: Full Simulation vs. Hybrid (Physics-Informed) Model')
plt.legend()
plt.tight_layout()
plt.show()




print('done')

# =============================================================================
# 4. Learn a model for the discrepancy without delay embeddings
# =============================================================================
# Here, we assume the discrepancy can be written as a function of the instantaneous
# measured state [x2, v2]. We build a candidate library using polynomials of these two variables.
# Construct a feature matrix using x2 and v2:
X_features = np.hstack([x2_data, v2_est.reshape(-1, 1)])

# Build a polynomial library (up to degree 3)
library = ps.PolynomialLibrary(degree=3)
Theta = library.fit_transform(X_features)

# Use Lasso (sparse regression) to learn coefficients xi in:
#   discrepancy ≈ Θ(x2, v2) · xi
lasso = Lasso(alpha=0.001, fit_intercept=False, max_iter=10000)
lasso.fit(Theta, discrepancy)
xi = lasso.coef_

# Report the learned nonzero terms.
feature_names = library.get_feature_names()
print("Learned discrepancy model (nonzero coefficients):")
for coef, name in zip(xi, feature_names):
    if np.abs(coef) > 1e-5:
        print(f"  {name:20s} : {coef: .5f}")

# =============================================================================
# 5. Simulate the hybrid (physics-informed) model without delay embedding
# =============================================================================
# The hybrid model combines the known physics with the learned discrepancy:
#
#     x2'' = - c2 * v2 + g(x2, v2)
#
# We will simulate the hybrid model in discrete time using Euler integration.
# We define the state as y = [x2, v2]. The update equations are:
#
#     v2[n+1] = v2[n] + dt * (-c2 * v2[n] + g(x2[n], v2[n]) )
#     x2[n+1] = x2[n] + dt * v2[n]
#
# Here, g(x2, v2) is computed by evaluating the candidate library on the state
# and applying the learned coefficients.

n_steps = len(t)
hybrid_x2 = np.zeros(n_steps)
hybrid_v2 = np.zeros(n_steps)

# Initialize with the first measured values
hybrid_x2[0] = x2_data[0]
hybrid_v2[0] = v2_est[0]

# Euler integration of the hybrid model
for n in range(n_steps - 1):
    # Current state: [x2, v2]
    current_state = np.array([[hybrid_x2[n], hybrid_v2[n]]])
    Theta_current = library.transform(current_state)
    # Evaluate the learned discrepancy model: g(x2, v2)
    g_pred = np.dot(Theta_current, xi)[0]
    # g_pred = discrepancy[n]
    
    # Compute the hybrid acceleration:
    a_hybrid = - c2 * hybrid_v2[n] + g_pred
    
    # Euler updates:
    hybrid_v2[n+1] = hybrid_v2[n] + dt * a_hybrid
    hybrid_x2[n+1] = hybrid_x2[n] + dt * hybrid_v2[n]

# =============================================================================
# 6. Plot and compare the results
# =============================================================================


# Optionally, compare the learned discrepancy with the true discrepancy:
Theta_all = library.transform(X_features)
d_pred = Theta_all.dot(xi)

plt.figure(figsize=(10, 4))
plt.plot(t, discrepancy, 'k', label='True discrepancy')
plt.plot(t, d_pred, 'r--', label='Learned discrepancy')
plt.xlabel('Time')
plt.ylabel("Discrepancy (x2'')")
plt.title('Discrepancy: True vs. Learned  no delays')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, x2, 'k', label='Full simulation (x2 true)')
plt.plot(t, hybrid_x2, 'r--', label='Hybrid simulation (x2 hybrid)')
plt.xlabel('Time')
plt.ylabel('x2 (position)')
plt.title('Full Simulation vs. Hybrid (Physics-Informed) Model no delays')
plt.legend()
plt.tight_layout()
plt.show()
