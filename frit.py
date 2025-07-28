import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize


# --- 1. Define the "Hidden" Plant: Two-Mass Spring-Damper System ---

# System parameters
m1 = 2.0  # kg
m2 = 1.0  # kg
k1 = 1.0  # N/m
k2 = 1.5  # N/m
c1 = 0.8  # Ns/m
c2 = 0.6  # Ns/m
mu = 0.1  # Friction coefficient

def two_mass_spring_damper(state, t, F):
    """
    Defines the differential equations for the two-mass spring-damper system.
    state = [x1, x1_dot, x2, x2_dot]
    t = time
    F = Force on mass 1
    """
    x1, x1_dot, x2, x2_dot = state

    # Friction forces (simple model: proportional to velocity)
    f_friction1 = -mu * x1_dot
    f_friction2 = -mu * x2_dot

    # Equations of motion
    x1_ddot = (F - k1*x1 - c1*x1_dot - k2*(x1 - x2) - c2*(x1_dot - x2_dot) + f_friction1) / m1
    x2_ddot = (k2*(x1 - x2) + c2*(x1_dot - x2_dot) + f_friction2) / m2

    return [x1_dot, x1_ddot, x2_dot, x2_ddot]

# --- 2. Define the Initial, Poorly-Tuned PID Controller ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.previous_error = 0

    def calculate(self, error):
        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        D = self.Kd * derivative
        self.previous_error = error

        return P + I + D

class PIDControllerWithFilter:
    def __init__(self, Kp, Ki, Kd, Tf, dt):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Tf = Tf  # Filter time constant for the derivative
        self.dt = dt
        
        # Pre-calculate filter coefficient
        self.alpha = self.dt / (self.Tf + self.dt)
        
        self.integral = 0
        self.previous_error = 0
        self.filtered_derivative = 0

    def calculate(self, error):
        # Proportional
        P = self.Kp * error
        
        # Integral
        self.integral += error * self.dt
        I = self.Ki * self.integral
        
        # Filtered Derivative
        # This implements D = (Kd/dt) * (1-z^-1) / (1 + (Tf/dt)*(1-z^-1))
        unfiltered_derivative = (error - self.previous_error)
        self.filtered_derivative = self.alpha * (self.filtered_derivative + unfiltered_derivative)
        D = (self.Kd / self.dt) * self.filtered_derivative

        self.previous_error = error
        return P + I + D

class LeadLagController:
    def __init__(self, K, z, p, dt):
        self.K, self.z, self.p, self.dt = K, z, p, dt
        # Discretize using Tustin's method: s = 2/dt * (1-z^-1)/(1+z^-1)
        # C(z) = K * (s+z)/(s+p)
        den_k = (self.p*self.dt + 2)
        self.b0 = self.K * (self.z*self.dt + 2) / den_k
        self.b1 = self.K * (self.z*self.dt - 2) / den_k
        self.a1 = (self.p*self.dt - 2) / den_k
        self.previous_error = 0
        self.previous_u = 0

    def calculate(self, error):
        # Difference equation: u[k] = -a1*u[k-1] + b0*e[k] + b1*e[k-1]
        u = -self.a1 * self.previous_u + self.b0 * error + self.b1 * self.previous_error
        self.previous_error = error
        self.previous_u = u
        return u

# --- 3. Run the "One-Shot" Simulation ---

# Simulation parameters
dt = 0.01  # Time step
T = 10    # Total simulation time
n_steps = int(T / dt)
t = np.linspace(0, T, n_steps + 1)

# Reference command
r = np.ones(n_steps + 1) * 1.0  # Step command to x2 = 1

# Initial PID gains (stable but poor performance)
Kp_init = 1.0
Ki_init = 0.5
Kd_init = 1.0
# initial_controller = PIDController(Kp_init, Ki_init, Kd_init, dt)
Tf_init = 0.05 # A small but non-zero filter time constant
initial_controller = PIDControllerWithFilter(Kp_init, Ki_init, Kd_init, Tf_init, dt)

# Measurement noise characteristics
noise_std_dev = 0.03

# Initialize state and history arrays
# state = [x1, x1_dot, x2, x2_dot]
state = np.array([0.0, 0.0, 0.0, 0.0])
history_x2 = np.zeros(n_steps + 1)
history_u = np.zeros(n_steps + 1)
measured_x2_history = np.zeros(n_steps + 1)

# Main simulation loop
for i in range(n_steps):
    # Get the "measured" output (with noise)
    measured_x2 = state[2] + np.random.normal(0, noise_std_dev)
    measured_x2_history[i] = measured_x2
    history_x2[i] = state[2]

    # Calculate controller output
    error = r[i] - measured_x2
    u = initial_controller.calculate(error)
    history_u[i] = u

    # Update plant state using simple Euler integration for one step
    # Note: For more accuracy, a more sophisticated solver like RK45
    # from scipy.integrate would be better, but Euler is sufficient here.
    state_dot = two_mass_spring_damper(state, t[i], u)
    state = state + np.array(state_dot) * dt

# Store the final state
history_x2[-1] = state[2]
measured_x2_history[-1] = state[2] + np.random.normal(0, noise_std_dev)
history_u[-1] = history_u[-2] # Hold last control value

# This is our collected "one-shot" data
oneshot_r = r
oneshot_y = measured_x2_history # The noisy measurement is what the controller sees
oneshot_u = history_u

# --- 4. Define and Generate the Desired Response ---

# Desired model parameters (as specified)
zeta = 0.7       # Damping ratio
omega_n = 2.0    # Natural frequency (rad/s)

# Create the second-order transfer function model: G(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
numerator = [omega_n**2]
denominator = [1, 2*zeta*omega_n, omega_n**2]
reference_model = signal.TransferFunction(numerator, denominator)

# Simulate the reference model's response to the step input `r`
# The output is the desired trajectory y_d(t)
_, y_d, _ = signal.lsim((reference_model.num, reference_model.den), U=oneshot_r, T=t)


# --- 5. Plot the Results ---

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Position of Mass 2
ax1.plot(t, oneshot_y, label='Measured Position of Mass 2 (x2)')
ax1.plot(t, y_d, 'r--', label=f'Desired Response (y_d)\nζ={zeta}, ωn={omega_n} rad/s', linewidth=2.5)
ax1.plot(t, oneshot_r, 'k--', label='Reference')
ax1.set_ylabel('Position (m)')
ax1.set_title('Initial Controller Performance')
ax1.legend()
ax1.grid(True)

# Plot 2: Control Input
ax2.plot(t, oneshot_u, label='Control Input (Force on Mass 1)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Force (N)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print the shapes of our collected data to confirm
print(f"Shape of reference data (r): {oneshot_r.shape}")
print(f"Shape of measured output data (y): {oneshot_y.shape}")
print(f"Shape of control input data (u): {oneshot_u.shape}")
print(f"Shape of desired output data (y_d): {y_d.shape}")

# --- 6. Calculate the Fictitious Reference ---

# # First, define the discrete transfer function for the initial PID controller C0(z)
# # C0(z) = [b0 + b1*z^-1 + b2*z^-2] / [a0 + a1*z^-1 + a2*z^-2]
# # For a standard PID, this is:
# # C0(z) = Kp + Ki*dt/(1-z^-1) + (Kd/dt)*(1-z^-1)
# # After combining terms over a common denominator:
# # Denominator of C0(z) is (1 - z^-1)
# a_C0 = [1, -1]
# # Numerator of C0(z) is (Kp+Ki*dt+Kd/dt) + (-Kp-2*Kd/dt)*z^-1 + (Kd/dt)*z^-2
# b_C0 = [Kp_init + Ki_init*dt + Kd_init/dt,
#         -Kp_init - 2*Kd_init/dt,
#         Kd_init/dt]

# # We need the inverse, 1/C0(z), which has its numerator and denominator swapped.
# # So we will filter `oneshot_u` with a system where b = a_C0 and a = b_C0.
# b_inv = a_C0
# a_inv = b_C0

# The transfer function of the filtered PID is more complex. Using Tustin's method for discretization:
# C(z) = Kp + Ki*dt/2 * (1+z^-1)/(1-z^-1) + Kd/Tf * (1-z^-1)/((1+dt/2/Tf) - (1-dt/2/Tf)z^-1)
# This results in a 2nd order numerator and denominator (biproper).
# After much algebra, the coefficients are:
k_d_f = Kd_init / (Tf_init + dt)
k_i_f = Ki_init * dt

# Numerator b0, b1, b2
b0 = Kp_init + k_i_f + k_d_f
b1 = k_i_f - Kp_init - 2 * k_d_f
b2 = k_d_f
b_C0 = [b0, b1, b2]

# Denominator a0, a1, a2
a0 = 1
a1 = k_i_f * Tf_init / (Tf_init + dt) - 1
a2 = -k_i_f * Tf_init / (Tf_init + dt)
a_C0 = [a0, a1, a2]

# The inverse controller 1/C0(z) simply swaps the coefficients.
# Because C0(z) is now biproper, its inverse is also biproper and computable.
b_inv = a_C0
a_inv = b_C0

# Calculate the fictitious error `e_f` by filtering `u` with the inverse controller dynamics
e_f = signal.lfilter(b_inv, a_inv, oneshot_u)

# Calculate the fictitious reference `r_f`
r_f = e_f + y_d

# --- 7. Plot the Results for Visualization ---

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 6))

plt.plot(t, oneshot_r, 'k:', label='Original Reference (r)')
plt.plot(t, y_d, '--', color='green', label='Desired Output (y_d)', linewidth=2)
plt.plot(t, r_f, label='Fictitious Reference (r_f)', color='purple', linewidth=2)

plt.title('Fictitious Reference Calculation')
plt.xlabel('Time (s)')
plt.ylabel('Signal Value')
plt.legend()
plt.grid(True)
plt.show()

# Print shapes to confirm
print(f"Shape of fictitious error data (e_f): {e_f.shape}")
print(f"Shape of fictitious reference data (r_f): {r_f.shape}")


def cost_function_pid(params, r_f, y_meas, u_meas, dt):
    """
    Calculates the cost for a given set of PID parameters.
    The cost is the sum of squared errors between the measured control signal
    and the one calculated with the new parameters.
    """
    Kp, Ki, Kd = params
    
    # Calculate the error for the new controller
    error_new = r_f - y_meas
    
    # Simulate the new controller's output u_new
    u_new = np.zeros_like(u_meas)
    integral, previous_error = 0, 0
    for i in range(len(error_new)):
        integral += error_new[i] * dt
        derivative = (error_new[i] - previous_error) / dt
        previous_error = error_new[i]
        u_new[i] = Kp * error_new[i] + Ki * integral + Kd * derivative
        
    # Return the cost (Mean Squared Error)
    return np.mean((u_meas - u_new)**2)

def cost_function_leadlag(params, r_f, y_meas, u_meas, dt):
    K, z, p = params
    # Ensure pole and zero are positive (for stability and lead/lag behavior)
    if K < 0 or z < 0 or p < 0: return 1e10

    # Simulate the new Lead-Lag controller's output u_new
    temp_controller = LeadLagController(K, z, p, dt)
    u_new = np.zeros_like(u_meas)
    error_new = r_f - y_meas
    for i in range(len(error_new)):
        u_new[i] = temp_controller.calculate(error_new[i])
        
    return np.mean((u_meas - u_new)**2)

print("--- FRIT Optimization Results ---")
print(f"Initial PID Gains: Kp={Kp_init:.2f}, Ki={Ki_init:.2f}, Kd={Kd_init:.2f}")

# Initial guess for the optimization can be the initial parameters
# Run the optimization
# initial_guess = [Kp_init, Ki_init, Kd_init]
# result = minimize(cost_function_pid, initial_guess, args=(r_f, oneshot_y, oneshot_u, dt),
#                   method='Nelder-Mead')
# Extract the optimized parameters
# Kp_opt, Ki_opt, Kd_opt = result.x
# print(f"Optimized PID Gains: Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f}")

initial_guess_ll = [5.0, 1.0, 10.0] # K, zero, pole
result_ll = minimize(cost_function_leadlag, initial_guess_ll, args=(r_f, oneshot_y, oneshot_u, dt),
                  method='Nelder-Mead')
K_opt, z_opt, p_opt = result_ll.x
print(f"Optimized Lead-Lag Gains: K={K_opt:.2f}, zero={z_opt:.2f}, pole={p_opt:.2f}")
print(f"Final Cost: {result_ll.fun:.4f}")

# --- 8. Validation with a New Simulation ---
final_ll_controller = LeadLagController(K_opt, z_opt, p_opt, dt)
state = np.zeros(4)
history_x2_opt = np.zeros(n_steps + 1)
integral, previous_error = 0, 0
for i in range(n_steps):
    measured_x2 = state[2] + np.random.normal(0, noise_std_dev)
    error = r[i] - measured_x2
    # PID
    # integral += error * dt
    # derivative = (error - previous_error) / dt
    # previous_error = error
    # u = Kp_opt * error + Ki_opt * integral + Kd_opt * derivative
    # LEAD-LAG
    u = final_ll_controller.calculate(error)
    state_dot = two_mass_spring_damper(state, t[i], u)
    state += np.array(state_dot) * dt
    history_x2_opt[i] = measured_x2
history_x2_opt[-1] = state[2] + np.random.normal(0, noise_std_dev)

# --- 9. Plot Final Results ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))
plt.plot(t, oneshot_y, label='Initial Response (Untuned)', alpha=0.5)
plt.plot(t, history_x2_opt, 'b', label='FRIT-Tuned Response', linewidth=2.5)
plt.plot(t, y_d, 'r--', label='Desired Reference Model', linewidth=2.5)
plt.title('FRIT Performance: Initial vs. Optimized Controller', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position of Mass 2 (m)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
