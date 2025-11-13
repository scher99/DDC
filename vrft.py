import numpy as np
import scipy.signal as signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# --- 1. NONLINEAR System Simulation with SMOOTHED Friction ---
def nonlinear_system_dynamics(state, t, u, t_vec, m1, m2, k1, k2, c1, c2, mu1, mu2, g):
    """Defines the system of differential equations with a smoothed friction model."""
    x1, v1, x2, v2 = state
    alpha = 100
    f_friction1 = mu1 * m1 * g * np.tanh(alpha * v1)
    f_friction2 = mu2 * m2 * g * np.tanh(alpha * v2)
    current_u = np.interp(t, t_vec, u)
    a1 = (current_u - k1*x1 - c1*v1 - k2*(x1 - x2) - c2*(v1 - v2) - f_friction1) / m1
    a2 = (k2*(x1 - x2) + c2*(v1 - v2) - f_friction2) / m2
    return [v1, a1, v2, a2]

def simulate_nonlinear_system(u, t, noise_variance=0.0):
    """Simulates the NONLINEAR system response using an ODE solver."""
    m1, m2, k1, k2, c1, c2, g, mu1, mu2 = 1.0, 1.5, 1.0, 0.5, 0.1, 0.2, 9.81, 0.15, 0.1
    initial_state = [0.0, 0.0, 0.0, 0.0]
    state_trajectory = integrate.odeint(
        nonlinear_system_dynamics, initial_state, t,
        args=(u, t, m1, m2, k1, k2, c1, c2, mu1, mu2, g), rtol=1e-6, atol=1e-6
    )
    y = state_trajectory[:, 2]
    y_noisy = y + np.sqrt(noise_variance) * np.random.randn(len(t))
    return y_noisy

# --- Simulation Setup ---
fs = 100
T = 60
t = np.arange(0, T, 1/fs)
np.random.seed(0) # Use a fixed random seed for reproducible results
prbs = np.random.choice([-1.0, 1.0], size=len(t)) * 20.0 
b_filter, a_filter = signal.butter(4, 10, fs=fs)
u_exp = signal.lfilter(b_filter, a_filter, prbs)

y_exp = simulate_nonlinear_system(u_exp, t)
print("Experimental data has been generated from the NONLINEAR plant.")

# --- 2. Controller Structures (Unchanged) ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, fs, filter_tau):
        num_s2 = Kp * filter_tau + Kd
        num_s1 = Kp + Ki * filter_tau
        num_s0 = Ki
        num_coeffs = [num_s2, num_s1, num_s0]
        den_coeffs = [filter_tau, 1, 0]
        self.tf_s = signal.TransferFunction(num_coeffs, den_coeffs)
        self.tf_z = self.tf_s.to_discrete(1/fs, method='bilinear')
    def get_tf(self):
        return self.tf_z

# --- 3. Virtual Reference Feedback Tuning (VRFT) - CORRECTED & ROBUST IMPLEMENTATION ---
def vrft_tune(u, y, M, controller_class, fs, filter_tau):
    """
    Tunes a controller using a robust formulation of VRFT.
    """
    # Define the prefilter L=M for stability
    L_num, L_den = M.num, M.den

    # The virtual error e_v is never explicitly calculated.
    # Instead, we create the signals needed for the regression directly.
    # We want to solve L*u = C * L*(inv(M)-1)*y
    # Let's define the filtered regressor source: y_tilde = L*(inv(M)-1)*y
    
    # Step 1: Calculate (inv(M)-1)*y
    inv_M_num, inv_M_den = M.den, M.num
    # Numerator of (inv(M)-1) is num(inv_M) - den(inv_M) = den(M) - num(M)
    num_invM_minus_1 = np.polyadd(inv_M_den, -inv_M_num) 
    den_invM_minus_1 = inv_M_den
    
    # This is (inv(M)-1)*y
    y_processed = signal.lfilter(num_invM_minus_1, den_invM_minus_1, y)
    
    # Step 2: Filter with L=M to get the final regressor source
    # y_tilde = L*y_processed = M*(inv(M)-1)*y = (1-M)*y
    y_tilde = signal.lfilter(L_num.flatten(), L_den.flatten(), y_processed)

    # The target vector is the filtered input signal
    target_vector_u = signal.lfilter(L_num.flatten(), L_den.flatten(), u)

    # Step 3: Build the regressor matrix Phi from y_tilde
    if controller_class == PIDController:
        phi_p = y_tilde # Proportional regressor
        
        integ_tf_s = signal.TransferFunction([1], [1, 0])
        integ_tf_z = integ_tf_s.to_discrete(1/fs, 'bilinear')
        phi_i = signal.lfilter(integ_tf_z.num[0], integ_tf_z.den, y_tilde) # Integral
        
        deriv_tf_s = signal.TransferFunction([1, 0], [filter_tau, 1])
        deriv_tf_z = deriv_tf_s.to_discrete(1/fs, 'bilinear')
        phi_d = signal.lfilter(deriv_tf_z.num[0], deriv_tf_z.den, y_tilde) # Derivative
        
        # The equation is C*y_tilde = u_f => [phi_p, phi_i, phi_d]*rho = u_f
        # This is incorrect. The controller acts on the error. Let's re-think.

        # Let's use the simplest, most direct formulation:
        # u = C * e_v. Filter both sides: L*u = C*(L*e_v)
        # This IS what I had before. What went wrong?
        # Let's rewrite it cleanly one last time.

        # 1. Generate Virtual Reference and Error
        inv_M_num, inv_M_den = M.den, M.num
        r_v = signal.lfilter(inv_M_num, inv_M_den, y)
        e_v = r_v - y
        
        # 2. Choose the prefilter L=M for stability
        L_num, L_den = M.num.flatten(), M.den.flatten()

        # 3. Filter the target signal `u` and the source signal `e_v`
        u_f = signal.lfilter(L_num, L_den, u)
        e_v_f = signal.lfilter(L_num, L_den, e_v)
        
        # 4. Build the regressor matrix Phi from the filtered virtual error `e_v_f`
        phi_p = e_v_f # Proportional regressor
        
        integ_tf_s = signal.TransferFunction([1], [1, 0])
        integ_tf_z = integ_tf_s.to_discrete(1/fs, 'bilinear')
        phi_i = signal.lfilter(integ_tf_z.num[0], integ_tf_z.den, e_v_f) # Integral
        
        deriv_tf_s = signal.TransferFunction([1, 0], [filter_tau, 1])
        deriv_tf_z = deriv_tf_s.to_discrete(1/fs, 'bilinear')
        phi_d = signal.lfilter(deriv_tf_z.num[0], deriv_tf_z.den, e_v_f) # Derivative
        
        Phi = np.vstack([phi_p, phi_i, phi_d]).T
    else:
        raise NotImplementedError("This script only supports PID.")

    # 5. Solve the stable least-squares problem: Phi * rho = u_f
    rho_hat, _, _, _ = np.linalg.lstsq(Phi, u_f, rcond=None)
    
    noise_var_hat = np.var(u_f - Phi @ rho_hat)
    try:
        param_cov = noise_var_hat * np.linalg.inv(Phi.T @ Phi)
        param_std_dev = np.sqrt(np.diag(param_cov))
    except np.linalg.LinAlgError:
        param_cov, param_std_dev = None, np.array([np.nan]*len(rho_hat))
    return rho_hat, param_cov, param_std_dev


# --- VRFT Execution ---
omega_n = 3.0
zeta = 0.9
M_s = signal.TransferFunction([omega_n**2], [1, 2*zeta*omega_n, omega_n**2])
M_z = M_s.to_discrete(1/fs, method='bilinear')
N_filter = 10
tau = 1 / (N_filter * omega_n)

rho_pid, cov_pid, std_dev_pid = vrft_tune(u_exp, y_exp, M_z, PIDController, fs, filter_tau=tau)

print("\n--- VRFT Results for PID Controller ---")
print(f"Estimated Parameters (Kp, Ki, Kd): {rho_pid}")
print(f"Standard Deviation of Parameters:    {std_dev_pid}")

# --- 4. Closed-Loop Performance Analysis (Unchanged) ---
def plot_closed_loop_performance(controller_tf_z, reference_model_z, fs, title):
    t_step = np.arange(0, 15, 1/fs)
    r_step = np.ones_like(t_step)
    t_out_ref, y_ref_series = signal.dlsim(reference_model_z, r_step, t=t_step)
    y_cl = np.zeros_like(t_step)
    u_cl = np.zeros_like(t_step)
    b, a = controller_tf_z.num.flatten(), controller_tf_z.den.flatten()
    zi = signal.lfiltic(b, a, y=[0], x=[0])
    plant_state = np.array([0.0, 0.0, 0.0, 0.0])

    for i in range(len(t_step)):
        error = r_step[i] - (y_cl[i-1] if i > 0 else 0)
        u_out_array, zi = signal.lfilter(b, a, [error], zi=zi)
        u_cl[i] = u_out_array[0]
        t_span = [t_step[i-1] if i > 0 else 0, t_step[i]]
        
        def temp_dynamics(state, t):
            m1, m2, k1, k2, c1, c2, g, mu1, mu2 = 1.0, 1.5, 1.0, 0.5, 0.1, 0.2, 9.81, 0.15, 0.1
            alpha = 100
            x1, v1, x2, v2 = state
            f_friction1 = mu1 * m1 * g * np.tanh(alpha * v1)
            f_friction2 = mu2 * m2 * g * np.tanh(alpha * v2)
            a1 = (u_cl[i] - k1*x1 - c1*v1 - k2*(x1 - x2) - c2*(v1 - v2) - f_friction1) / m1
            a2 = (k2*(x1 - x2) + c2*(v1 - v2) - f_friction2) / m2
            return [v1, a1, v2, a2]

        result_state = integrate.odeint(temp_dynamics, plant_state, t_span, rtol=1e-6, atol=1e-6)
        plant_state = result_state[-1]
        y_cl[i] = plant_state[2]

    plt.figure(figsize=(12, 7))
    plt.plot(t_step, y_cl, label='Closed-Loop with VRFT-Tuned Controller', linewidth=2.5)
    plt.plot(t_out_ref, y_ref_series, label='Reference Model (Desired Response)', linestyle='--', color='red', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel('Time (s)')
    plt.ylabel('Position of Second Mass')
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

tuned_pid = PIDController(rho_pid[0], rho_pid[1], rho_pid[2], fs, filter_tau=tau)
plot_closed_loop_performance(tuned_pid.get_tf(), M_z, fs, "VRFT: Final Performance on Nonlinear Plant")