import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize


# --- 1. Define the "Hidden" Plant: Two-Mass Spring-Damper System ---

# Global system parameters
m1 = 1.0  # kg
m2 = 18.0  # kg
k1 = 0.0  # N/m
k2 = (32.0*2*np.pi)**2  # N/m
c1 = 75.0  # Ns/m
c2 = 300.0  # Ns/m
mu = 0.0  # Friction coefficient
Fc = 0.0  # Coulomb friction force

# plant = two_mass_spring_damper. this is hidden, we use it only to generate data
def plant(state, t, F):
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
    # Coulomb friction
    f_coulomb1 = -Fc * np.sign(x1_dot) if x1_dot != 0 else 0.0
    f_coulomb2 = -Fc * np.sign(x2_dot) if x2_dot != 0 else 0.0

    # Equations of motion
    # x1_ddot = (F - k1*x1 - c1*x1_dot - k2*(x1 - x2) - c2*(x1_dot - x2_dot) + f_friction1 + f_coulomb1) / m1
    # x2_ddot = (k2*(x1 - x2) + c2*(x1_dot - x2_dot) + f_friction2 + f_coulomb2) / m2
    x1_ddot = (F - c1*x1_dot - k2*(x1 - x2) + f_friction1 + f_coulomb1) / m1
    x2_ddot = (k2*(x1 - x2) - c2*(x2_dot) + f_friction2 + f_coulomb2) / m2
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

class ComplexController:
    """
    A controller cascading a PID, Lead, Lag, and Notch filter.
    C_total = C_notch * C_lag * C_lead * C_pid
    """
    def __init__(self, params, dt):
        (self.Kp, self.Ki, self.Kd, self.Tf, 
         self.z_lead, self.p_lead, 
         self.z_lag, self.p_lag, 
         self.w0_notch, self.Q_notch) = params
        self.dt = dt

        # --- Internal states for each filter stage ---
        # PID states (unchanged)
        self.pid_integral = 0; self.pid_prev_error = 0; self.pid_filt_deriv = 0
        self.pid_alpha = self.dt / (self.Tf + self.dt if self.Tf > 0 else 1e-9)
        
        # Lead filter states and coefficients (unchanged)
        den_k_lead = (self.p_lead * self.dt + 2); self.lead_b0 = (self.z_lead * self.dt + 2) / den_k_lead
        self.lead_b1 = (self.z_lead * self.dt - 2) / den_k_lead; self.lead_a1 = (self.p_lead * self.dt - 2) / den_k_lead
        self.lead_prev_in = 0; self.lead_prev_out = 0

        # Lag filter states and coefficients (unchanged)
        den_k_lag = (self.p_lag * self.dt + 2); self.lag_b0 = (self.z_lag * self.dt + 2) / den_k_lag
        self.lag_b1 = (self.z_lag * self.dt - 2) / den_k_lag; self.lag_a1 = (self.p_lag * self.dt - 2) / den_k_lag
        self.lag_prev_in = 0; self.lag_prev_out = 0

        # --- CORRECTED Notch filter states and coefficients (Bilinear Transform) ---
        w0, Q = self.w0_notch, self.Q_notch
        if w0 < 1e-3 or Q < 1e-3: # Handle case of inactive notch
            self.notch_b0, self.notch_b1, self.notch_b2 = 1.0, 0.0, 0.0
            self.notch_a1, self.notch_a2 = 0.0, 0.0
        else:
            # Intermediate terms for clarity, based on the standard Audio EQ Cookbook formulas
            wc = w0 * self.dt
            alpha = np.sin(wc) / (2.0 * Q)
            cos_wc = np.cos(wc)
            
            a0_inv = 1.0 / (1.0 + alpha) # Pre-calculate normalization factor
            
            # Normalized coefficients for H(z) = (b0+b1z^-1+b2z^-2)/(1+a1z^-1+a2z^-2)
            self.notch_b0 = a0_inv
            self.notch_b1 = -2.0 * cos_wc * a0_inv
            self.notch_b2 = a0_inv
            self.notch_a1 = -2.0 * cos_wc * a0_inv
            self.notch_a2 = (1.0 - alpha) * a0_inv

        self.notch_prev_in1, self.notch_prev_in2 = 0, 0
        self.notch_prev_out1, self.notch_prev_out2 = 0, 0

    def calculate(self, error):
        # Stage 1: PID (unchanged)
        self.pid_integral += error * self.dt
        unfiltered_deriv = (error - self.pid_prev_error)
        self.pid_filt_deriv = self.pid_alpha * (self.pid_filt_deriv + unfiltered_deriv)
        pid_out = (self.Kp * error + self.Ki * self.pid_integral + (self.Kd / self.dt) * self.pid_filt_deriv)
        self.pid_prev_error = error
        
        # Stage 2: Lead Filter (unchanged)
        lead_out = -self.lead_a1 * self.lead_prev_out + self.lead_b0 * pid_out + self.lead_b1 * self.lead_prev_in
        self.lead_prev_in = pid_out; self.lead_prev_out = lead_out
        
        # Stage 3: Lag Filter (unchanged)
        lag_out = -self.lag_a1 * self.lag_prev_out + self.lag_b0 * lead_out + self.lag_b1 * self.lag_prev_in
        self.lag_prev_in = lead_out; self.lag_prev_out = lag_out
        
        # --- CORRECTED Stage 4: Notch Filter Difference Equation ---
        notch_out = (self.notch_b0 * lag_out + self.notch_b1 * self.notch_prev_in1 + self.notch_b2 * self.notch_prev_in2 -
                     self.notch_a1 * self.notch_prev_out1 - self.notch_a2 * self.notch_prev_out2)
        
        # Update notch states
        self.notch_prev_in2 = self.notch_prev_in1
        self.notch_prev_in1 = lag_out
        self.notch_prev_out2 = self.notch_prev_out1
        self.notch_prev_out1 = notch_out

        return notch_out

def desired_response(t, oneshot_r, zeta=0.7, omega_n=2.0):
    """
    Generates the desired response of a second-order system.
    zeta: Damping ratio
    omega_n: Natural frequency (rad/s)
    """
    # Second-order system transfer function: G(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    num = [omega_n**2]
    den = [1, 2*zeta*omega_n, omega_n**2]
    reference_model = signal.TransferFunction(num, den)
    
    # Generate step response
    _, y_d, _ = signal.lsim((reference_model.num, reference_model.den), U=oneshot_r, T=t)
    return y_d


# --- 5. Plot the Results ---
def plot_results(t, oneshot_y, oneshot_u, oneshot_r, y_d, zeta=0.7, omega_n=2.0):
    """
    Plots the results of the simulation.
    """
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
    # print(f"Shape of reference data (r): {oneshot_r.shape}")
    # print(f"Shape of measured output data (y): {oneshot_y.shape}")
    # print(f"Shape of control input data (u): {oneshot_u.shape}")
    # print(f"Shape of desired output data (y_d): {y_d.shape}")

# --- 6. Calculate the Fictitious Reference ---
# def calculate_fictitious_reference_pid(oneshot_u, y_d, dt, params):
#     """
#     Calculates the fictitious reference signal based on the control input and desired output.
#     This is used for the FRIT optimization.
#     """
#     # # First, define the discrete transfer function for the initial PID controller C0(z)
#     # # C0(z) = [b0 + b1*z^-1 + b2*z^-2] / [a0 + a1*z^-1 + a2*z^-2]
#     # # For a standard PID, this is:
#     # # C0(z) = Kp + Ki*dt/(1-z^-1) + (Kd/dt)*(1-z^-1)
#     # # After combining terms over a common denominator:
#     # # Denominator of C0(z) is (1 - z^-1)
#     # a_C0 = [1, -1]
#     # # Numerator of C0(z) is (Kp+Ki*dt+Kd/dt) + (-Kp-2*Kd/dt)*z^-1 + (Kd/dt)*z^-2
#     # b_C0 = [Kp_init + Ki_init*dt + Kd_init/dt,
#     #         -Kp_init - 2*Kd_init/dt,
#     #         Kd_init/dt]

#     # # We need the inverse, 1/C0(z), which has its numerator and denominator swapped.
#     # # So we will filter `oneshot_u` with a system where b = a_C0 and a = b_C0.
#     # b_inv = a_C0
#     # a_inv = b_C0

#     # The transfer function of the filtered PID is more complex. Using Tustin's method for discretization:
#     # C(z) = Kp + Ki*dt/2 * (1+z^-1)/(1-z^-1) + Kd/Tf * (1-z^-1)/((1+dt/2/Tf) - (1-dt/2/Tf)z^-1)
#     # This results in a 2nd order numerator and denominator (biproper).
#     # After much algebra, the coefficients are:
#     Kp_init, Ki_init, Kd_init, Tf_init = params

#     k_d_f = Kd_init / (Tf_init + dt)
#     k_i_f = Ki_init * dt

#     # Numerator b0, b1, b2
#     b0 = Kp_init + k_i_f + k_d_f
#     b1 = k_i_f - Kp_init - 2 * k_d_f
#     b2 = k_d_f
#     b_C0 = [b0, b1, b2]

#     # Denominator a0, a1, a2
#     a0 = 1
#     a1 = k_i_f * Tf_init / (Tf_init + dt) - 1
#     a2 = -k_i_f * Tf_init / (Tf_init + dt)
#     a_C0 = [a0, a1, a2]

#     # The inverse controller 1/C0(z) simply swaps the coefficients.
#     # Because C0(z) is now biproper, its inverse is also biproper and computable.
#     b_inv = a_C0
#     a_inv = b_C0

#     # Calculate the fictitious error `e_f` by filtering `u` with the inverse controller dynamics
#     e_f = signal.lfilter(b_inv, a_inv, oneshot_u)

#     # Calculate the fictitious reference `r_f`
#     r_f = e_f + y_d

#     return r_f, e_f

def calculate_fictitious_reference_pid(oneshot_u, y_d, dt, params):
    """
    Calculates the fictitious reference by NUMERICALLY inverting the controller.
    This is more robust than deriving transfer function coefficients.
    """
    Kp, Ki, Kd, Tf = params

    # We need to find the error signal `e_f` that would have produced `oneshot_u`.
    # u[k] = Kp*e[k] + Ki*integral[k] + D_term[k]
    # This is a system of equations that is hard to invert directly.
    # A simpler and correct approach is to derive the true transfer function for
    # the implemented controller.

    # Let's derive the correct transfer function C(z) = U(z)/E(z) for the implementation.
    # P(z)/E(z) = Kp
    # I(z)/E(z) = Ki*dt / (1 - z^-1)
    # D(z)/E(z) = (Kd/dt) * alpha * (1 - z^-1) / (1 - (1-alpha)z^-1)
    # where alpha = dt / (Tf + dt)

    alpha = dt / (Tf + dt)

    # Combine all terms over a common denominator: (1 - z^-1) * (1 - (1-alpha)z^-1)
    # Denominator polynomial: 1 - (2-alpha)z^-1 + (1-alpha)z^-2
    a_C0 = [1, -(2 - alpha), (1 - alpha)]

    # Numerator is more complex:
    # Kp_term = Kp * (1 - (2-alpha)z^-1 + (1-alpha)z^-2)
    # Ki_term = Ki*dt * (1 - (1-alpha)z^-1)
    # Kd_term = (Kd/dt)*alpha * (1 - z^-1)*(1-z^-1) = (Kd/dt)*alpha * (1 - 2z^-1 + z^-2)
    # Sum the coefficients for z^0, z^-1, z^-2
    b0 = Kp + Ki*dt + (Kd/dt)*alpha
    b1 = -Kp*(2-alpha) - Ki*dt*(1-alpha) - 2*(Kd/dt)*alpha
    b2 = Kp*(1-alpha) + (Kd/dt)*alpha
    b_C0 = [b0, b1, b2]
    
    # The inverse controller 1/C0(z) swaps the coefficients.
    b_inv = a_C0
    a_inv = b_C0

    # This check is crucial. If the inverse is unstable, something is wrong.
    # The poles of the inverse are the zeros of the forward controller.
    if np.any(np.abs(np.roots(a_inv)) > 1.0):
        print("WARNING: Inverse controller is unstable. FRIT will likely fail.")
        # This can happen if the optimizer picks pathological PID values.
        # We return a very high cost to steer it away.
        # In this function, we can't return a cost, but we can return a zero signal.
        return np.zeros_like(y_d), np.zeros_like(y_d)


    # Calculate the fictitious error `e_f` by filtering `u` with the inverse controller
    e_f = signal.lfilter(b_inv, a_inv, oneshot_u)

    # Calculate the fictitious reference `r_f`
    r_f = e_f + y_d

    return r_f, e_f

def calculate_fictitious_reference_complex(u_signal, y_signal, dt, params):
    """
    Calculates the fictitious reference by sequentially inverting each
    component of the complex controller.
    """
    (Kp, Ki, Kd, Tf, z_lead, p_lead, z_lag, p_lag, w0_notch, Q_notch) = params

    # --- Helper to get discrete Lead/Lag coefficients (unchanged) ---
    def get_leadlag_coeffs(z, p, dt):
        den_k = (p * dt + 2); b0 = (z * dt + 2) / den_k
        b1 = (z * dt - 2) / den_k; a1 = (p * dt - 2) / den_k
        return [b0, b1], [1, a1]

    # --- CORRECTED Helper to get discrete Notch coefficients ---
    def get_notch_coeffs(w0, Q, dt):
        if w0 < 1e-3 or Q < 1e-3:
            return [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        wc = w0 * dt; alpha = np.sin(wc) / (2.0 * Q); cos_wc = np.cos(wc)
        a0_inv = 1.0 / (1.0 + alpha)
        
        b0 = a0_inv; b1 = -2.0 * cos_wc * a0_inv; b2 = a0_inv
        a1 = -2.0 * cos_wc * a0_inv; a2 = (1.0 - alpha) * a0_inv
        
        return [b0, b1, b2], [1.0, a1, a2] # Denominator has leading 1
        
    # --- Get coefficients for each forward filter component ---
    # PID Part (unchanged)
    alpha_pid = dt / (Tf + dt if Tf > 0 else 1e-9); a_pid = [1, -(2 - alpha_pid), (1 - alpha_pid)]
    b0_pid = Kp + Ki*dt + (Kd/dt)*alpha_pid; b1_pid = -Kp*(2-alpha_pid) - Ki*dt*(1-alpha_pid) - 2*(Kd/dt)*alpha_pid
    b2_pid = Kp*(1-alpha_pid) + (Kd/dt)*alpha_pid; b_pid = [b0_pid, b1_pid, b2_pid]
    
    # Lead/Lag Parts (unchanged)
    b_lead, a_lead = get_leadlag_coeffs(z_lead, p_lead, dt)
    b_lag, a_lag = get_leadlag_coeffs(z_lag, p_lag, dt)

    # Notch Part (uses the new corrected helper)
    b_notch, a_notch = get_notch_coeffs(w0_notch, Q_notch, dt)

    # --- Apply inverse filters sequentially (unchanged) ---
    if (np.any(np.abs(np.roots(b_pid)) > 1.0) or np.any(np.abs(np.roots(b_lead)) > 1.0) or
        np.any(np.abs(np.roots(b_lag)) > 1.0) or np.any(np.abs(np.roots(b_notch)) > 1.0)):
        return np.zeros_like(y_signal), np.zeros_like(y_signal)

    s1 = signal.lfilter(a_notch, b_notch, u_signal)
    s2 = signal.lfilter(a_lag, b_lag, s1)
    s3 = signal.lfilter(a_lead, b_lead, s2)
    e_f = signal.lfilter(a_pid, b_pid, s3)
    
    r_f = e_f + y_signal
    return r_f, e_f


# def calculate_fictitious_reference_ll(oneshot_u, y_d, dt, params):
#     """
#     Calculates the fictitious reference signal based on the control input and desired output.
#     This is used for the FRIT optimization.
#     """
#     # Get the Tustin-discretized forward controller coefficients
#     # (These are the same as inside the class, calculated again for clarity)
#     K_init, z_init, p_init = params
#     den_k_init = (p_init * dt + 2)
#     b0_C0 = K_init * (z_init * dt + 2) / den_k_init
#     b1_C0 = K_init * (z_init * dt - 2) / den_k_init
#     a1_C0 = (p_init * dt - 2) / den_k_init

#     # B(z) numerator coefficients of FORWARD controller C0(z)
#     b_forward = [b0_C0, b1_C0]
#     # A(z) denominator coefficients of FORWARD controller C0(z)
#     a_forward = [1.0, a1_C0]

#     # For the INVERSE controller 1/C0(z), we just swap them:
#     b_inverse = a_forward
#     a_inverse = b_forward

#     # print(f"Forward C(z) Numerator (B): {np.round(b_forward, 3)}")
#     # print(f"Forward C(z) Denominator (A): {np.round(a_forward, 3)}")
#     # print(f"Inverse C(z) Numerator (A): {np.round(b_inverse, 3)}")
#     # print(f"Inverse C(z) Denominator (B): {np.round(a_inverse, 3)}")

#     # Calculate the fictitious error `e_f` by filtering `u` with the inverse controller
#     e_f = signal.lfilter(b_inverse, a_inverse, oneshot_u)

#     # Calculate the fictitious reference `r_f`
#     r_f = e_f + y_d
#     return r_f, e_f


# # Helper function to get a continuous-time TF for the PID
# def get_pid_filter_tf(pid_params, dt):
#     """Calculates the continuous-time transfer function for the PID with filter."""
#     Kp, Ki, Kd, Tf = pid_params
#     num_C = [(Kp * Tf + Kd), (Kp + Ki * Tf), Ki]
#     den_C = [Tf, 1, 0]
#     return signal.TransferFunction(num_C, den_C)

def cost_function(params, y_0, u_0, t, dt, M_tf):
    """
    Calculates the cost based on the true "Output Error" formulation from FRIT papers.
    J = || y_0 - M * r_fict ||^2
    where r_fict depends on the new controller parameters.
    """
    # Basic parameter constraints
    # if any(p < 0.000 for p in params): return 1e10
    if params[-1] > 1.0: return 1e10

    try:
        # Step 1: Calculate the inverse of the CANDIDATE controller C_new (rho)
        r_fict, _ = calculate_fictitious_reference_pid(u_0, y_0, dt, params)

        # Step 3: Simulate this r_fict through the desired model M(s) (Td)
        _, y_sim, _ = signal.lsim(M_tf, U=r_fict, T=t)

        # Step 4: The cost is the error between the real output and this simulated ideal output
        cost = np.mean((y_0 - y_sim)**2) + 10*np.max((y_0 - y_sim)**2) #+ np.std((y_0[0:1000] - y_sim[0:1000])**2)

        # reg = 1e-6 * np.sum(params**2)

        # cost += reg  # Regularization term to prevent overfitting

    except Exception as e:
        return 1e10 # Return high cost if any numerical instability occurs

    return cost

def cost_function_complex(params, y_0, u_0, t, dt, M_tf):
    """
    Cost function for the complex cascaded controller, using the
    robust "Output Error" formulation from the literature.
    """
    # Unpack all 10 parameters
    (Kp, Ki, Kd, Tf, z_lead, p_lead, z_lag, p_lag, w0_notch, Q_notch) = params

    # Basic parameter constraints
    if any(p < 0.0 for p in [Kp, Ki, Kd, Tf, z_lead, p_lead, z_lag, p_lag, w0_notch, Q_notch]):
        return 1e10
    # Physical constraints: lead zero < lead pole, lag zero > lag pole
    # if z_lead >= p_lead or z_lag <= p_lag:
    #     return 1e10

    try:
        # Step 1: Calculate the new fictitious reference r_fict for the candidate controller
        r_fict, _ = calculate_fictitious_reference_complex(u_0, y_0, dt, params)
        
        # If the inverse was unstable, r_fict will be all zeros, return high cost
        if np.all(r_fict == 0):
             return 1e10

        # Step 2: Simulate this r_fict through the desired model M(s)
        _, y_sim, _ = signal.lsim(M_tf, U=r_fict, T=t)

        # Step 3: The cost is the error between the real output and this simulated ideal output
        cost = np.mean((y_0 - y_sim)**2) + 100*np.max((y_0 - y_sim)**4)

    except Exception as e:
        return 1e10 # Return high cost if any numerical instability occurs

    return cost

# def cost_function_pid(params, r_f, y_meas, u_meas, dt):
#     """
#     Calculates the cost for a given set of PID parameters.
#     The cost is the sum of squared errors between the measured control signal
#     and the one calculated with the new parameters.
#     """
#     Kp, Ki, Kd, Tf = params
    
#     # Add constraints to keep parameters reasonable
#     if any(p < 0 for p in params):
#         return 1e10 # All params must be positive
#     if Tf > 1.0: # Filter time constant shouldn't be excessively large
#         return 1e10

#     # Simulate the new PID controller's output u_new
#     temp_controller = PIDControllerWithFilter(Kp, Ki, Kd, Tf, dt)
#     u_new = np.zeros_like(u_meas)
#     error_new = r_f - y_meas
#     for i in range(len(error_new)):
#         u_new[i] = temp_controller.calculate(error_new[i])
        
#     # Return the cost
#     return np.mean((u_meas - u_new)**2)

# def cost_function_leadlag(params, r_f, y_meas, u_meas, dt):
#     K, z, p = params
#     # Ensure pole and zero are positive (for stability and lead/lag behavior)
#     if K < 0 or z < 0 or p < 0: return 1e10

#     # Simulate the new Lead-Lag controller's output u_new
#     temp_controller = LeadLagController(K, z, p, dt)
#     u_new = np.zeros_like(u_meas)
#     error_new = r_f - y_meas
#     for i in range(len(error_new)):
#         u_new[i] = temp_controller.calculate(error_new[i])
        
#     return np.mean((u_meas - u_new)**2)

# print("--- FRIT Optimization Results ---")
# print(f"Initial PID Gains: Kp={Kp_init:.2f}, Ki={Ki_init:.2f}, Kd={Kd_init:.2f}")

# --- 9. Plot Final Results ---
def plot_final_results(t, r, oneshot_y, history_x2_opt, y_d, original_y=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))
    plt.plot(t, r, 'k--',label='input', alpha=0.5)
    plt.plot(t, oneshot_y, label='Initial Response (Untuned)', alpha=0.5)
    plt.plot(t, history_x2_opt, 'b', label='FRIT-Tuned Response', linewidth=2.5)
    plt.plot(t, y_d, 'r--', label='Desired Reference Model', linewidth=2.5)
    plt.title('FRIT Performance: Initial vs. Optimized Controller', fontsize=16)
    if original_y is not None:
        plt.plot(t, original_y, 'g--', label='Original Response', alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Position of Mass 2 (m)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def run_simulation(controller, r, t, noise_std_dev, dt):
    # state[2] = x2, state[3]=x2_dot
    state = np.zeros(4)
    history_x2 = np.zeros(len(t))
    history_u = np.zeros(len(t))
    measured_x2_history = np.zeros(len(t))
    for i in range(len(t)-1):
        measured_x2 = state[3] + np.random.normal(0, noise_std_dev)
        measured_x2_history[i] = measured_x2
        history_x2[i] = state[3] # Rate of change of x2
        error = r[i] - measured_x2
        u = controller.calculate(error)
        history_u[i] = u
        state_dot = plant(state, t[i], u)
        state = state + np.array(state_dot) * dt
    history_x2[-1] = state[3]
    measured_x2_history[-1] = state[3] + np.random.normal(0, noise_std_dev)
    history_u[-1] = history_u[-2]
    return measured_x2_history, history_u, r

def optimize_controller(fun, t, y_meas, u_meas, dt, initial_guess, bounds=None, M_tf=None):
    """ Modified to accept and use bounds. """
    # method='Nelder-Mead', # Note: Nelder-Mead does not support bounds. Switching to a method that does.
    # L-BFGS-B is a good choice for box-constrained optimization.
    if bounds is None:
        method='Nelder-Mead'
    else:
        method='L-BFGS-B'
    result = minimize(fun, initial_guess, 
                      args=(y_meas, u_meas, t, dt, M_tf), 
                      method=method,
                      bounds=bounds)
    return result.x, result.fun

def main():
    # --- SETUP ---
    dt = 0.001; T = 5; n_steps = int(T / dt); t = np.linspace(0, T, n_steps + 1)
    r = np.ones(n_steps + 1) * 1.0; r[0:int(1/dt)] = 0.0; r[int(4/dt):int(8/dt)] = 0.0
    # zeta, omega_n = 0.7, 10.0*2*np.pi; 
    noise_std_dev = 0.003

    # --- Draw Bode Plot of Linearized Plant ---
    # Linearize the plant: neglect friction and Coulomb nonlinearities
    # State-space: [x1, x1_dot, x2, x2_dot]
    # Input: F (force on mass 1)
    # Output: x2 (position of mass 2)
    # Build A, B, C, D matrices
    A = np.array([
        [0, 1, 0, 0],
        [-k2/m1, -c1/m1, k2/m1, 0],
        [0, 0, 0, 1],
        [k2/m2, 0, -k2/m2, -c2/m2]
    ])
    B = np.array([[0], [1/m1], [0], [0]])
    C = np.array([[0, 0, 0, 1]])  # Output x2
    D = np.array([[0]])

    # Create state-space system and convert to transfer function
    sys_ss = signal.StateSpace(A, B, C, D)
    sys_tf = signal.TransferFunction(*signal.ss2tf(A, B, C, D))

    # Plot Bode plot
    w, mag, phase = signal.bode(sys_tf)
    # plt.figure(figsize=(10,6))
    # plt.subplot(2,1,1)
    # plt.semilogx(w, mag)
    # plt.title("Bode Plot of Linearized Plant (x2/F)")
    # plt.ylabel("Magnitude (dB)")
    # plt.grid(True, which='both')
    # plt.subplot(2,1,2)
    # plt.semilogx(w, phase)
    # plt.xlabel("Frequency (rad/s)")
    # plt.ylabel("Phase (deg)")
    # plt.grid(True, which='both')
    # plt.tight_layout()
    # plt.show()

    # --- PI Controller Design via Pole Placement ---
    # Desired closed-loop bandwidth (omega_n) and damping ratio (zeta)
    omega_n = 10 * np.pi * 2  # rad/s
    zeta = 0.8

    # The plant transfer function: sys_tf (from above)
    # For a PI controller: C(s) = Kp + Ki/s
    # Closed-loop characteristic equation: Den(s) + Num(s)*C(s) = 0

    # Get plant numerator and denominator
    num_p, den_p = sys_tf.num, sys_tf.den

    # For a SISO system, assume plant: G(s) = b/(s^2 + a1*s + a0)
    # PI controller: C(s) = Kp + Ki/s
    # Closed-loop TF: G(s)*C(s)/(1 + G(s)*C(s))
    # Characteristic equation: den_p(s)*s + num_p(s)*(Kp*s + Ki) = 0

    # For our plant, num_p = [k], den_p = [1, a1, a0, 0] (third order due to double integrator)
    # But for two-mass system, let's use only the dominant mode for pole placement:
    # For simplicity, use the low-frequency approximation: G(s) ≈ k/(m2*s^2 + c2*s + k2)
    # So, G(s) ≈ k2/(m2*s^2 + c2*s + k2)

    # PI controller: C(s) = Kp + Ki/s
    # Open-loop: G(s)*C(s) = [Kp*k2 + Ki*k2/s]/(m2*s^2 + c2*s + k2)
    # Closed-loop characteristic equation:
    # m2*s^2 + c2*s + k2 + Kp*k2*s + Ki*k2 = 0
    # Group terms:
    # m2*s^2 + (c2 + Kp*k2)*s + (k2 + Ki*k2) = 0

    # Set desired characteristic equation: s^2 + 2*zeta*omega_n*s + omega_n^2 = 0
    # Match coefficients:
    # m2*s^2 + (c2 + Kp*k2)*s + (k2 + Ki*k2) = m2*(s^2 + 2*zeta*omega_n*s + omega_n^2)
    # So:
    # s^2: m2 = m2
    # s^1: c2 + Kp*k2 = m2*2*zeta*omega_n  => Kp = (m2*2*zeta*omega_n - c2)/k2
    # s^0: k2 + Ki*k2 = m2*omega_n^2       => Ki = (m2*omega_n**2 - k2)/k2

    Kp_pi = (m2 * 2 * zeta * omega_n - c2) / k2 /dt
    Ki_pi = (m2 * omega_n**2 - k2) / k2 / dt

    print(f"Pole Placement PI Controller: Kp = {Kp_pi:.4f}, Ki = {Ki_pi:.4f}")

    # You can now use these gains in PIDController or PIDControllerWithFilter (with Kd=0, Tf=0.05)
    # pi_controller = PIDControllerWithFilter(Kp_pi, Ki_pi, 0.0, 0.05, dt)
    # y_pi, u_pi, r_pi = run_simulation(pi_controller, r, t, noise_std_dev, dt)
    # y_d = desired_response(t, r, zeta=zeta, omega_n=omega_n)
    # plot_results(t, y_pi, u_pi, r_pi, y_d, zeta=zeta, omega_n=omega_n)

    # --- INITIAL EXPERIMENT (with the starting PID Controller) ---
    # This is C_0
    # Kp_init, Ki_init, Kd_init, Tf_init = 1.0, 0.5, 1.0, 0.05
    # Kp_init, Ki_init, Kd_init, Tf_init = Kp_pi, Ki_pi, 0.0, 0.05
    # initial_controller = PIDControllerWithFilter(Kp_init, Ki_init, Kd_init, Tf_init, dt)
    # --- Define the full initial parameter set for the Complex Controller ---
    # We will start with a flat Lead, Lag, and a non-active Notch
    initial_params = [
        Kp_pi, Ki_pi, 0.0, 0.05,  # Kp, Ki, Kd, Tf
        10.0, 10.1,               # z_lead, p_lead (very close = almost flat)
        10.1, 10.0,               # z_lag, p_lag (very close = almost flat)
        200.0, 10.0              # w0_notch, Q_notch (High Q, may not be near resonance)
    ]
    initial_controller = ComplexController(initial_params, dt)
    
    # This is (y_0, u_0)
    y_0, u_0, r_0 = run_simulation(initial_controller, r, t, noise_std_dev, dt)
    y_d = desired_response(t, r, zeta=zeta, omega_n=omega_n)
    # Define the desired closed-loop model M(s) (or Td)
    M_tf = signal.TransferFunction([omega_n**2], [1, 2*zeta*omega_n, omega_n**2])
    
    print("Showing initial controller performance...")
    # plot_results(t, y_0, u_0, r_0, y_d, zeta=zeta, omega_n=omega_n)
    
    # --- ITERATIVE TUNING LOOP ---
    N_iterations = 1 # Let's do 5 iterations to see clear convergence
    
    # Initialize the parameters and simulation data for the loop
    # current_params = [Kp_init, Ki_init, Kd_init, Tf_init]
    current_params = initial_params
    sim_y = y_0
    sim_u = u_0

    for i in range(N_iterations):
        print(f"\n--- Starting FRIT Iteration {i+1} ---")
        
        # Optimize for the next controller (C_{i+1}) using the data from
        # the previous iteration (y_i, u_i) and the new r_f.
        # We start the search from the last known best parameters.
        # print(f"Optimizing for new controller parameters...")
        # We state that each parameter must be at least its initial value.
        # We set the upper bound to None (or a very large number) for infinity.
        # param_bounds = [
        #     (current_params[0], None),  # Kp must be >= Kp_init
        #     (current_params[1], None),  # Ki must be >= Ki_init
        #     (current_params[2], None),  # Kd must be >= Kd_init
        #     (0.001, None)   # Tf must be >= Tf_init (or you could give it a small lower bound like 0.001)
        # ]
        # param_bounds = [
        #     (0.001, None),  # Kp must be >= Kp_init
        #     (0.001, None),  # Ki must be >= Ki_init
        #     (0.001, None),  # Kd must be >= Kd_init
        #     (0.001, 0.05)   # Tf must be >= Tf_init (or you could give it a small lower bound like 0.001)
        # ]
        param_bounds = None
        # print(f"Using Bounds: {param_bounds}")
        # new_params, cost = optimize_controller(
        #     cost_function_pid, r_f_iter, sim_y, sim_u, dt, current_params
        # )
        # new_params, cost = optimize_controller(
        #     cost_function, t, sim_y, sim_u, dt, current_params, bounds=param_bounds, M_tf=M_tf)
        new_params, cost = optimize_controller(
        cost_function_complex, t, sim_y, sim_u, dt, current_params, bounds=param_bounds, M_tf=M_tf)


        # print(f"Result (Iter {i+1}): Kp={new_params[0]:.2f}, Ki={new_params[1]:.2f}, Kd={new_params[2]:.2f}, Tf={new_params[3]:.4f}, cost={cost:.6f}")
        print(f"Result (Iter {i+1}): Kp={new_params[0]:.2f}, Ki={new_params[1]:.2f}, Kd={new_params[2]:.2f}, Tf={new_params[3]:.4f}, ")
        print(f"                     z_lead={new_params[4]:.2f}, p_lead={new_params[5]:.2f}, z_lag={new_params[6]:.2f}, p_lag={new_params[7]:.4f},")
        print(f"                     w_notch={new_params[8]:.2f}, Q_notch={new_params[9]:.2f}, cost={cost:.6f}")

        # Update the current parameters to the new ones we just found
        current_params = new_params
        
        # STEP 3: Run a NEW simulation with the NEWLY optimized controller (C_{i+1})
        # to generate the dataset (y_{i+1}, u_{i+1}) for the NEXT loop.
        print("Validating new controller and generating data for next iteration...")
        # current_controller = PIDControllerWithFilter(
        #     current_params[0], current_params[1], 
        #     current_params[2], current_params[3], dt
        # )
        current_controller = ComplexController(current_params, dt)

        sim_y, sim_u, _ = run_simulation(current_controller, r, t, noise_std_dev, dt)
        
        # Plot the progress, comparing to the very first response
        plot_final_results(t, r, y_0, sim_y, y_d)
        
    print("\n--- Iterative FRIT Process Complete ---")



if __name__ == "__main__":
    main()