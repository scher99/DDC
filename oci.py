import numpy as np
import scipy.signal as signal
from scipy.linalg import lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Step 1: Define the Hidden Plant ---
def create_two_mass_spring_damper(m1=1.0, m2=1.5, k1=1.0, k2=0.5, b1=0.2, b2=0.1, Ts=0.1):
    A = [[0, 1, 0, 0],
         [-k2/m1, -b1/m1, k2/m1, 0],
         [0, 0, 0, 1],
         [k2/m2, 0, -k2/m2, -b2/m2]]
    B = [[0], [1/m1], [0], [0]]
    C = [[0, 0, 0, 1]] # measure x2dot
    D = [[0]]
    continuous_sys = signal.StateSpace(A, B, C, D)
    discrete_sys_tf = continuous_sys.to_tf().to_discrete(dt=Ts)
    print("--- Hidden Plant Created ---")
    print(f"Sampling Time (Ts): {Ts}s")
    print(f"Plant Numerator: {np.round(discrete_sys_tf.num.flatten(), 4).tolist()}")
    print(f"Plant Denominator: {np.round(discrete_sys_tf.den.flatten(), 4).tolist()}\n")
    return discrete_sys_tf

# --- Step 2: Define the Desired Performance ---
def create_reference_model(damping=0.9, natural_freq=2.0, Ts=0.1):
    num_c = [natural_freq**2]
    den_c = [1, 2 * damping * natural_freq, natural_freq**2]
    continuous_model = signal.TransferFunction(num_c, den_c)
    discrete_model_tf = continuous_model.to_discrete(dt=Ts)
    print("--- Reference Model Created ---")
    print(f"Desired Damping: {damping}, Desired Natural Frequency: {natural_freq} rad/s")
    print(f"Model Numerator: {np.round(discrete_model_tf.num.flatten(), 4).tolist()}")
    print(f"Model Denominator: {np.round(discrete_model_tf.den.flatten(), 4).tolist()}\n")
    return discrete_model_tf

# --- Step 3: Select a Controller Structure ---
def pid_controller_tf(params, Ts, N=20):
    Kp, Ki, Kd = params
    b0 = Kp + Ki*Ts + Kd*N
    b1 = -(Kp + Kp*Ts*N - Ki*Ts + 2*Kd*N)
    b2 = Kp*Ts*N + Kd*N
    a0 = 1
    a1 = -(1 + Ts*N)
    a2 = Ts*N
    num_pid = [b0, b1, b2]
    den_pid = [a0, a1, a2]
    return signal.TransferFunction(num_pid, den_pid, dt=Ts)

# or, function to build the complex controller 
def create_complex_controller_tf(params, Ts):
    """
    Builds the complete controller transfer function from physical parameters.
    params = [Kp, Ki, Kd, N, T_lead, alpha_lead, T_lag, alpha_lag, w0_notch, Q_notch]
    """
    Kp, Ki, Kd, N, T_lead, alpha_lead, T_lag, alpha_lag, w0_notch, Q_notch = params
    
    # 1. PID Part (Proper)
    b0_pid = Kp + Ki*Ts + Kd*N
    b1_pid = -(Kp + Kp*Ts*N - Ki*Ts + 2*Kd*N)
    b2_pid = Kp*Ts*N + Kd*N
    a0_pid = 1; a1_pid = -(1 + Ts*N); a2_pid = Ts*N
    num_pid = [b0_pid, b1_pid, b2_pid]
    den_pid = [a0_pid, a1_pid, a2_pid]

    # 2. Lead Compensator Part C(s) = (1+s*T)/(1+s*alpha*T)
    # Using Tustin (bilinear) transform s = 2/Ts * (z-1)/(z+1)
    num_lead = [(2*T_lead + Ts), (Ts - 2*T_lead)]
    den_lead = [(2*alpha_lead*T_lead + Ts), (Ts - 2*alpha_lead*T_lead)]
    
    # 3. Lag Compensator Part C(s) = (1+s*T)/(1+s*alpha*T) (with alpha > 1)
    num_lag = [(2*T_lag + Ts), (Ts - 2*T_lag)]
    den_lag = [(2*alpha_lag*T_lag + Ts), (Ts - 2*alpha_lag*T_lag)]

    # 4. Notch Filter Part
    # Using Tustin transform on the continuous-time notch filter
    Omega = 2/Ts * np.tan(w0_notch * Ts / 2)
    alpha_notch = 4 + (Omega*Ts)**2
    b0_notch = (4 + (Omega*Ts)**2) / alpha_notch
    b1_notch = (2*(Omega*Ts)**2 - 8) / alpha_notch
    b2_notch = (4 + (Omega*Ts)**2) / alpha_notch
    a0_notch = 1.0
    a1_notch = (2*(Omega*Ts)**2 - 8) / alpha_notch
    a2_notch = (4 - 2*Omega*Ts/Q_notch + (Omega*Ts)**2) / (4 + 2*Omega*Ts/Q_notch + (Omega*Ts)**2) # Correction needed here.
    
    # Let's use a standard library for robustness
    # This avoids algebraic errors.
    notch_tf_cont = signal.TransferFunction([1, 0, w0_notch**2], [1, w0_notch/Q_notch, w0_notch**2])
    notch_tf_disc = notch_tf_cont.to_discrete(dt=Ts, method='bilinear')
    num_notch = notch_tf_disc.num.flatten()
    den_notch = notch_tf_disc.den.flatten()

    # Combine all parts by multiplying polynomials
    num_total = np.polymul(np.polymul(np.polymul(num_pid, num_lead), num_lag), num_notch)
    den_total = np.polymul(np.polymul(np.polymul(den_pid, den_lead), den_lag), den_notch)
    
    return num_total, den_total


# # --- Step 4: Collect Input-Output Data ---
# def generate_simulation_data(plant, sim_time=50, Ts=0.1):
#     N_points = int(sim_time / Ts)
#     t = np.linspace(0, sim_time, N_points)
#     # prbs = np.random.choice([-1, 1], size=N_points)
#     # b, a = signal.butter(4, 0.1)
#     # u = signal.lfilter(b, a, prbs)
#     # u = u / np.max(np.abs(u))
#     # Generate a step input for simplicity
#     u = np.ones(N_points)  # Step input
#     u[0:int(1/Ts)] = 0
#     _, y = signal.dlsim(plant, u, t)
#     print("--- Generated Input-Output Data for Identification ---\n")
#     return t, u, y.flatten()

def generate_simulation_data(plant, sim_time=50, Ts=0.1, signal_type='chirp'):
    """
    Generates input-output data from the plant.
    signal_type can be 'prbs' or 'chirp'.
    """
    N_points = int(sim_time / Ts)
    t = np.linspace(0, sim_time, N_points)
    
    if signal_type == 'prbs':
        prbs = np.random.choice([-1, 1], size=N_points)
        b, a = signal.butter(4, 0.1)
        u = signal.lfilter(b, a, prbs)
    elif signal_type == 'chirp':
        # A chirp signal sweeps from a low to a high frequency
        f0 = 0.01  # Start frequency
        f1 = 1 / (2 * Ts) * 0.5 # End frequency (half of Nyquist)
        u = signal.chirp(t, f0=f0, f1=f1, t1=sim_time, method='logarithmic')
    else:
        raise ValueError("signal_type must be 'prbs' or 'chirp'")
        
    u = u / np.max(np.abs(u)) # Normalize
    _, y = signal.dlsim(plant, u, t)
    # Add measurement noise
    noise = np.random.normal(0, 0.001, y.shape)  # 1% noise
    y = y + 0.0*noise
    print(f"--- Generated Input-Output Data using '{signal_type}' signal ---\n")
    return t, u, y.flatten()


# --- Step 5: OCI Identification with CORRECTED Parameter Conversion ---
def oci_identify_controller(u, y, M, Ts, N=20):
    """
    Identifies the optimal controller parameters for the proper PID structure
    using the corrected algebraic inversion to find Kp, Ki, Kd.
    """
    M_num = M.num.flatten()
    M_den = M.den.flatten()
    
    order_diff = len(M_den) - len(M_num)
    padded_M_num = np.pad(M_num, (order_diff, 0), 'constant')
    one_minus_M_num = np.polysub(M_den, padded_M_num)
    
    y_f = signal.lfilter(one_minus_M_num, M_den, y)
    u_f = signal.lfilter(M_num, M_den, u)

    a0 = 1
    a1 = -(1 + Ts*N)
    a2 = Ts*N
    
    target = a0*u_f[2:] + a1*u_f[1:-1] + a2*u_f[0:-2]
    
    regressor = np.vstack([y_f[2:], y_f[1:-1], y_f[0:-2]]).T
    
    b_opt, _, _, _ = lstsq(regressor[:-1], target)
    b0, b1, b2 = b_opt
    
    # This is the key fix to the negative coefficient problem.
    Ki = (b0 + b1 + b2) / (2 * Ts)
    Kp = (b2 - (b0 - Ki*Ts)) / (Ts*N - 1)
    Kd = (b0 - Kp - Ki*Ts) / N
    # --- End of Fix ---
    
    optimal_params = [Kp, Ki, Kd]
    
    print("--- Optimal Controller Identification Complete ---")
    print(f"Identified Controller Numerator Params (b0, b1, b2): {[round(p, 4) for p in b_opt]}")
    print(f"Identified PID Parameters (Kp, Ki, Kd): {[round(p, 4) for p in optimal_params]}\n")
    
    # Sanity check for the user
    if any(p < 0 for p in optimal_params):
        print("!!! WARNING: One or more identified PID coefficients are negative.")
        print("!!! This may indicate poor data quality or issues with the reference model.\n")
        
    return optimal_params

def oci_identify_linear_coeffs(u, y, M, Ts, order):
    """
    Identifies the numerator and denominator coefficients of a controller of a given order.
    """
    # Pre-filter the data (this part is always the same)
    M_num = M.num.flatten(); M_den = M.den.flatten()
    order_diff = len(M_den) - len(M_num)
    padded_M_num = np.pad(M_num, (order_diff, 0), 'constant')
    one_minus_M_num = np.polysub(M_den, padded_M_num)
    y_f = signal.lfilter(one_minus_M_num, M_den, y)
    u_f = signal.lfilter(M_num, M_den, u)

    # Build the regressor matrix (Phi) and target vector (Y)
    # The number of columns is 2 * order.
    # We need 'order' number of delayed u's and 'order'+1 of delayed y's.
    # To keep arrays aligned, we start from an index equal to the order.
    
    # Target Vector Y is u_f[k], since a_0 = 1
    target = u_f[order:]

    # Regressor Matrix Φ
    regressor_cols = []
    # Add delayed u_f columns for a_1, a_2, ...
    for i in range(1, order + 1):
        regressor_cols.append(-u_f[order-i:-i])
    # Add delayed y_f columns for b_0, b_1, ...
    for i in range(order + 1):
        regressor_cols.append(y_f[order-i:len(y_f)-i-1])
        
    regressor = np.vstack(regressor_cols).T
    
    # Solve the linear system θ = (ΦᵀΦ)⁻¹ΦᵀY
    coeffs, _, _, _ = lstsq(regressor, target)
    
    # Separate the found coefficients into A and B polynomials
    a_coeffs_no_a0 = coeffs[:order]
    b_coeffs = coeffs[order:]
    
    # Add the a_0 = 1 coefficient back to the front
    a_coeffs = np.concatenate(([1.0], a_coeffs_no_a0))

    print("--- Linear Controller Identification Complete ---")
    print(f"Identified Denominator (A) coeffs: {[round(c, 4) for c in a_coeffs]}")
    print(f"Identified Numerator (B) coeffs: {[round(c, 4) for c in b_coeffs]}\n")
    
    return b_coeffs, a_coeffs

def oci_identify_complex_controller(u, y, M, Ts, initial_guess, bounds):
    """
    Identifies the physical parameters of the complex controller using optimization.
    """
    # Pre-filter the data just like before
    M_num = M.num.flatten(); M_den = M.den.flatten()
    order_diff = len(M_den) - len(M_num)
    padded_M_num = np.pad(M_num, (order_diff, 0), 'constant')
    one_minus_M_num = np.polysub(M_den, padded_M_num)
    y_f = signal.lfilter(one_minus_M_num, M_den, y)
    u_f = signal.lfilter(M_num, M_den, u)

    # This is the function we want to minimize
    def cost_function(params):
        # 1. Build the controller from the current guess of parameters
        A, B = create_complex_controller_tf(params, Ts)
        
        # 2. Calculate the prediction error based on the Golden Rule: e = A*u_f - B*y_f
        # We need to filter u_f with A and y_f with B
        error_signal = signal.lfilter(A, 1, u_f) - signal.lfilter(B, 1, y_f[:-1])
        
        # 3. The cost is the sum of the squared errors
        return np.sum(error_signal**2)

    print("--- Starting Non-Linear Optimization for Complex Controller ---")
    print("This may take a few moments...")
    
    # Use scipy.optimize.minimize
    result = minimize(
        cost_function,
        initial_guess,
        method='L-BFGS-B', # A good method that can handle bounds
        bounds=bounds,
        options={'disp': True}
    )
    
    optimal_params = result.x
    print("\n--- Optimization Complete ---")
    param_names = ['Kp', 'Ki', 'Kd', 'N', 'T_lead', 'alpha_lead', 'T_lag', 'alpha_lag', 'w0_notch', 'Q_notch']
    for name, val in zip(param_names, optimal_params):
        print(f"Identified {name}: {val:.4f}")
    
    return optimal_params



# --- Step 6: Validate the Controller and Plot Results ---
def validate_and_plot(plant, controller_params, M, Ts, N=20):
    controller = pid_controller_tf(controller_params, Ts, N)
    num_cl = np.polymul(plant.num.flatten(), controller.num.flatten())
    den_cl = np.polyadd(np.polymul(plant.den.flatten(), controller.den.flatten()), num_cl)
    oci_closed_loop = signal.TransferFunction(num_cl, den_cl, dt=Ts)
    
    sim_time = 5
    N_sim = int(sim_time / Ts)
    t = np.linspace(0, sim_time, N_sim)
    ref_signal = np.ones(N_sim)
    ref_signal[0:int(1/Ts)] = 0  # Step input for reference
    
    _, y_ol = signal.dlsim(plant, ref_signal, t)
    _, y_ref = signal.dlsim(M, ref_signal, t)
    _, y_cl = signal.dlsim(oci_closed_loop, ref_signal, t)
    
    error = ref_signal - y_cl[:-1].flatten()
    _, u = signal.dlsim(controller, error, t)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    axs[0].plot(t, ref_signal, 'k--', label='Reference (Setpoint)')
    axs[0].plot(t, y_ol[:-1], 'r-', alpha=0.7, label='Uncontrolled Plant (Open-Loop)')
    axs[0].plot(t, y_ref[:-1], 'g:', lw=3, label='Desired Response (Reference Model)')
    axs[0].plot(t, y_cl[:-1], 'b-', lw=2, label='OCI Controlled Plant')
    axs[0].set_title('Step Response Comparison: OCI Controller Performance', fontsize=14)
    axs[0].set_ylabel('Position of Mass 1', fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True)
    
    axs[1].plot(t, u[:-1], 'm-', label='Control Signal u(t)')
    axs[1].set_title('Control Signal Generated by OCI Controller', fontsize=14)
    axs[1].set_xlabel('Time (seconds)', fontsize=12)
    axs[1].set_ylabel('Force Applied', fontsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def validate_and_plot_linear(plant, controller_coeffs, M, Ts):
    num_controller, den_controller = controller_coeffs
    controller = signal.TransferFunction(num_controller, den_controller, dt=Ts)
    
    # ... rest of the plotting function is the same ...
    num_cl = np.polymul(plant.num.flatten(), controller.num.flatten())
    den_cl = np.polyadd(np.polymul(plant.den.flatten(), controller.den.flatten()), num_cl)
    oci_closed_loop = signal.TransferFunction(num_cl, den_cl, dt=Ts)
    sim_time = 25
    N_sim = int(sim_time / Ts)
    
    t = np.linspace(0, sim_time, N_sim)
    ref_signal = np.ones(N_sim)
    _, y_ol = signal.dlsim(plant, ref_signal, t)
    _, y_ref = signal.dlsim(M, ref_signal, t)
    _, y_cl = signal.dlsim(oci_closed_loop, ref_signal, t)
    error = ref_signal - y_cl.flatten()[:-1]
    _, u = signal.dlsim(controller, error, t)

    plt.style.use('seaborn-v0_8-whitegrid'); fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t, ref_signal, 'k--', label='Reference (Setpoint)')
    axs[0].plot(t, y_ol[:-1], 'r-', alpha=0.7, label='Uncontrolled Plant (Open-Loop)')
    axs[0].plot(t, y_ref[:-1], 'g:', lw=3, label='Desired Response (Reference Model)')
    axs[0].plot(t, y_cl[:-1], 'b-', lw=2, label='OCI Controlled Plant')
    axs[0].set_title('Step Response Comparison: OCI Controller Performance', fontsize=14)
    axs[0].set_ylabel('Position of Mass 1', fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True)
    axs[1].plot(t, u[:-1], 'm-', label='Control Signal u(t)')
    axs[1].set_title('Control Signal Generated by OCI Controller', fontsize=14)
    axs[1].set_xlabel('Time (seconds)', fontsize=12)
    axs[1].set_ylabel('Force Applied', fontsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

def validate_and_plot_complex(plant, controller_params, M, Ts):
    num_controller, den_controller = create_complex_controller_tf(controller_params, Ts)
    controller = signal.TransferFunction(num_controller, den_controller, dt=Ts)
    
    num_cl = np.polymul(plant.num.flatten(), controller.num.flatten())
    den_cl = np.polyadd(np.polymul(plant.den.flatten(), controller.den.flatten()), num_cl)
    oci_closed_loop = signal.TransferFunction(num_cl, den_cl, dt=Ts)
    
    # ... rest of the plotting function is the same ...
    sim_time = 25; N_sim = int(sim_time / Ts); t = np.linspace(0, sim_time, N_sim)
    ref_signal = np.ones(N_sim); _, y_ol = signal.dlsim(plant, ref_signal, t)
    _, y_ref = signal.dlsim(M, ref_signal, t); _, y_cl = signal.dlsim(oci_closed_loop, ref_signal, t)
    error = ref_signal - y_cl[:-1].flatten(); _, u = signal.dlsim(controller, error, t)
    plt.style.use('seaborn-v0_8-whitegrid'); fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t, ref_signal, 'k--', label='Reference (Setpoint)'); axs[0].plot(t, y_ol[:-1], 'r-', alpha=0.7, label='Uncontrolled Plant (Open-Loop)')
    axs[0].plot(t, y_ref[:-1], 'g:', lw=3, label='Desired Response (Reference Model)'); axs[0].plot(t, y_cl[:-1], 'b-', lw=2, label='OCI Controlled Plant')
    axs[0].set_title('Step Response Comparison: OCI Controller Performance', fontsize=14); axs[0].set_ylabel('Position of Mass 1', fontsize=12)
    axs[0].legend(fontsize=10); axs[0].grid(True); axs[1].plot(t, u[:-1], 'm-', label='Control Signal u(t)')
    axs[1].set_title('Control Signal Generated by OCI Controller', fontsize=14); axs[1].set_xlabel('Time (seconds)', fontsize=12)
    axs[1].set_ylabel('Force Applied', fontsize=12); axs[1].legend(fontsize=10); axs[1].grid(True); plt.tight_layout(); plt.show()



# --- Main Simulation Orchestrator ---
if __name__ == '__main__':
    SAMPLING_TIME = 0.001
    FILTER_COEFF = 3
    SIM_TIME_IDENT=40

    m1 = 1.0  # kg
    m2 = 18.0  # kg
    k1 = 0.0  # N/m
    k2 = (32.0*2*np.pi)**2  # N/m
    b1 = 75.0  # Ns/m
    b2 = 300.0  # Ns/m
    mu = 0.0  # Friction coefficient
    Fc = 0.0  # Coulomb friction force
    
    plant_tf = create_two_mass_spring_damper(
        m1=m1, m2=m2, k1=k1, k2=k2, b1=b1, b2=b2, Ts=SAMPLING_TIME
    )
    # this is open loop
    time_vec, input_data, output_data = generate_simulation_data(
        plant=plant_tf, sim_time=SIM_TIME_IDENT, Ts=SAMPLING_TIME, signal_type='chirp'
    )

    # at this point we have input and output data, we can create the reference model
    reference_model_tf = create_reference_model(
        damping=0.9, natural_freq=10, Ts=SAMPLING_TIME
    )
    
    optimal_pid_params = oci_identify_controller(
        u=input_data,
        y=output_data,
        M=reference_model_tf,
        Ts=SAMPLING_TIME,
        N=FILTER_COEFF
    )
    validate_and_plot(
        plant=plant_tf,
        controller_params=optimal_pid_params,
        M=reference_model_tf,
        Ts=SAMPLING_TIME,
        N=FILTER_COEFF
    )

    # here we try to identify a controller which is more complex than PID, but all we guess is the coefficients
    # Determine the total order of our desired controller
    # Proper PID (2nd order) + Lead (1st) + Lag (1st) + Notch (2nd)
    # Total Order = 2 + 1 + 1 + 2 = 6
    # CONTROLLER_ORDER = 6
    
    # optimal_coeffs = oci_identify_linear_coeffs(
    #     u=input_data, 
    #     y=output_data, 
    #     M=reference_model_tf, 
    #     Ts=SAMPLING_TIME, 
    #     order=CONTROLLER_ORDER
    # )
    
    # validate_and_plot_linear(
    #     plant=plant_tf, 
    #     controller_coeffs=optimal_coeffs, 
    #     M=reference_model_tf, 
    #     Ts=SAMPLING_TIME
    # )

    # Now we can try to identify a more complex controller with physical parameters
    # PID gains + Lead poles + Lag poles + Notch Q and w0
    # The optimizer NEEDS a reasonable starting point.
    # [Kp, Ki, Kd, N, T_lead, alpha_lead, T_lag, alpha_lag, w0_notch, Q_notch]
    initial_guess = [
        5.0, 1.0, 0.1,  # PID
        10,             # N filter
        0.1, 0.1,       # Lead (alpha < 1)
        1.0, 10,        # Lag (alpha > 1)
        3.0, 1.0        # Notch (w0 in rad/s, Q)
    ]
    
    # Bounds help the optimizer search in a sensible space.
    bounds = [
        (0, None), (0, None), (0, None), # Kp, Ki, Kd > 0
        (2, 20),                         # N
        (0, None), (0.01, 1.0),          # T_lead > 0, alpha_lead < 1
        (0, None), (1.0, 100),           # T_lag > 0, alpha_lag > 1
        (0.1, 10), (0.1, 10)             # w0 > 0, Q > 0
    ]
    
    # optimal_physical_params = oci_identify_complex_controller(
    #     input_data, output_data, reference_model_tf, SAMPLING_TIME, initial_guess, bounds
    # )
    
    # validate_and_plot_complex(
    #     plant=plant_tf,
    #     controller_params=optimal_physical_params,
    #     M=reference_model_tf,
    #     Ts=SAMPLING_TIME
    # )