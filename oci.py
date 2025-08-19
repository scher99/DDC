import numpy as np
import scipy.signal as signal
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# --- Step 1: Define the Hidden Plant ---
def create_two_mass_spring_damper(m1=1.0, m2=1.5, k1=1.0, k2=0.5, b1=0.2, b2=0.1, Ts=0.1):
    A = [[0, 1, 0, 0],
         [-(k1+k2)/m1, -(b1+b2)/m1, k2/m1, b2/m1],
         [0, 0, 0, 1],
         [k2/m2, b2/m2, -k2/m2, -b2/m2]]
    B = [[0], [1/m1], [0], [0]]
    C = [[1, 0, 0, 0]]
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
    
    # --- CORRECTED ALGEBRAIC INVERSION ---
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

# --- Step 6: Validate the Controller and Plot Results ---
def validate_and_plot(plant, controller_params, M, Ts, N=20):
    controller = pid_controller_tf(controller_params, Ts, N)
    num_cl = np.polymul(plant.num.flatten(), controller.num.flatten())
    den_cl = np.polyadd(np.polymul(plant.den.flatten(), controller.den.flatten()), num_cl)
    oci_closed_loop = signal.TransferFunction(num_cl, den_cl, dt=Ts)
    
    sim_time = 50
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

# --- Main Simulation Orchestrator ---
if __name__ == '__main__':
    SAMPLING_TIME = 0.01
    FILTER_COEFF = 10
    SIM_TIME_IDENT=200
    
    plant_tf = create_two_mass_spring_damper(
        m1=2.0, m2=1.0, k1=2.0, k2=1.0, b1=0.4, b2=0.2, Ts=SAMPLING_TIME
    )
    reference_model_tf = create_reference_model(
        damping=0.9, natural_freq=6.5, Ts=SAMPLING_TIME
    )
    # time_vec, input_data, output_data = generate_simulation_data(
    #     plant=plant_tf, sim_time=100, Ts=SAMPLING_TIME
    # )
    time_vec, input_data, output_data = generate_simulation_data(
        plant=plant_tf, sim_time=SIM_TIME_IDENT, Ts=SAMPLING_TIME, signal_type='chirp'
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