import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("QtAgg") # Uncomment if you run locally and want pop-up windows
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

def plot_raw_data(data):
    """
    Quick sanity plots for loaded text-file data.
    """
    t = np.array(data["time"])

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

    axs[0, 0].plot(t, data["AzLoadAngle"], label="psi (AzLoadAngle)")
    axs[0, 0].plot(t, data["ElLoadAngle"], label="theta (ElLoadAngle)")
    axs[0, 0].set_title("Load Angles")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_ylabel("Angle (rad)")

    axs[0, 1].plot(t, data["AzMotorRate"], label="psidot (AzMotorRate)")
    axs[0, 1].plot(t, data["ElMotorRate"], label="thetadot (ElMotorRate)")
    axs[0, 1].set_title("Motor Rates")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_ylabel("Rate (rad/s)")

    axs[1, 0].plot(t, data["LoadrRate"], label="wz (LoadrRate)")
    axs[1, 0].plot(t, data["LoadqRate"], label="wy (LoadqRate)")
    axs[1, 0].set_title("Load Rates")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].set_ylabel("Rate (rad/s)")

    axs[1, 1].plot(t, data["AzMotorCurrent"], label="t_mot_y")
    axs[1, 1].plot(t, data["ElMotorCurrent"], label="t_mot_p")
    axs[1, 1].set_title("Motor Torques")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_ylabel("Torque (Nm A)")

    axs[2, 0].plot(t, data["BaseRateX"], label="p")
    axs[2, 0].plot(t, data["BaseRateY"], label="q")
    axs[2, 0].plot(t, data["BaseRateZ"], label="r")
    axs[2, 0].set_title("Base Rates")
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    axs[2, 0].set_ylabel("Rate (rad/s)")
    axs[2, 0].set_ylim(-1, 1)  # set axes range

    axs[2, 1].plot(t, data["wz_cmd"], label="wz_cmds")
    axs[2, 1].plot(t, data["wy_cmd"], label="wy_cmds")
    axs[2, 1].set_title("Rate Commands")
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    axs[2, 1].set_ylabel("Rate (rad/s)")

    # axs[2, 1].axis("off")

    for ax in axs.flat:
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show(block=True)

# --- 1. Data Loading and Preprocessing ---
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return None
        # return generate_mock_data()

    # New: load CSV-like text with header
    # Expected header:
    # t,p,q,r,wy,theta,thetadot,t_mot_p,wz,psi,psidot,t_mot_y,pitchrate_cmds,yawrate_cmds
    data = np.genfromtxt(filepath, delimiter=",", names=True, dtype=None, encoding="utf-8")

    # Map to expected keys (based on header)
    return {
        "time": data["t"],
        # Yaw (psi)
        "AzMotorCurrent": data["t_mot_y"],
        "AzMotorRate":    data["psidot"],
        "AzLoadAngle":    data["psi"],
        "LoadrRate":      data["wz"],
        # Pitch (theta)
        "ElMotorCurrent": data["t_mot_p"],
        "ElMotorRate":    data["thetadot"],
        "ElLoadAngle":    data["theta"],
        "LoadqRate":      data["wy"],
        # base rates (not used)
        "BaseRateX": data["p"],
        "BaseRateY": data["q"],
        "BaseRateZ": data["r"],
        # commands
        "wy_cmd":   data["pitchrate_cmds"],
        "wz_cmd":   data["yawrate_cmds"],
    }

def preprocess_data(data, dt=0.01, N=-1):
    """
    Extracts state vectors (assuming no backlash, direct drive).
    State vector x = [Theta_l, Omega_l]
    """
    # 1. Time vector
    if 'time' in data:
        t = np.array(data['time'])
        # Derive dt from time if available
        dt = float(np.mean(np.diff(t)))
    else:
        # Assume constant rate based on length
        t = np.arange(len(data['LoadrRate'])) * dt

    if(N == -1):
        N = len(t)

    # 2. Extract States
    
    # Load states
    # vm = np.array(data['AzMotorRate']) # use when incorporating base rates
    xl = np.array(data['AzLoadAngle'])
    vl = np.array(data['LoadrRate'])
    
    # Input
    u = np.array(data['AzMotorCurrent'])

    X_yaw = np.stack([xl, vl], axis=1) # Shape (N, 2)
    # we can derivate relative_rate from load angle directly or use the provided rate

    yaw_data = {
        't': t[N:]-t[N],
        'X': X_yaw[N:],
        'u': u[N:],
        'cmd': data['wz_cmd'][N:],
        'dt': dt
    }

    # Load states
    # vm = np.array(data['ElMotorRate']) # use when incorporating base rates   
    xl = np.array(data['ElLoadAngle'])
    vl = np.array(data['LoadqRate'])
    
    # Input
    u = np.array(data['ElMotorCurrent'])

    X_pitch = np.stack([xl, vl], axis=1) # Shape (N, 2)

    pitch_data = {
        't': t[:N]-t[0],
        'X': X_pitch[:N],
        'u': u[:N],
        'cmd': data['wy_cmd'][:N],
        'dt': dt
    }

    return yaw_data, pitch_data

def calculate_derivatives(X, dt):
    """
    Calculates time derivatives X_dot.
    We enforce X_dot[0] = X[1] and X_dot[2] = X[3] later, 
    but we need derivatives for states 1 and 3 (velocities) to fit accelerations.
    """
    X_dot = np.zeros_like(X)
    
    # Derivatives for positions are exactly velocities (Kinematic constraint)
    X_dot[:, 0] = X[:, 1]
    
    # Numerical derivatives for velocities (Accelerations)
    # Savitzky-Golay filter for smooth differentiation
    X_dot[:, 1] = savgol_filter(X[:, 1], window_length=11, polyorder=3, deriv=1, delta=dt)
    
    return X_dot

# --- 2. SINDy Algorithm (Lasso) ---

class SINDy1DGimbal:
    def __init__(self, threshold=0.005):
        # NOTE: When using Lasso, 'threshold' effectively acts as the Alpha (L1 regularization strength).
        # A smaller alpha (e.g. 0.001) is often needed compared to a hard threshold (0.01).
        self.threshold = threshold
        self.coefs = None
        # Features: [1, xl, vl, u, sign(vl), sin(xl), cos(xl), v_l|vl|]
        self.feature_names = ['1', 'x_l', 'v_l', 'u', 
                              'sgn(v_l)', 'sin(x_l)', 'v_l|v_l|']

    def build_library(self, X, u):
        """
        Constructs the library matrix Theta.
        Rows = time steps, Cols = candidate functions.
        """
        xl = X[:, 0]
        vl = X[:, 1]
        
        Theta = np.column_stack([
            np.ones_like(xl),       # 0: Bias
            xl,                     # 1: Load Pos
            vl,                     # 2: Load Vel
            u,                      # 3: Input
            np.sign(vl),            # 4: Coulomb Load
            np.sin(xl),             # 5: Gravity (Sin)
            vl * np.abs(vl)         # 6: Quadratic Drag
        ])
        return Theta

    def fit(self, X, X_dot, u):
        """
        Uses sklearn.linear_model.Lasso to find sparse dynamics.
        We fit equations for v_l_dot (idx 1) 
        """
        Theta = self.build_library(X, u)
        Y = X_dot[:, [1]] # Targets: Accel
        
        # Initialize coefficient matrix [n_features, 1]
        self.coefs = np.zeros((Theta.shape[1], 1))
        
        # Initialize Lasso
        # alpha is the L1 regularization strength (using self.threshold)
        # fit_intercept=False because Theta already has a column of ones.
        lasso_model = Lasso(alpha=self.threshold, fit_intercept=False, max_iter=50000)#, tol=1e-4)
        
        # Fit for Load Acceleration
        lasso_model.fit(Theta, Y[:, 0])
        self.coefs[:, 0] = lasso_model.coef_
        
        return self.coefs

    def print_equations(self):
        if self.coefs is None:
            print("Model not fitted.")
            return
        
        targets = ["v_l_dot"]
        
        print("\n--- Discovered Equations of Motion (Lasso) ---")
        
        # Kinematics (Hardcoded)
        print(f"d(x_l)/dt = 1.000 * v_l")
        
        # Dynamics (Learned)
        for i, target_name in enumerate(targets):
            eq_str = f"{target_name} = "
            terms = []
            for j, coef in enumerate(self.coefs[:, i]):
                if abs(coef) > 1e-2:
                    terms.append(f"{coef:+.4f}*{self.feature_names[j]}")
            print(eq_str + " ".join(terms))
            
        print("--------------------------------------\n")

        # Extract Gear Ratio N
        # beta  = self.coefs[1, 0] # Coef of xl in Load Equation

def validate_model_response(sindy_model, t, X_true, u_true):
    """
    Simulates the discovered model using the real input u and compares to X_true.
    """
    print("\n=== Validating Model (Open-Loop Simulation) ===")
    
    # Create interpolator for input u(t)
    u_func = interp1d(t, u_true, kind='zero', fill_value="extrapolate")
    
    def dynamics(t_now, y):
        # y = [xl, vl]
        xl, vl = y
        u_now = float(u_func(t_now))
        
        state_vec = np.array([[xl, vl]])
        u_vec = np.array([u_now])
        
        # Build library row (Single step)
        Theta = sindy_model.build_library(state_vec, u_vec)
        
        # Predict accelerations
        accs = Theta @ sindy_model.coefs
        acc_l = accs[0, 0]
        
        return [vl, acc_l]

    # Integrate
    y0 = X_true[0, :]
    sol = solve_ivp(dynamics, (t[0], t[-1]), y0, t_eval=t, method='RK45')
    
    X_sim = sol.y.T # Shape (N, 4)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    state_names = ['Load Angle (Rad)', 'Load Rate (Rad/s)']
    
    for i in range(2):
        plt.subplot(2, 1, i+1)
        plt.plot(t, X_true[:, i], 'k-', label='Real Data', alpha=0.6)
        plt.plot(sol.t, X_sim[:, i], 'r--', label='SINDy ID', linewidth=1.5)
        plt.title(state_names[i])
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)
        
    plt.suptitle(f"SINDy Model Verification (open-loop) vs Real Data", fontsize=16)
    plt.tight_layout()
    # plt.savefig('sindy_verification.png')
    print("Verification plot generated.")


# --- 3. Differentiable Plant (PyTorch) ---

class DifferentiablePlant(nn.Module):
    def __init__(self, sindy_coefs):
        super().__init__()
        # sindy_coefs shape: (n_features, 2)
        self.W = torch.tensor(sindy_coefs, dtype=torch.float32)
        
    def forward(self, x, u):
        """
        x: [batch, 2] -> [xl, vl]
        u: [batch, 1]
        Returns x_dot
        """
        xl = x[:, 0:1]
        vl = x[:, 1:2]
        
        # Matches build_library structure
        # [1, xm, vm, xl, vl, u, sgn(vl), sin(xl), vl|vl|]
        Theta = torch.cat([
            torch.ones_like(xl),
            xl, vl, u,
            torch.tanh(100*vl), # Smooth sign
            torch.sin(xl),
            vl * torch.abs(vl)
        ], dim=1)
        
        # Compute Accelerations: Theta * W
        accs = torch.matmul(Theta, self.W)
        
        acc_l = accs[:, 0:1]
        
        x_dot = torch.cat([vl, acc_l], dim=1)
        return x_dot

# --- 4. Controller Optimization ---

class PIDController(nn.Module):
    def __init__(self, kp=1.0, ki=0.1, kd=0.5):
        super().__init__()
        # Initialize gains
        self.raw_Kp = nn.Parameter(torch.tensor(kp))
        self.raw_Ki = nn.Parameter(torch.tensor(ki))
        self.raw_Kd = nn.Parameter(torch.tensor(kd))
        
    def get_gains(self):
        return self.raw_Kp, self.raw_Ki, self.raw_Kd

    def forward(self, error, integral_error, velocity):
        Kp, Ki, Kd = self.get_gains()
        return Kp * error + Ki * integral_error - Kd * velocity

class ClosedLoopSystem(nn.Module):
    def __init__(self, plant, controller, cmd=None, t_vec=None):
        super().__init__()
        self.plant = plant
        self.controller = controller
        self.cmd = torch.as_tensor(cmd, dtype=torch.float32) if cmd is not None else None
        self.t_vec = torch.as_tensor(t_vec, dtype=torch.float32) if t_vec is not None else None

    def forward(self, t, state):
        # state: [xl, vl, integral_error]
        x_sys   = state[0:2].unsqueeze(0) # Batch dim
        int_err = state[2].unsqueeze(0)
        
        # get command at time t
        idx = (self.t_vec - t).abs().argmin().item()
        target = self.cmd[idx] #1.0 if t > 0.1 else 0.0
        
        # Controller uses Load Angle (idx 0) and Load Velocity (idx 1)
        current_pos = x_sys[:, 0]
        current_vel = x_sys[:, 1]
        
        error = target - current_vel
        
        u = self.controller(error, int_err, current_vel)
        u = u.unsqueeze(1) # [1, 1]
        
        # Plant dynamics
        dx_sys = self.plant(x_sys, u)
        
        # Integral error dynamics
        d_int_err = error
        
        return torch.cat([dx_sys.squeeze(0), d_int_err])

def train_controller(plant_model, epochs=100):
    print("\n--- Optimizing PID Controller using Differential Programming ---")
    
    Kt = 0.0150 # Motor torque constant (Nm/A)
    # these are starting points based on manual tuning (simulink) multiplied by Kt
    # because we get the torque in the recordings, not the current like in simulink
    controller = PIDController(kp=180.0*Kt, ki=40000.0*Kt, kd=0.04*Kt)
    # controller = PIDController(kp=10.0, ki=0.0, kd=0.0)
    t_eval = torch.linspace(0, 4, int(0.5*1200))
    cmd_eval = torch.tensor([1.0 if t > 0.1 else 0.0 for t in t_eval])

    system = ClosedLoopSystem(plant_model, controller, 
                              cmd=cmd_eval,
                              t_vec=t_eval)
    
    optimizer = optim.Adam(controller.parameters(), lr=0.05)
    
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        x0 = torch.zeros(3) # Start at rest, 0 integral error
        
        # Simulate
        traj = odeint(system, x0, t_eval, method='rk4')
        
        # Extract Load rate (vl) as the output we want to control
        v_load = traj[:, 1]
        
        # Target vector
        target = cmd_eval.clone() # Step to 1.0 rad/s at t > 0.1s
        
        # Loss
        loss = torch.mean((v_load - target)**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            kp, ki, kd = controller.get_gains()
            print(f"Epoch {epoch}: Loss={loss.item():.3e} | Kp={kp.item():.2f}, Ki={ki.item():.2f}, Kd={kd.item():.2f}")
            
    return controller


def discover_dynamics(data, X_dot):
    # Reduced threshold for Lasso (serves as alpha)
    # 0.001 is a good starting point for normalized-scale data in Lasso
    plant = SINDy1DGimbal(threshold=0.001) 
    sindy_coefs = plant.fit(data['X'], X_dot, data['u'])
    plant.print_equations()

    validate_model_response(plant, data['t'], data['X'], data['u']) # Plot verification
    
    return sindy_coefs

def optimize_controller(sindy_coefs):
    plant_torch = DifferentiablePlant(sindy_coefs)
    best_controller = train_controller(plant_torch)
    
    Kp, Ki, Kd = best_controller.get_gains()
    print(f"\nFINAL RESULT:\nOptimal PID Gains: Kp={Kp.item():.4f}, Ki={Ki.item():.4f}, Kd={Kd.item():.4f}")
    
    return best_controller, plant_torch

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data
    # filepath = "PedTelem_20230722092156.pkl" 
    # filepath = "PedTelem_20230629012751.pkl" 
    filepath = "pitch_then_yaw_rate_cmds.txt" 
    
    data = load_data(filepath)
    plot_raw_data(data)

    N = len(data['time']//2) # where to split yaw/pitch data (assuming they are concatenated)

    # 2. Preprocess
    yaw_data, pitch_data = preprocess_data(data, N=int(N/2)) # t, X, u, dt
    X_yaw_dot   = calculate_derivatives(yaw_data['X'], yaw_data['dt'])
    X_pitch_dot = calculate_derivatives(pitch_data['X'], pitch_data['dt'])
    
    # 3. Discover Dynamics (SINDy with Lasso)
    print("Discovering Dynamics for Yaw...")
    sindy_coefs_yaw = discover_dynamics(yaw_data, X_yaw_dot)
    best_controller_yaw, plant_torch_yaw = optimize_controller(sindy_coefs_yaw)
    
    # 5. Visualize Result
    system_yaw = ClosedLoopSystem(plant_torch_yaw, best_controller_yaw, 
                                  cmd=yaw_data['cmd'], t_vec=yaw_data['t'])
    # t_sim = torch.linspace(0, 5, 200)
    t_sim = torch.tensor(yaw_data["t"], dtype=torch.float32)
    with torch.no_grad():
        traj_yaw = odeint(system_yaw, torch.zeros(3), t_sim)
    
    plt.figure()
    plt.plot(t_sim, traj_yaw[:, 1], label="Rate (Simulated)")
    plt.plot(yaw_data["t"], yaw_data['X'][:, 1], label="Rate (measured)", alpha=0.6)
    # plt.plot(t_sim, [1.0 if t>0.1 else 0.0 for t in t_sim], 'k--', label="Target")
    plt.title("Optimized Controller Response (Yaw)")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (rad/s)")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

    # print("Discovering Dynamics for Pitch...")
    # sindy_coefs_pitch = discover_dynamics(pitch_data, X_pitch_dot)
    # best_controller_pitch, plant_torch_pitch = optimize_controller(sindy_coefs_pitch)

    # system_pitch = ClosedLoopSystem(plant_torch_pitch, best_controller_pitch, cmd=pitch_data['cmd'], t_vec=pitch_data['t'])
    # t_sim = torch.tensor(pitch_data["t"], dtype=torch.float32)
    # with torch.no_grad():
    #     traj_pitch = odeint(system_pitch, torch.zeros(3), t_sim)
    
    # plt.figure()
    # plt.plot(t_sim, traj_pitch[:, 2], label="Rate (Simulated)")
    # plt.plot(pitch_data["t"], pitch_data['X'][:, 2], label="Rate (measured)", alpha=0.6)
    # plt.title("Optimized Controller Response (Pitch)")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Rate (rad/s)")
    # plt.legend()
    # plt.grid(True)
    # plt.show(block=True)