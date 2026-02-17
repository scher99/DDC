%% DeePC Offline Data Preparation
% Assumes you have u_sim and y_sim from your PID experiment workspace

vv=[floor(6/dt):floor(10/dt)];
u_sim = out.u_simout.signals.values(vv);
y_sim = out.y_simout.signals.values(vv);

T_total = length(u_sim); % Total number of samples
T_ini = 20;             % Initial horizon (should be > max system delay) This needs to be long enough to capture the "state" of the system (its inertia and current behavior). Usually, $T_{ini}$ is set to be at least the order of the system. For a vehicle or complex robot, $T_{ini} = 10 \text{ to } 50$ samples is common ($10\text{ms}$ to $50\text{ms}$).
N = 50;                 % Prediction horizon This is how far into the future the controller "looks." If you want to look $0.1\text{s}$ ahead, $N = 100$.
L = T_ini + N;          % Total window length per column

% $T$ (Recorded Data Length): To ensure the Hankel matrix is "persistently exciting" (mathematically solvable), you need:$$T \ge (m+1)(T_{ini} + N + n) - 1$$Where $m$ is the number of inputs and $n$ is the system order. Practically, for a $1000\text{Hz}$ sim, try to record 2 to 5 seconds of data ($T = 2000$ to $5000$ samples) to get a robust model.

% 1. Construct Hankel Matrix for Inputs (U) and Outputs (Y)
% A Hankel matrix shifts the data sequence down each column
U_hankel = hankel(u_sim(1:L), u_sim(L:end));
Y_hankel = hankel(y_sim(1:L), y_sim(L:end));

% 2. Split into "Past" and "Future"
% Past data (p) is used to estimate the current 'state'
% Future data (f) is used to predict the trajectory
Up = U_hankel(1:T_ini, :);
Yp = Y_hankel(1:T_ini, :);
Uf = U_hankel(T_ini+1:end, :);
Yf = Y_hankel(T_ini+1:end, :);

% 3. Define Weights (Tuning Knobs)
% Q = diag(repmat(10, 1, N));   % Output tracking weight
% R = diag(repmat(0.1, 1, N));  % Control effort penalty
% lambda_g = 100;               % Stability regularization (Crucial!)
% lambda_s = 1e6;               % Noise slack penalty

% Save these to your workspace for the Simulink block
save('DeePC_Data_gimbal.mat', 'Up', 'Yp', 'Uf', 'Yf');