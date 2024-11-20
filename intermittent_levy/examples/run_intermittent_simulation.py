# run_intermittent_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from intermittent_levy.processes import intermittent3
from intermittent_levy.moments import mom2_serg_log, mom4_serg_log
from intermittent_levy.optimization import to_optimize_mom4_and_2_serg_log
from intermittent_levy.classification import form_groups
from intermittent_levy.utils import adjusted_r_square
from scipy import optimize
import os  # Import os module

# Define the results directory
results_dir = 'results/intermittent'
os.makedirs(results_dir, exist_ok=True)

# Initialize parameters
Nr_iterations = 200  # Adjust as needed
tau_list = np.arange(1, 20)  # Define your tau_list appropriately

# Lists to store results
int_params = []n
list_X_traj = []
list_Y_traj = []
r_squared_int = []
opt_list_int_params = []
int_fit_list_mom2 = []
int_fit_list_mom4 = []
gen_dx4_log_list = []
gen_dx2_log_list = []

# Collect lists for plotting at the end
all_dx2_log = []
all_dx4_log = []
all_int_fit_2 = []
all_int_fit_4 = []

# Begin simulation loop
for itera in range(Nr_iterations):
    print(f"Iteration {itera + 1}/{Nr_iterations}")

    # Simulation parameters
    N = 300000
    integration_factor = 3
    g_tau = 1
    gv_list = [8, 50]
    gD_list = [0.02, 1]
    glambdaB_list = [0.02, 1]
    glambdaD_list = [0.0005, 0.02]

    # Randomly select parameters
    g_v0 = np.random.uniform(gv_list[0], gv_list[1])
    g_D = np.random.uniform(gD_list[0], gD_list[1])
    g_lambda_B = np.random.uniform(glambdaB_list[0], glambdaB_list[1])
    g_lambda_D = np.random.uniform(glambdaD_list[0], glambdaD_list[1])

    # Simulate intermittent process
    xsynth, ysynth = intermittent3(
        N * integration_factor,
        g_tau / integration_factor,
        g_v0,
        g_D,
        g_lambda_B,
        g_lambda_D
    )

    # Store parameters and trajectories
    int_params.append([g_v0, g_D, g_lambda_B, g_lambda_D])
    list_X_traj.append(xsynth)
    list_Y_traj.append(ysynth)

    # Compute moments
    dx2 = []
    dx4 = []
    for tau in tau_list:
        lldx = np.diff(xsynth[::int(tau)])
        lldy = np.diff(ysynth[::int(tau)])
        dx2.append(np.mean(lldx**2 + lldy**2))
        dx4.append(np.mean((lldx**2 + lldy**2)**2))

    dx2_log = np.log(dx2)
    dx4_log = np.log(dx4)
    all_dx2_log.append(dx2_log)
    all_dx4_log.append(dx4_log)

    # Classification
    dS = np.sqrt(np.diff(xsynth)**2 + np.diff(ysynth)**2)
    raw_threshold_array = np.linspace(np.min(dS[dS > 0]), np.max(dS), 20)
    threshold_array = raw_threshold_array / np.max(dS)
    detection, detectionfisher, lkmin, lfishermin = form_groups(
        dS, threshold_array, graph=False, x_label='v', title='title', x_axis_format='%.2f'
    )
    lthreshold = raw_threshold_array[np.argmin(detection)]
    binary_vector = (dS > lthreshold).astype(int)
    Nfix = len(binary_vector) - np.sum(binary_vector)
    Nsacc = np.sum(binary_vector)
    Ntransi = int(np.sum(np.abs(np.diff(binary_vector))) / 2)
    est_lambda_B = -np.log(1 - Ntransi / Nsacc)
    est_lambda_D = -np.log(1 - Ntransi / Nfix)
    g_v0_est = np.mean(dS[binary_vector == 1])
    g_D_est = np.mean(dS[binary_vector == 0]) / 1.8

    # Optimization setup
    rranges = [
        (g_v0_est / 10, g_v0_est * 10),
        (g_D_est / 10, g_D_est * 10),
        (est_lambda_B / 10, est_lambda_B * 10),
        (est_lambda_D / 10, est_lambda_D * 10)
    ]
    args = (tau_list, dx2_log, dx4_log)

    # Optimization
    result = optimize.dual_annealing(
        to_optimize_mom4_and_2_serg_log,
        bounds=rranges,
        args=args
    )
    best_params = result.x
    opt_list_int_params.append(best_params)

    # Compute fitted moments
    int_fit_2 = mom2_serg_log(tau_list, *best_params)
    int_fit_4 = mom4_serg_log(tau_list, *best_params)
    all_int_fit_2.append(int_fit_2)
    all_int_fit_4.append(int_fit_4)

    # Calculate adjusted R-squared
    r2_mom2 = adjusted_r_square(dx2_log, int_fit_2, degrees_freedom=4)
    r2_mom4 = adjusted_r_square(dx4_log, int_fit_4, degrees_freedom=4)
    r_squared_int.append([r2_mom4, r2_mom2])

    int_fit_list_mom2.append(int_fit_2)
    int_fit_list_mom4.append(int_fit_4)
    gen_dx4_log_list.append(dx4_log)
    gen_dx2_log_list.append(dx2_log)

# Convert lists to arrays for easy averaging
all_dx2_log = np.array(all_dx2_log)
all_dx4_log = np.array(all_dx4_log)
all_int_fit_2 = np.array(all_int_fit_2)
all_int_fit_4 = np.array(all_int_fit_4)

# Calculate the average across all iterations
avg_dx2_log = np.mean(all_dx2_log, axis=0)
avg_dx4_log = np.mean(all_dx4_log, axis=0)
avg_int_fit_2 = np.mean(all_int_fit_2, axis=0)
avg_int_fit_4 = np.mean(all_int_fit_4, axis=0)

# Plot the average results
plt.figure(figsize=(10, 5))

# Plotting average empirical log M2 and average fitted log M2
plt.subplot(1, 2, 1)
plt.plot(np.log(tau_list), avg_dx2_log, 'o', label='Average Empirical log M2')
plt.plot(np.log(tau_list), avg_int_fit_2, '-', label='Average Fitted log M2')
plt.xlabel('log(tau)')
plt.ylabel('log M2')
plt.legend()

# Plotting average empirical log M4 and average fitted log M4
plt.subplot(1, 2, 2)
plt.plot(np.log(tau_list), avg_dx4_log, 'o', label='Average Empirical log M4')
plt.plot(np.log(tau_list), avg_int_fit_4, '-', label='Average Fitted log M4')
plt.xlabel('log(tau)')
plt.ylabel('log M4')
plt.legend()

plt.tight_layout()
plt.show()
# Saving the results
np.savetxt(os.path.join(results_dir, 'int_generated_params.txt'), int_params)
np.savetxt(os.path.join(results_dir, 'int_generated_r_squared_int.txt'), r_squared_int)
np.savetxt(os.path.join(results_dir, 'int_generated_opt_list_int_params.txt'), opt_list_int_params)
np.savetxt(os.path.join(results_dir, 'int_generated_int_fit_list_mom2.txt'), int_fit_list_mom2)
np.savetxt(os.path.join(results_dir, 'int_generated_int_fit_list_mom4.txt'), int_fit_list_mom4)
np.savetxt(os.path.join(results_dir, 'int_generated_logdx2_list.txt'), gen_dx2_log_list)
np.savetxt(os.path.join(results_dir, 'int_generated_logdx4_list.txt'), gen_dx4_log_list)