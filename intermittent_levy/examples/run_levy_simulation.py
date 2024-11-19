# run_levy_simulation.py

'''import numpy as np
import matplotlib.pyplot as plt
from intermittent_levy.processes import levy_flight_2D_Simplified
from intermittent_levy.moments import levy_moments_log
from intermittent_levy.optimization import to_optimize_levy
from intermittent_levy.utils import adjusted_r_square, adjusted_r_square_array
from scipy import optimize
import os  # Import os module

# Define the results directory
results_dir = 'results/levy'
os.makedirs(results_dir, exist_ok=True)

# Initialize parameters
Nr_iterations = 200
tau_list = np.arange(1, 20)


# Parameter ranges for simulation
gtmin_list = [0.005, 0.05]
gv_list = [10, 30]
galfa_list = [2.6, 2.95]
measuring_dt = 1

# Lists to store results
lev_params = []
list_X_traj = []
list_Y_traj = []
r_squared_lev = []
lev_fit_list_mom2 = []
lev_fit_list_mom4 = []
opt_list_lev_params = []
gen_dx4_log_list = []
gen_dx2_log_list = []

for itera in range(Nr_iterations):
    print(f"Iteration {itera + 1}/{Nr_iterations}")
    # Randomly select parameters
    galfa = np.random.uniform(galfa_list[0], galfa_list[1])
    gtmin = np.random.uniform(gtmin_list[0], gtmin_list[1])
    gv = np.random.uniform(gv_list[0], gv_list[1])

    print(f"Simulating with alpha={galfa}, tmin={gtmin}, v_mean={gv}")

    # Estimate number of redirections
    k = 20 * (galfa - 2)
    G_redirect = int(90000 * (10 ** (0.05 * k)) / gtmin)

    # Simulate Lévy flight
    xsynth, ysynth = levy_flight_2D_Simplified(
        G_redirect, 300000, galfa, gtmin, gv, measuring_dt
    )

    # Store parameters and trajectories
    lev_params.append([gv, gtmin, galfa])
    list_X_traj.append(xsynth)
    list_Y_traj.append(ysynth)

    # Compute moments
    dx2 = []
    dx4 = []
    for tau in tau_list:
        lldx = np.diff(xsynth[::int(tau)])
        lldy = np.diff(ysynth[::int(tau)])
        dx2.append(np.mean(lldx ** 2 + lldy ** 2))
        dx4.append(np.mean((lldx ** 2 + lldy ** 2) ** 2))

    dx2_log = np.log(dx2)
    dx4_log = np.log(dx4)

    gen_dx2_log_list.append(dx2_log)
    gen_dx4_log_list.append(dx4_log)

    # Initial guesses
    initial_alpha = 2.5
    initial_v_mean = gv

    # Optimization bounds
    bounds = [(2.1, 2.99), (gv / 10, gv * 10)]  # alpha between 2.1 and 2.99, v_mean around gv

    # Perform optimization
    result = optimize.dual_annealing(
        to_optimize_levy,
        bounds=bounds,
        args=(tau_list, dx2_log, dx4_log, gtmin)
    )

    optimized_alpha, optimized_v_mean = result.x

    # Calculate fitted moments
    fitted_dx2_log = levy_moments_log(2, optimized_alpha, optimized_v_mean, tau_list, gtmin)
    fitted_dx4_log = levy_moments_log(4, optimized_alpha, optimized_v_mean, tau_list, gtmin)

    # Calculate adjusted R-squared
    r2_dx2 = adjusted_r_square(dx2_log, fitted_dx2_log, degrees_freedom=2)
    r2_dx4 = adjusted_r_square(dx4_log, fitted_dx4_log, degrees_freedom=2)

    r_squared_lev.append([r2_dx4, r2_dx2])
    lev_fit_list_mom2.append(fitted_dx2_log)
    lev_fit_list_mom4.append(fitted_dx4_log)
    opt_list_lev_params.append([optimized_v_mean, gtmin, optimized_alpha])

# Convert lists to arrays for averaging
gen_dx2_log_array = np.array(gen_dx2_log_list)
gen_dx4_log_array = np.array(gen_dx4_log_list)
lev_fit_list_mom2_array = np.array(lev_fit_list_mom2)
lev_fit_list_mom4_array = np.array(lev_fit_list_mom4)

# Calculate the average across all iterations
avg_dx2_log = np.mean(gen_dx2_log_array, axis=0)
avg_dx4_log = np.mean(gen_dx4_log_array, axis=0)
avg_fitted_dx2_log = np.mean(lev_fit_list_mom2_array, axis=0)
avg_fitted_dx4_log = np.mean(lev_fit_list_mom4_array, axis=0)

# Plot the average results
plt.figure(figsize=(10, 5))

# Plotting average empirical log M2 and average fitted log M2
plt.subplot(1, 2, 1)
plt.plot(np.log(tau_list), avg_dx2_log, 'o', label='Average Empirical log M2')
plt.plot(np.log(tau_list), avg_fitted_dx2_log, '-', label='Average Fitted log M2')
plt.xlabel('log(tau)')
plt.ylabel('log(M2)')
plt.legend()

# Plotting average empirical log M4 and average fitted log M4
plt.subplot(1, 2, 2)
plt.plot(np.log(tau_list), avg_dx4_log, 'o', label='Average Empirical log M4')
plt.plot(np.log(tau_list), avg_fitted_dx4_log, '-', label='Average Fitted log M4')
plt.xlabel('log(tau)')
plt.ylabel('log(M4)')
plt.legend()

plt.tight_layout()
plt.show()

# After all iterations, you can save the collected data
np.savetxt(os.path.join(results_dir, 'lev_generated_params.txt'), lev_params)
np.savetxt(os.path.join(results_dir, 'lev_generated_r_squared_lev.txt'), r_squared_lev)
np.savetxt(os.path.join(results_dir, 'lev_generated_opt_list_lev_params.txt'), opt_list_lev_params)
np.savetxt(os.path.join(results_dir, 'lev_generated_lev_fit_list_mom2.txt'), lev_fit_list_mom2)
np.savetxt(os.path.join(results_dir, 'lev_generated_lev_fit_list_mom4.txt'), lev_fit_list_mom4)
np.savetxt(os.path.join(results_dir, 'lev_generated_logdx2_list.txt'), gen_dx2_log_list)
np.savetxt(os.path.join(results_dir, 'lev_generated_logdx4_list.txt'), gen_dx4_log_list)'''
import numpy as np
import matplotlib.pyplot as plt
from intermittent_levy.processes import levy_flight_2D_Simplified
from intermittent_levy.moments import levy_moments_log
from intermittent_levy.optimization import to_optimize_levy
from intermittent_levy.utils import adjusted_r_square, adjusted_r_square_array
from intermittent_levy.classification import form_groups
from scipy import optimize
import os

# Define the results directory
results_dir = 'results/levy'
os.makedirs(results_dir, exist_ok=True)

# Initialize parameters
Nr_iterations = 10
tau_list = np.arange(1, 20)

# Parameter ranges for simulation
gtmin_list = [0.005, 0.05]
gv_list = [10, 30]
galfa_list = [2.6, 2.95]
measuring_dt = 1

# Lists to store results
lev_params = []
list_X_traj = []
list_Y_traj = []
r_squared_lev = []
lev_fit_list_mom2 = []
lev_fit_list_mom4 = []
opt_list_lev_params = []
gen_dx4_log_list = []
gen_dx2_log_list = []
lev_lambda_BD_list = []
lev_lambda_DB_list = []

for itera in range(Nr_iterations):
    print(f"Iteration {itera + 1}/{Nr_iterations}")
    # Randomly select parameters
    galfa = np.random.uniform(galfa_list[0], galfa_list[1])
    gtmin = np.random.uniform(gtmin_list[0], gtmin_list[1])
    gv = np.random.uniform(gv_list[0], gv_list[1])

    print(f"Simulating with alpha={galfa}, tmin={gtmin}, v_mean={gv}")

    # Estimate number of redirections
    k = 20 * (galfa - 2)
    G_redirect = int(90000 * (10 ** (0.05 * k)) / gtmin)

    # Simulate Lévy flight
    xsynth, ysynth = levy_flight_2D_Simplified(
        G_redirect, 300000, galfa, gtmin, gv, measuring_dt
    )

    # Store parameters and trajectories
    lev_params.append([gv, gtmin, galfa])
    list_X_traj.append(xsynth)
    list_Y_traj.append(ysynth)

    # Compute moments
    dx2 = []
    dx4 = []
    for tau in tau_list:
        lldx = np.diff(xsynth[::int(tau)])
        lldy = np.diff(ysynth[::int(tau)])
        dx2.append(np.mean(lldx ** 2 + lldy ** 2))
        dx4.append(np.mean((lldx ** 2 + lldy ** 2) ** 2))

    dx2_log = np.log(dx2)
    dx4_log = np.log(dx4)

    gen_dx2_log_list.append(dx2_log)
    gen_dx4_log_list.append(dx4_log)

    # Classification step to estimate lambda_BD and lambda_DB
    dS = np.sqrt(np.diff(xsynth) ** 2 + np.diff(ysynth) ** 2)
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

    # Store estimated lambdas
    lev_lambda_BD_list.append(est_lambda_B)
    lev_lambda_DB_list.append(est_lambda_D)

    # Initial guesses
    initial_alpha = 2.5
    initial_v_mean = gv

    # Optimization bounds
    bounds = [(2.1, 2.99), (gv / 10, gv * 10)]  # alpha between 2.1 and 2.99, v_mean around gv

    # Perform optimization
    result = optimize.dual_annealing(
        to_optimize_levy,
        bounds=bounds,
        args=(tau_list, dx2_log, dx4_log, gtmin)
    )

    optimized_alpha, optimized_v_mean = result.x

    # Calculate fitted moments
    fitted_dx2_log = levy_moments_log(2, optimized_alpha, optimized_v_mean, tau_list, gtmin)
    fitted_dx4_log = levy_moments_log(4, optimized_alpha, optimized_v_mean, tau_list, gtmin)

    # Calculate adjusted R-squared
    r2_dx2 = adjusted_r_square(dx2_log, fitted_dx2_log, degrees_freedom=2)
    r2_dx4 = adjusted_r_square(dx4_log, fitted_dx4_log, degrees_freedom=2)

    r_squared_lev.append([r2_dx4, r2_dx2])
    lev_fit_list_mom2.append(fitted_dx2_log)
    lev_fit_list_mom4.append(fitted_dx4_log)
    opt_list_lev_params.append([optimized_v_mean, gtmin, optimized_alpha])

# Convert lists to arrays for averaging
gen_dx2_log_array = np.array(gen_dx2_log_list)
gen_dx4_log_array = np.array(gen_dx4_log_list)
lev_fit_list_mom2_array = np.array(lev_fit_list_mom2)
lev_fit_list_mom4_array = np.array(lev_fit_list_mom4)

# Calculate the average across all iterations
avg_dx2_log = np.mean(gen_dx2_log_array, axis=0)
avg_dx4_log = np.mean(gen_dx4_log_array, axis=0)
avg_fitted_dx2_log = np.mean(lev_fit_list_mom2_array, axis=0)
avg_fitted_dx4_log = np.mean(lev_fit_list_mom4_array, axis=0)

# Plot the average results
plt.figure(figsize=(10, 5))

# Plotting average empirical log M2 and average fitted log M2
plt.subplot(1, 2, 1)
plt.plot(np.log(tau_list), avg_dx2_log, 'o', label='Average Empirical log M2')
plt.plot(np.log(tau_list), avg_fitted_dx2_log, '-', label='Average Fitted log M2')
plt.xlabel('log(tau)')
plt.ylabel('log(M2)')
plt.legend()

# Plotting average empirical log M4 and average fitted log M4
plt.subplot(1, 2, 2)
plt.plot(np.log(tau_list), avg_dx4_log, 'o', label='Average Empirical log M4')
plt.plot(np.log(tau_list), avg_fitted_dx4_log, '-', label='Average Fitted log M4')
plt.xlabel('log(tau)')
plt.ylabel('log(M4)')
plt.legend()

plt.tight_layout()
plt.show()

# Save the collected data
np.savetxt(os.path.join(results_dir, 'lev_generated_params.txt'), lev_params)
np.savetxt(os.path.join(results_dir, 'lev_generated_r_squared_lev.txt'), r_squared_lev)
np.savetxt(os.path.join(results_dir, 'lev_generated_opt_list_lev_params.txt'), opt_list_lev_params)
np.savetxt(os.path.join(results_dir, 'lev_generated_lev_fit_list_mom2.txt'), lev_fit_list_mom2)
np.savetxt(os.path.join(results_dir, 'lev_generated_lev_fit_list_mom4.txt'), lev_fit_list_mom4)
np.savetxt(os.path.join(results_dir, 'lev_generated_logdx2_list.txt'), gen_dx2_log_list)
np.savetxt(os.path.join(results_dir, 'lev_generated_logdx4_list.txt'), gen_dx4_log_list)
np.savetxt(os.path.join(results_dir, 'lev_generated_lambda_BD.txt'), lev_lambda_BD_list)
np.savetxt(os.path.join(results_dir, 'lev_generated_lambda_DB.txt'), lev_lambda_DB_list)

