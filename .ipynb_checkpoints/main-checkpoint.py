import os
import sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit, dual_annealing
from scipy.stats import linregress
# Determine the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to 'intermittentLevy' directory relative to the script directory
intermittentLevy_path = os.path.join(script_dir, 'intermittentLevy')
sys.path.append(intermittentLevy_path)

# Now import your custom functions
from functions import (intermittent2,levy_flight_2D_2, load_parameters, setup_kde,
                       perform_iterations, mom4_serg_log,
                       to_optimize_mom4_serg_log, to_optimize_mom22_4_diff_serg_log,
                       mom22_4_diff_serg_log)

# Load parameters and set up KDE
normed_loc_params, mean_params, std_params = load_parameters('intermittent_est_params.txt')
kde = setup_kde(normed_loc_params)

# Simulation parameters
N = 10000
N_iter = 2000
integration_factor = 1
g_tau = 1
tau_list = np.arange(1, 60)  # Adjust to ensure compatibility with data generation

# Perform iterations
og_params, lev_params_int, adj_r_square_int_lev, adj_r_square_int_int, est_params, est_params2 = perform_iterations(
    N_iter, N, integration_factor, g_tau, kde, std_params, mean_params, tau_list
)


# Function to write data to a text file
def write_to_file(filename, content):
    path = f"{filename}.txt"
    with open(path, "w") as file:
        for item in np.atleast_1d(content):
            file.write(f"{item}\n")
    return path


# Write summary results to files
file_paths = [
    write_to_file("summary_og_params", og_params),
    write_to_file("summary_levy_params", lev_params_int),
    write_to_file("summary_adj_r_square_levy", adj_r_square_int_lev),
    write_to_file("summary_adj_r_square_int", adj_r_square_int_int),
    write_to_file("summary_est_params", est_params),
    write_to_file("summary_est_params2", est_params2)
]

# Print the paths of the written files
print("File paths of the written summaries:", file_paths)


# Define a safe logarithm function
def safe_log(x, min_val=1e-10, max_val=1e30):
    return np.log(np.clip(x, min_val, max_val))

# Generate synthetic data for dx4_log based on tau_list
# Using the intermittent2 function and the procedure from your original script
g_v0, g_D, g_lambda_B, g_lambda_D = 5.0, 1.0, 0.05, 0.005
factor1, factor2, factor3, factor4 = 1, 1, 1, 1
x_loc, y_loc = intermittent2(N * integration_factor, g_tau / integration_factor, g_v0 * factor1, g_D * factor2,
                             g_lambda_B * factor3, g_lambda_D * factor4)

dx_list, dx2, dx4 = [], [], []
for i in tau_list:
    dx = np.diff(x_loc[::i * integration_factor])
    dx_list.append(dx)
    dx2.append(np.mean(dx**2))
    dx4.append(np.mean(dx**4))

dx2_log = np.log(dx2)
dx4_log = np.log(dx4)

# Calculate the difference
difference = np.array(dx4_log) - 2 * np.array(dx2_log)

# Define the model function for curve_fit
def model_func(tau, v0, D, lambdaB, lambdaD):
    return mom4_serg_log(tau, v0, D, lambdaB, lambdaD)

# Perform the curve fitting
initial_guess = [g_v0, g_D, g_lambda_B, g_lambda_D]
popt, pcov = curve_fit(model_func, tau_list, dx4_log, p0=initial_guess)

# Plot the synthetic data and the fitted model
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# First subplot
axs[0].plot(np.log10(tau_list), dx4_log, 'ks', alpha=1, label='synthetic data')
axs[0].plot(np.log10(tau_list), model_func(np.array(tau_list), *popt), label='Fitted model', c='red', linewidth=2)
axs[0].set_xlabel(r'$\log_{10}(\tau)$')
axs[0].set_ylabel(r'$\log_{10}(dx^4)$')
axs[0].legend()
axs[0].set_title('Synthetic Data and Fitted Model')

# Second subplot
axs[1].plot(np.log10(tau_list), difference, 'ks', alpha=1, label='synthetic data')
axs[1].plot(np.log10(tau_list),
            mom22_4_diff_serg_log(np.array(tau_list), g_v0 * factor1, g_D * factor2, g_lambda_B * factor3,
                                  g_lambda_D * factor4), label='Model Fit', c='red')
axs[1].set_xlabel(r'$\log_{10}(\tau)$')
axs[1].set_ylabel(r'$\log_{10} \|dx^4\| - 2\log_{10} \|dx^2\|$')
axs[1].legend()
axs[1].set_title('Model Fit Comparison')

plt.tight_layout()
plt.show()