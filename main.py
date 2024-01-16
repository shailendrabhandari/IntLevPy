import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Append the path to 'intermittentLevy' directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'intermittentLevy'))

# Import functions from 'momentum.py'
from momentum import intermittent2, levy_flight_2D_2, form_groups, load_parameters, setup_kde, perform_iterations

# Generate an intermittent trajectory
x, y = intermittent2(nt=1000, dt=1, mean_bal_sac=1, diffusion=1, rate21=0.5, rate12=0.5)

# Plot intermittent trajectory
plt.plot(x, y)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Intermittent Random Walk')
plt.show()

# Generate Lévy flight trajectory
x_measured, y_measured, _ = levy_flight_2D_2(n_redirections=100, n_max=1000, lalpha=1.5, tmin=1, measuring_dt=1)

# Plot Lévy flight trajectory
plt.plot(x_measured, y_measured)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Lévy Walk')
plt.show()

# Define a random vector
vector = np.random.rand(100)

# Define 'threshold_array' with appropriate thresholds
threshold_array = np.linspace(0, 1, 1000)

# Define the x-axis format
x_axis_format = "%.2f"

# Process the data and obtain detection values and minimum thresholds
detection, detectionfisher, min_k, min_fisher, _, _, _ = form_groups(vector, threshold_array, x_axis_format)

# Calculate log-k values
log_k = np.log(detection)

# Plot for 'detection' with threshold lines
plt.figure(figsize=(8, 4))
plt.plot(threshold_array, detection, label='k')
plt.plot(threshold_array, detectionfisher, label='log-fisher')
plt.xlabel("Threshold")
plt.title("Detection (k) and log-fisher")
plt.xticks(rotation=45)
plt.ylabel("Values")

# Add vertical lines for minimum thresholds
if min_k is not None:
    plt.axvline(x=min_k, color='r', linestyle='--', label='Min k threshold')
if min_fisher is not None:
    plt.axvline(x=min_fisher, color='g', linestyle='--', label='Min Fisher threshold')

plt.legend()
plt.tight_layout()
plt.show()

# Load parameters, set up KDE, and define other simulation parameters
normed_loc_params, mean_params, std_params = load_parameters('intermittent_est_params.txt')
kde = setup_kde(normed_loc_params)

N = 10000  # Number of points
N_iter = 200  # Number of iterations
integration_factor = 1
g_tau = 1
tau_list = np.arange(1, 30)

# Perform iterations
og_params, lev_params_int, adj_r_square_int_lev, adj_r_square_int_int, est_params, est_params2 = perform_iterations(
    N_iter, N, integration_factor, g_tau, kde, std_params, mean_params, tau_list
)

# Plotting the results of the simulations
plt.figure(figsize=(10, 6))
plt.plot(adj_r_square_int_lev, label='Adjusted R-square Levy', marker='o')
plt.plot(adj_r_square_int_int, label='Adjusted R-square Int', marker='x')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Adjusted R-square', fontsize=14)
plt.title('Adjusted R-square over Iterations', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()