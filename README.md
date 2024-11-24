[![Documentation Status](https://readthedocs.org/projects/intlevpy/badge/?version=latest)](https://intlevpy.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](https://github.com/shailendrabhandari/IntLevPy/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/intlevpy)](https://pypi.org/project/IntLevPy/)
[![PyPI](https://img.shields.io/pypi/v/intlevpy)](https://pypi.org/project/IntLevPy/)
[![Downloads](https://pepy.tech/badge/intlevpy)](https://pepy.tech/project/intlevpy)
[![GitHub Watchers](https://img.shields.io/github/watchers/shailendrabhandari/IntLevPy?style=social)](https://github.com/shailendrabhandari/IntLevPy/watchers)
[![GitHub Stars](https://img.shields.io/github/stars/shailendrabhandari/IntLevPy?style=social)](https://github.com/shailendrabhandari/IntLevPy/stargazers)


# IntLevPy: Intermittent Lévy Processes Package

**IntLevPy** is a Python package for simulating and analyzing intermittent and Lévy processes. It provides tools for process simulation, moments calculation, optimization, and classification methods, making it ideal for researchers and practitioners in fields like statistical physics and complex systems.

## Overview

- **Process Simulation:** Generate synthetic intermittent and Lévy flight trajectories.
- **Moments Calculation:** Compute theoretical and empirical moments of trajectories.
- **Optimization:** Fit model parameters to empirical data using optimization techniques.
- **Classification:** Differentiate between intermittent and Lévy processes using statistical methods.

For detailed documentation, visit the [IntLevPy Documentation](https://intlevpy.readthedocs.io/en/latest/).

## Model

![Intermittent Lévy Process Model](https://raw.githubusercontent.com/shailendrabhandari/IntLevPy/main/intermittent_levy/examples/results/model.jpg)

*Figure: Schematic representation of the intermittent Lévy process model.*

## Installation

Install **IntLevPy** directly from PyPI:

```bash
pip install IntLevPy
```

## Usage

### Simulating and Analyzing an Intermittent Process

```python
import numpy as np
import matplotlib.pyplot as plt
from intermittent_levy.processes import intermittent3
from intermittent_levy.moments import mom2_serg_log, mom4_serg_log
from intermittent_levy.optimization import to_optimize_mom4_and_2_serg_log
from intermittent_levy.utils import adjusted_r_square
from scipy import optimize

# Simulation parameters
N = 300000                   # Number of steps
integration_factor = 3
tau = 1.0 / integration_factor
v0 = 10.0                    # Mean velocity
D = 0.5                      # Diffusion coefficient
lambda_B = 0.1               # Transition rate from ballistic to diffusive
lambda_D = 0.05              # Transition rate from diffusive to ballistic

# Simulate intermittent process
x_traj, y_traj = intermittent3(
    N * integration_factor,
    tau,
    v0,
    D,
    lambda_B,
    lambda_D
)

# Time intervals for analysis
tau_list = np.arange(1, 100, 5)

# Calculate empirical moments
dx2 = []
dx4 = []
for tau_i in tau_list:
    dx = x_traj[int(tau_i):] - x_traj[:-int(tau_i)]
    dy = y_traj[int(tau_i):] - y_traj[:-int(tau_i)]
    displacement = dx**2 + dy**2
    dx2.append(np.mean(displacement))
    dx4.append(np.mean(displacement**2))

dx2_log = np.log(dx2)
dx4_log = np.log(dx4)

# Parameter estimation using optimization
def objective_function(params, tau_list, dx2_log, dx4_log):
    v0_opt, D_opt, lambda_B_opt, lambda_D_opt = params
    model_dx2_log = mom2_serg_log(tau_list, v0_opt, D_opt, lambda_B_opt, lambda_D_opt)
    model_dx4_log = mom4_serg_log(tau_list, v0_opt, D_opt, lambda_B_opt, lambda_D_opt)
    error = np.sum((dx2_log - model_dx2_log)**2 + (dx4_log - model_dx4_log)**2)
    return error

initial_guess = [v0, D, lambda_B, lambda_D]
bounds = [
    (v0 * 0.5, v0 * 1.5),
    (D * 0.5, D * 1.5),
    (lambda_B * 0.5, lambda_B * 1.5),
    (lambda_D * 0.5, lambda_D * 1.5)
]

result = optimize.minimize(
    objective_function,
    initial_guess,
    args=(tau_list, dx2_log, dx4_log),
    bounds=bounds
)

estimated_params = result.x
print("Estimated Parameters:")
print(f"v0 = {estimated_params[0]:.4f}")
print(f"D = {estimated_params[1]:.4f}")
print(f"lambda_B = {estimated_params[2]:.4f}")
print(f"lambda_D = {estimated_params[3]:.4f}")

# Plot empirical and fitted moments
fitted_dx2_log = mom2_serg_log(tau_list, *estimated_params)
fitted_dx4_log = mom4_serg_log(tau_list, *estimated_params)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(np.log(tau_list), dx2_log, 'o', label='Empirical log M2')
plt.plot(np.log(tau_list), fitted_dx2_log, '-', label='Fitted log M2')
plt.xlabel('log(tau)')
plt.ylabel('log(M2)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.log(tau_list), dx4_log, 'o', label='Empirical log M4')
plt.plot(np.log(tau_list), fitted_dx4_log, '-', label='Fitted log M4')
plt.xlabel('log(tau)')
plt.ylabel('log(M4)')
plt.legend()

plt.tight_layout()
plt.show()
```

### Simulating and Analyzing a Lévy Flight Process

```python
import numpy as np
import matplotlib.pyplot as plt
from intermittent_levy.processes import levy_flight_2D_Simplified
from intermittent_levy.moments import levy_moments_log
from intermittent_levy.optimization import to_optimize_levy
from intermittent_levy.utils import adjusted_r_square
from scipy import optimize

# Simulation parameters
N = 300000               # Number of steps
alpha = 1.7              # Lévy exponent (1 < alpha < 2)
tmin = 0.01              # Minimum time between steps
v_mean = 10.0            # Mean velocity
dt = 1.0                 # Time increment

# Simulate Lévy flight
x_traj, y_traj = levy_flight_2D_Simplified(N, alpha, tmin, v_mean, dt)

# Time intervals for analysis
tau_list = np.arange(1, 100, 5)

# Calculate empirical moments
dx2 = []
dx4 = []
for tau_i in tau_list:
    dx = x_traj[int(tau_i):] - x_traj[:-int(tau_i)]
    dy = y_traj[int(tau_i):] - y_traj[:-int(tau_i)]
    displacement = dx**2 + dy**2
    dx2.append(np.mean(displacement))
    dx4.append(np.mean(displacement**2))

dx2_log = np.log(dx2)
dx4_log = np.log(dx4)

# Parameter estimation using optimization
def objective_function(params, tau_list, dx2_log, dx4_log, tmin):
    alpha_opt, v_mean_opt = params
    model_dx2_log = levy_moments_log(2, alpha_opt, v_mean_opt, tau_list, tmin)
    model_dx4_log = levy_moments_log(4, alpha_opt, v_mean_opt, tau_list, tmin)
    error = np.sum((dx2_log - model_dx2_log)**2 + (dx4_log - model_dx4_log)**2)
    return error

initial_guess = [alpha, v_mean]
bounds = [(1.1, 2.0), (v_mean * 0.5, v_mean * 1.5)]

result = optimize.minimize(
    objective_function,
    initial_guess,
    args=(tau_list, dx2_log, dx4_log, tmin),
    bounds=bounds
)

estimated_alpha, estimated_v_mean = result.x
print("Estimated Parameters:")
print(f"alpha = {estimated_alpha:.4f}")
print(f"v_mean = {estimated_v_mean:.4f}")

# Plot empirical and fitted moments
fitted_dx2_log = levy_moments_log(2, estimated_alpha, estimated_v_mean, tau_list, tmin)
fitted_dx4_log = levy_moments_log(4, estimated_alpha, estimated_v_mean, tau_list, tmin)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(np.log(tau_list), dx2_log, 'o', label='Empirical log M2')
plt.plot(np.log(tau_list), fitted_dx2_log, '-', label='Fitted log M2')
plt.xlabel('log(tau)')
plt.ylabel('log(M2)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.log(tau_list), dx4_log, 'o', label='Empirical log M4')
plt.plot(np.log(tau_list), fitted_dx4_log, '-', label='Fitted log M4')
plt.xlabel('log(tau)')
plt.ylabel('log(M4)')
plt.legend()

plt.tight_layout()
plt.show()
```

### Applying to Real-Life Data

**IntLevPy** can also be applied to real-life datasets, such as eye-tracking data or financial time series, for classification and analysis using the provided statistical methods.

## Contact

For questions or inquiries, please contact:

- **Shailendra Bhandari**
  - Email: shailendra.bhandari@oslomet.no
- **Pedro Lencastre**
  - Email: pedroreg@oslomet.no

---

This package is licensed under the [MIT License](https://github.com/shailendrabhandari/IntLevPy/blob/main/LICENSE).

---

**GitHub Repository:** [IntLevPy on GitHub](https://github.com/shailendrabhandari/IntLevPy)

**PyPI Package:** [IntLevPy on PyPI](https://pypi.org/project/IntLevPy/)

**Documentation:** [IntLevPy Documentation](https://intlevpy.readthedocs.io/en/latest/)

**For a detailed list of contributors, visit:** [Contributors Page](https://intlevpy.readthedocs.io/en/latest/authors.html#contributors).