
# Intermittent Lévy Processes Package

**Version:** 0.1

## Overview

The **intermittent_levy** package provides tools for simulating and classifying intermittent and Lévy processes. It includes functions for:

- **Process Simulation:** Generate synthetic intermittent and Lévy flight trajectories.
- **Statistical Moments:** Calculate theoretical and empirical moments of trajectories.
- **Optimization:** Fit model parameters to empirical data using optimization techniques.
- **Classification:** Distinguish between intermittent and Lévy processes using statistical methods.
- **Utilities:** Common functions for data analysis and processing.

This package is intended for researchers and practitioners working in statistical physics, complex systems, or any field where modeling and analysis of anomalous diffusion processes are relevant.

## Installation

Clone the repository and install the package using pip:

```bash
git clone https://github.com/shailendrabhandari/IntLevy-Processes.git
cd intermittent_levy
pip install -e .
```

*Note:* The `-e` flag installs the package in editable mode, allowing for modifications without reinstallation.

## Dependencies

The package requires the following Python libraries:

- **numpy**
- **scipy**
- **matplotlib**
- **pandas**
- **seaborn**
- **scikit-learn**
- **pomegranate**

These can be installed via pip:

```bash
pip install numpy scipy matplotlib pandas seaborn scikit-learn pomegranate
```

## Usage

### Example: Running a Simulation and Classification

An example script is provided in the `examples/` directory. Here's how you can simulate an intermittent process and perform parameter estimation:

```python
from intermittent_levy.processes import intermittent3
from intermittent_levy.moments import mom2_serg_log, mom4_serg_log
from intermittent_levy.optimization import to_optimize_mom4_and_2_serg_log
from intermittent_levy.classification import form_groups
from intermittent_levy.utils import adjusted_r_square
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
N = 300000
dt = 1
mean_bal_sac = 10
diffusion = 0.1
rate21 = 0.1
rate12 = 0.05

# Simulate intermittent process
x, y = intermittent3(N, dt, mean_bal_sac, diffusion, rate21, rate12)

# Compute displacements
dS = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

# Define time lags
tau_list = np.arange(1, 20)

# Calculate empirical moments
dx2 = [np.mean((x[::int(tau)] - x[:-int(tau):int(tau)])**2 +
               (y[::int(tau)] - y[:-int(tau):int(tau)])**2) for tau in tau_list]
dx4 = [np.mean(((x[::int(tau)] - x[:-int(tau):int(tau)])**2 +
                (y[::int(tau)] - y[:-int(tau):int(tau)])**2)**2) for tau in tau_list]

dx2_log = np.log(dx2)
dx4_log = np.log(dx4)

# Initial parameter estimates
initial_params = [mean_bal_sac, diffusion, rate21, rate12]

# Optimization bounds
bounds = [
    (initial_params[0]/10, initial_params[0]*10),
    (initial_params[1]/10, initial_params[1]*10),
    (initial_params[2]/10, initial_params[2]*10),
    (initial_params[3]/10, initial_params[3]*10),
]

# Perform optimization
result = optimize.dual_annealing(
    to_optimize_mom4_and_2_serg_log,
    bounds=bounds,
    args=(tau_list, dx2_log, dx4_log)
)

# Extract optimized parameters
optimized_params = result.x

# Calculate fitted moments
fitted_dx2_log = mom2_serg_log(tau_list, *optimized_params)
fitted_dx4_log = mom4_serg_log(tau_list, *optimized_params)

# Calculate adjusted R-squared
r2_dx2 = adjusted_r_square(dx2_log, fitted_dx2_log, degrees_freedom=4)
r2_dx4 = adjusted_r_square(dx4_log, fitted_dx4_log, degrees_freedom=4)

# Plot empirical and fitted moments
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

### Running the Example Script

To run the example script:

```bash
python examples/run_simulation.py
```

## Package Structure

The package is organized into the following modules and subpackages:

```
intermittent_levy/
├── __init__.py
├── processes.py          # Simulation of intermittent and Lévy processes
├── moments.py            # Calculation of statistical moments
├── optimization.py       # Optimization routines for parameter estimation
├── classification.py     # Classification methods for process analysis
├── utils.py              # Utility functions for data processing
├── examples/
│   └── run_simulation.py # Example script demonstrating usage
├── tests/                # Unit tests for the package
│   ├── __init__.py
│   ├── test_processes.py
│   ├── test_moments.py
│   ├── test_optimization.py
│   ├── test_classification.py
│   └── test_utils.py
├── setup.py              # Installation script
└── README.md             # Package documentation
```

### Module Descriptions

- **intermittent_levy.processes**
  - `intermittent3`: Simulate intermittent processes with specified parameters.
  - `levy_flight_2D_Simplified`: Simulate 2D Lévy flights.
  
- **intermittent_levy.moments**
  - `mom2_serg_log`: Calculate the logarithm of the second moment.
  - `mom4_serg_log`: Calculate the logarithm of the fourth moment.
  
- **intermittent_levy.optimization**
  - `to_optimize_mom4_and_2_serg_log`: Objective function for optimizing parameters based on moments.
  
- **intermittent_levy.classification**
  - `form_groups`: Classify data into groups based on thresholds.
  - `real_k_and_fisher`: Statistical analysis using Fisher's exact test.
  
- **intermittent_levy.utils**
  - `adjusted_r_square`: Calculate the adjusted R-squared value.
  - `r_square`: Calculate the R-squared value.
  
- **intermittent_levy.examples**
  - `run_simulation.py`: Example script demonstrating how to use the package.
  
- **intermittent_levy.tests**
  - Unit tests for each module to ensure code reliability and correctness.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository:** Create your own fork of the project.
2. **Create a Branch:** Create a new branch for your feature or bug fix.
3. **Commit Changes:** Make your changes and commit them with descriptive messages.
4. **Push to Branch:** Push your changes to your forked repository.
5. **Submit a Pull Request:** Submit a pull request to the `main` branch of the original repository.

Please ensure your code adheres to the project's coding standards and passes all existing tests. Adding new tests for your contributions is highly appreciated.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

**Author:** Shailendra Bhanari, Pedro Lencastre
**Email:** shailendra.bhandari@oslomet.no, pedroreg@oslomet.no

For any questions or inquiries, please feel free to reach out via email.


