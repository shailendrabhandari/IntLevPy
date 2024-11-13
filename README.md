[![Documentation Status](https://readthedocs.org/projects/intlevy-processes/badge/?version=latest)](https://intlevy-processes.readthedocs.io/en/latest/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen)](https://github.com/shailendrabhandari/IntLevy-Processes/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/intermittent-levy)](https://pypi.org/project/intermittent-levy/)
[![PyPI](https://img.shields.io/pypi/v/intermittent-levy)](https://pypi.org/project/intermittent-levy/)
[![Downloads](https://pepy.tech/badge/intermittent-levy)](https://pepy.tech/project/intermittent-levy)

# Intermittent Lévy Processes Package
## Overview

The **intermittent_levy** package provides tools for simulating and analyzing intermittent and Lévy processes. It includes functions for:

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
cd IntLevy-Processes
pip install -e .
```

*Note:* The `-e` flag installs the package in editable mode, allowing for modifications without reinstallation.

## Dependencies

Install all required dependencies using:

```bash
pip install -r requirements.txt
```


## Usage

### Example 1: Simulating and Analyzing an Intermittent Process

An example script is provided in the `examples/` directory as `run_intermittent_simulation.py`. This script demonstrates how to simulate an intermittent process and perform parameter estimation.

#### Running the Example Script

```bash
python examples/run_intermittent_simulation.py
```

#### Script Overview

The script performs the following steps:

1. **Simulates** an intermittent process using `intermittent3` with random parameters within specified ranges.
2. **Calculates** the empirical second and fourth moments of the displacements.
3. **Performs Classification** to separate different movement phases using statistical methods.
4. **Performs Optimization** to estimate the model parameters by fitting theoretical moments to empirical data.
5. **Stores** the optimized parameters and R-squared values for analysis.
6. **Plots** the empirical and fitted moments for visualization.

### Example 2: Simulating and Analyzing Multiple Lévy Flights

An example script is provided in the `examples/` directory as `run_levy_simulation.py`. This script demonstrates how to simulate multiple Lévy flights over multiple iterations and perform parameter estimation.

#### Running the Example Script

```bash
python examples/run_levy_simulation.py
```
![model](intermittent_levy/examples/results/IntLevy.jpg)

#### Script Overview

The script performs the following steps:

1. **Simulates** multiple Lévy flights with random parameters within specified ranges.
2. **Calculates** the empirical second and fourth moments of the displacements.
3. **Performs Optimization** to estimate the Lévy exponent `alpha` and mean velocity `v_mean` by fitting theoretical moments to empirical data.
4. **Stores** the optimized parameters and R-squared values for analysis.
5. **Plots** the empirical and fitted moments for the first few iterations.

### Results Directory

Both scripts save their results into a dedicated `results/` directory with subdirectories for each process type:

```
results/
├── intermittent/
│   ├── int_generated_params.txt
│   ├── int_generated_r_squared_int.txt
│   ├── int_generated_opt_list_int_params.txt
│   ├── int_generated_int_fit_list_mom2.txt
│   ├── int_generated_int_fit_list_mom4.txt
│   ├── int_generated_logdx2_list.txt
│   └── int_generated_logdx4_list.txt
└── levy/
    ├── lev_generated_params.txt
    ├── lev_generated_r_squared_lev.txt
    ├── lev_generated_opt_list_lev_params.txt
    ├── lev_generated_lev_fit_list_mom2.txt
    ├── lev_generated_lev_fit_list_mom4.txt
    ├── lev_generated_logdx2_list.txt
    └── lev_generated_logdx4_list.txt
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
│   ├── run_simulation.py          # Example script for intermittent processes
│   └── run_levy_simulation.py     # Example script for Lévy processes
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
  - `mom2_serg_log`: Calculate the logarithm of the second moment for intermittent processes.
  - `mom4_serg_log`: Calculate the logarithm of the fourth moment for intermittent processes.
  - `levy_moments_log`: Calculate the logarithm of the moments for Lévy flights.

- **intermittent_levy.optimization**
  - `to_optimize_mom4_and_2_serg_log`: Objective function for optimizing intermittent process parameters based on moments.
  - `to_optimize_levy`: Objective function for optimizing Lévy flight parameters.

- **intermittent_levy.classification**
  - `form_groups`: Classify data into groups based on thresholds.
  - `real_k_and_fisher`: Statistical analysis using Fisher's exact test.

- **intermittent_levy.utils**
  - `adjusted_r_square`: Calculate the adjusted R-squared value.
  - `r_square`: Calculate the R-squared value.
  - `adjusted_r_square_array`: Calculate adjusted R-squared for multiple fits.

- **intermittent_levy.examples**
  - `run_simulation.py`: Example script demonstrating how to simulate and analyze intermittent processes.
  - `run_levy_simulation.py`: Example script demonstrating how to simulate and analyze Lévy processes.

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

This project is licensed under the [MIT License](https://github.com/shailendrabhandari/IntLevy-Processes/blob/main/LICENSE).

## Contact

**Authors:**

- **Shailendra Bhandari**
  - **Email:** shailendra.bhandari@oslomet.no
- **Pedro Lencastre**
  - **Email:** pedroreg@oslomet.no

For any questions or inquiries, please feel free to reach out via email.