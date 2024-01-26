# IntLevy Process


## Overview
This project simulates and analyzes intermittent processes and Levy flight behaviors in 2D. It incorporates custom functions for generating synthetic data, performing curve fitting, and analyzing data patterns. The script integrates data generation, statistical modeling, and visualization functionalities.

## Installation

### Prerequisites
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- scikit-learn


### Installing Dependencies
Install the required packages using the following command:
```bash
pip install numpy scipy matplotlib scikit-learn
```
### Setting Up the Project


1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Create a new directory named `intermittentLevy` inside the project directory.
4. Place your custom scripts or modules in the `intermittentLevy` directory.
5. Ensure the `functions.py` script in the `intermittentLevy` directory contains all the necessary custom functions. The script should define functions like `intermittent2`, `levy_flight_2D_2`, `load_parameters`, `setup_kde`, `perform_iterations`, `mom4_serg_log`, `to_optimize_mom4_serg_log`, `to_optimize_mom22_4_diff_serg_log`, and `mom22_4_diff_serg_log`.
6. The script expects a file named `intermittent_est_params.txt` for loading parameters. Ensure this file is placed in a location accessible by the script.

## Usage
Follow these steps to use the script:
1. Navigate to the directory containing the script.
2. Run the script with Python:
   ```bash
   python main.py

### Operations performed by the script:
. Load parameters and set up Kernel Density Estimation (KDE).
. Perform iterations to simulate intermittent Levy flights.
. Generate synthetic data and perform curve fitting.
. Create and save plots for data visualization.
. Write summary results to text files.
### Output
- Plots visualizing the synthetic data and model fits.
- Text files containing summary results:
  - `summary_og_params.txt`
  - `summary_levy_params.txt`
  - `summary_adj_r_square_levy.txt`
  - `summary_adj_r_square_int.txt`
  - `summary_est_params.txt`
  - `summary_est_params2.txt`

## Customization
You can customize the simulation by modifying the parameters in the script, such as:
- `N`: Number of points for simulation.
- `N_iter`: Number of iterations.
- `tau_list`: List of tau values for analysis.
- `g_v0`, `g_D`, `g_lambda_B`, `g_lambda_D`: Parameters for the `intermittent2` function.
- 
## Code Functions Overview

The functions are defined for calculating statistical moments and performing optimization. Below is an overview of the functions defined in the code:

### moment_functions

This function calculates various statistical moments based on the provided parameters.

### mom4_serg_log

`mom4_serg_log` is responsible for computing the 4th and 2nd moments in logarithmic space.

### to_optimize_mom4_serg_log

`to_optimize_mom4_serg_log` is used for the optimization of the 4th moment in logarithmic space.

### optimize_mom4_serg_log

`optimize_mom4_serg_log` is the function responsible for optimizing the 4th moment.

### numerical_stability

`numerical_stability` ensures numerical stability by adding a small epsilon value.
These functions are defined in the code and perform specific tasks related to moment calculations and optimization.
Feel free to explore the code further to understand the details of each function and how they are utilized.


## License
MIT License Copyright (c)

## Authors
- [List all]
