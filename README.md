# IntLevy Process

## Overview
The IntLevy Process project is dedicated to the study and simulation of two significant phenomena in stochastic processes: intermittent processes and Levy flights. Intermittent processes are characterized by alternating periods of intense activity and relative calm, resembling many real-world systems such as financial markets or geological events. Levy flights, on the other hand, represent a random walk where the step lengths have a heavy-tailed probability distribution, featuring occasional long jumps.

Our project offers a comprehensive toolkit for simulating these complex behaviors in a 2D space. The provided functions enable users to generate synthetic datasets that mimic the intricate patterns observed in intermittent and Levy flight processes. This capability is crucial for researchers and practitioners who need to understand and predict the behavior of systems exhibiting such stochastic dynamics.

Additionally, the project supports the analysis of real data, allowing users to apply our models to their datasets. This integration of simulation and real-world application makes our toolkit not only a valuable educational resource but also a practical tool for data analysis in fields like physics, ecology, finance, and more.

The script is designed with flexibility in mind, offering various customization options to fit different research needs and data characteristics. Whether it's exploring theoretical aspects of stochastic processes or analyzing real-world data, our project provides the necessary functions and a user-friendly approach to studying intermittent and Levy flight behaviors.


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



## Stochastic Processes Simulation

### `intermittent2(nt, dt, mean_bal_sac, diffusion, rate21, rate12)`
- **Description**: Simulates an intermittent stochastic process in 2D.
- **Parameters**:
  - `nt`: Number of time points (int).
  - `dt`: Time step size (float).
  - `mean_bal_sac`: Mean balance of sacs (float).
  - `diffusion`: Diffusion coefficient (float).
  - `rate21`, `rate12`: Transition rates between states (float).
- **Returns**: Simulated x and y coordinates (arrays).

### `levy_flight_2D_2(n_redirections, n_max, lalpha, tmin, measuring_dt)`
- **Description**: Simulates a 2D Levy flight with specified parameters.
- **Parameters**:
  - `n_redirections`: Number of redirections in the flight (int).
  - `n_max`: Maximum number of steps (int).
  - `lalpha`: Levy flight parameter alpha (float).
  - `tmin`: Minimum time between redirections (float).
  - `measuring_dt`: Measurement time step (float).
- **Returns**: x and y coordinates of the Levy flight, time of redirections (arrays).

## Data Analysis

### `frequency_matrix_2D(d__ss, threshold, normalized)`
- **Description**: Creates a frequency matrix from a 2D dataset.
- **Parameters**:
  - `d__ss`: Input dataset (array).
  - `threshold`: Threshold for frequency calculation (float).
  - `normalized`: Whether to normalize the matrix (boolean).
- **Returns**: Frequency matrix (2D array).

## Moment Calculations

### `mom4_serg_log(t, v0, D, lambdaB, lambdaD)`
- **Description**: Calculates the logarithm of the fourth moment of a stochastic process.
- **Parameters**:
  - `t`: Time parameter (float).
  - `v0`: Initial velocity (float).
  - `D`: Diffusion coefficient (float).
  - `lambdaB`: Rate parameter B (float).
  - `lambdaD`: Rate parameter D (float).
- **Returns**: Logarithm of the fourth moment (float).

### `mom2_serg_log(l_tau, v, D, l_lambdaB, l_lambdaD)`
- **Description**: Calculates the logarithm of the second moment of a stochastic process.
- **Parameters**:
  - `l_tau`: Time parameter (float).
  - `v`: Velocity parameter (float).
  - `D`: Diffusion coefficient (float).
  - `l_lambdaB`: Rate parameter B (float).
  - `l_lambdaD`: Rate parameter D (float).
- **Returns**: Logarithm of the second moment (float).

## Optimization Functions

### `to_optimize_mom4_serg_log(params)`
- **Description**: Optimizes the parameters for the `mom4_serg_log` function based on input data.
- **Parameters**:
  - `params`: A tuple containing the parameters to be optimized.
- **Returns**: Sum of absolute differences between empirical data and model (float).

### `to_optimize_mom22_4_diff_serg_log(params)`
- **Description**: Optimizes the parameters for the `mom22_4_diff_serg_log` function based on input data.
- **Parameters**:
  - `params`: A tuple containing the parameters to be optimized.
- **Returns**: Sum of absolute differences between empirical data and model (float).

### `mom22_4_diff_serg_log(l_tau, v, D, l_lambdaB, l_lambdaD)`
- **Description**: Calculates a modified version of the second moment in logarithmic form.
- **Parameters**: Same as in `mom2_serg_log`.
- **Returns**: Modified logarithmic second moment (float).


## License
MIT License Copyright (c)

## Authors
- [List all]
