# IntLevy Process

## Overview
The IntLevy Process project is a specialized Python toolkit for simulating and analyzing intermittent processes and Levy flights. It excels in creating synthetic datasets that replicate the alternating intense and calm periods of intermittent processes, and the long-jump characteristic of Levy flights. This tool is essential for understanding stochastic dynamics in various fields, including physics, ecology, and finance. It's user-friendly and adaptable, offering both theoretical exploration and practical analysis capabilities for real-world data.


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
5. 5. Ensure the `functions.py` script in the `intermittentLevy` directory contains all the necessary custom functions. The script should define functions such as `intermittent2`, `levy_flight_2D_2`, `load_parameters`, `setup_kde`, `perform_iterations`, `mom4_log`, `to_optimize_mom4_log`, `to_optimize_mom22_4_diff_log`, `mom22_4_diff_log`, `moment4`, `to_optimize_mom4`, `mom22_4_diff`, `to_optimize_mom22_4_diff`, `mom2_model`, `mom4_model`, `form_groups`, `adjusted_r_square`, `powerl_fit`, and `perform_estimation`.
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

## Example Notebook

An example notebook is provided to demonstrate the usage of the IntLevy Process toolkit. This notebook guides you through the steps to distinguish between intermittent processes and Levy flights, showcasing how to simulate these processes, analyze their characteristics, and differentiate between them.

Access the notebook here: [Example Notebook](https://github.com/shailendrabhandari/IntLevy-Processes/blob/main/example.ipynb)

### Quick Start

1. **Access the Notebook**: Use the link to view and download the notebook from GitHub.
2. **Run the Notebook**: Open it in Jupyter Notebook or JupyterLab on your local setup, or use an online platform like Google Colab.
3. **Follow the Steps**: The notebook includes step-by-step instructions and explanations, guiding you through simulations and analyses.
4. **Interactive Learning**: Experiment with the code, alter parameters, and observe the differences between intermittent and Levy flight processes.

For a detailed understanding, please refer to the comments and documentation within the notebook.

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



## Moment Calculations
#### `moment4(t, v0, D, lambdaB, lambdaD)`
- Calculates the logarithm of the fourth moment of a stochastic process. Useful in analyzing the dynamics of processes over time with given initial velocity and diffusion coefficients.

#### `mom4_log(t, v0, D, lambdaB, lambdaD)`
- imilar to moment4, it calculates the logarithm of the fourth moment, specifically designed for processes with certain characteristics.

#### `mom2_model(tau, param1, param2)`
- Defines the model for the second moment, offering a way to represent stochastic processes based on input values and parameters.


#### `mom2_log(l_tau, v, D, l_lambdaB, l_lambdaD)`
- Defines a logarithm of a second moment model.


#### `to_optimize_mom4_serg(params)`
-  Optimizes the parameters for the `mom4_serg_log` function based on input data.

#### `to_optimize_mom22_4_diff(params)`
- Optimizes the parameters for the `mom22_4_diff_serg_log` function based on input data.

#### `mom22_4_diff_log(l_tau, v, D, l_lambdaB, l_lambdaD)`
- Computes a ration of fourth and the square of second moment in logarithmic form.
#### `powerl_fit(l_x, l_c, l_a)`:
Calculates the value of a power-law function for given parameters, useful in modeling phenomena that follow a power-law distribution.

#### `r_square(l_emp_points, l_emp_fit)`:
Computes the R-squared value, indicating the fit quality of a regression model to empirical data.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/shailendrabhandari/IntLevy-Processes/blob/main/LICENSE) file for details.

### MIT License Copyright (c)

Copyright (c) 2024 authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

%## Authors
