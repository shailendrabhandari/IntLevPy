.. _features:

Features
========

The IntLevPy package offers a variety of features for simulating, analyzing, and classifying complex stochastic processes. Key features include:

### Process Simulation

- Generate synthetic intermittent and Lévy trajectories.
- Simulate realistic intermittent processes using various statistical distributions.

**Functions:**

1. **`intermittent3`**  
   - **Description**: Simulates an intermittent 2D random walk with switching regimes (diffusion and ballistic motion).
   - **Parameters**:
     - `nt` (int): Number of time steps.
     - `dt` (float): Time interval between steps.
     - `mean_bal_sac` (float): Mean velocity for ballistic regime.
     - `diffusion` (float): Diffusion coefficient.
     - `rate21` (float): Transition rate from ballistic to diffusive.
     - `rate12` (float): Transition rate from diffusive to ballistic.
   - **Returns**: `(x, y)` tuple of arrays with the x and y coordinates over time.

2. **`wait_times`**  
   - **Description**: Generates waiting times following a power-law distribution for Lévy processes.
   - **Parameters**:
     - `taui` (float): Scaling factor for waiting times.
     - `lN` (int): Number of samples.
     - `lalpha` (float): Lévy distribution exponent (1 < lalpha < 3).
   - **Returns**: Array of generated waiting times.

3. **`levy_flight_2D_Simplified`**  
   - **Description**: Simulates a 2D Lévy flight using specified parameters.
   - **Parameters**:
     - `n_redirections` (int): Number of redirection steps.
     - `n_max` (int): Maximum number of measurement points.
     - `alpha` (float): Lévy distribution exponent (1 < alpha < 3).
     - `tmin` (float): Minimum flight time.
     - `v_mean` (float): Mean velocity.
     - `measuring_dt` (float): Time interval for measurements.
   - **Returns**: `(x_measured, y_measured)` tuple of arrays with the x and y positions at measurement times.

### Moments Calculation

- Calculate theoretical and empirical second and fourth moments.
- Allows for detailed statistical analysis of trajectories.

**Functions:**

1. **`mom4_serg_log`**  
   - **Description**: Calculates the theoretical logarithm of the 4th moment for intermittent search processes.
   - **Parameters**:
     - `t` (array-like): Time lags.
     - `v0` (float): Mean velocity.
     - `D` (float): Diffusion coefficient.
     - `lambdaB` (float): Transition rate to ballistic regime.
     - `lambdaD` (float): Transition rate to diffusive regime.
   - **Returns**: Array of the logarithm of the 4th moment values.

2. **`mom2_serg_log`**  
   - **Description**: Calculates the theoretical logarithm of the 2nd moment for intermittent search processes.
   - **Parameters**:
     - `tau` (array-like): Time lags.
     - `v` (float): Mean velocity.
     - `D` (float): Diffusion coefficient.
     - `lambdaB` (float): Transition rate to ballistic regime.
     - `lambdaD` (float): Transition rate to diffusive regime.
   - **Returns**: Array of the logarithm of the 2nd moment values.

### Optimization for Model Fitting

- Provides optimization routines to fit model parameters to empirical data.
- Enables accurate parameter estimation for intermittent and Lévy models.

**Functions:**

1. **`to_optimize_mom4_serg_log`**  
   - **Description**: Objective function to optimize parameters for the 4th moment in intermittent search processes.
   - **Parameters**:
     - `variables` (list): `[v0, D, lambdaB, lambdaD]` values for velocity, diffusion, and transition rates.
     - `tau_list` (array-like): List of time lags.
     - `logdx4` (array-like): Empirical 4th moment data (log scale).
   - **Returns**: Mean squared error between empirical and theoretical log 4th moments.

2. **`to_optimize_mom2_serg_log`**  
   - **Description**: Objective function to optimize parameters for the 2nd moment in intermittent search processes.
   - **Parameters**:
     - `variables` (list): `[v0, D, lambdaB, lambdaD]` values for velocity, diffusion, and transition rates.
     - `tau_list` (array-like): List of time lags.
     - `logdx2` (array-like): Empirical 2nd moment data (log scale).
   - **Returns**: Mean squared error between empirical and theoretical log 2nd moments.

### Classification

- Classify processes as intermittent or Lévy based on statistical properties.
- Uses threshold-based and Fisher’s exact test-based classifications.

**Functions:**

1. **`real_k_and_fisher`**  
   - **Description**: Calculates frequency matrix and detection metrics for a binary sequence, with Fisher’s exact test for correlation.
   - **Parameters**:
     - `binary_vector` (array-like): Sequence of binary values (0s and 1s).
   - **Returns**: Tuple containing the frequency matrix, detection values, and log of Fisher exact test values.

2. **`frequency_matrix_2D`**  
   - **Description**: Creates a 2D frequency matrix based on a threshold, optionally normalizing by row sums.
   - **Parameters**:
     - `d__ss` (array-like): Input data sequence.
     - `threshold` (float): Threshold to create binary vector from data.
     - `normalized` (bool): Whether to normalize by row sums.
   - **Returns**: 2x2 frequency matrix.

3. **`form_groups`**  
   - **Description**: Calculates detection and Fisher test metrics across multiple thresholds, with optional plotting.
   - **Parameters**:
     - `vector` (array-like): Input data sequence.
     - `threshold_array` (array-like): Array of thresholds to evaluate.
     - `graph` (bool): If `True`, plots detection metrics.
     - `x_label` (str): X-axis label for plot.
     - `title` (str): Title for plot.
     - `x_axis_format` (str): Format for x-axis labels.
   - **Returns**: Tuple of lists for detection metrics, Fisher metrics, and minimum values of detection and Fisher test results.


For more detailed usage examples, please see the Usage section or check out the example scripts in the `examples/` directory.
