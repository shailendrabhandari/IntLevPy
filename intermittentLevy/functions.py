# Necessary library imports for the momentum functions
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.neighbors import KernelDensity
import functools
import math
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.linear_model import LinearRegression
import warnings
import scipy.optimize

# Configuration for warning messages
warnings.filterwarnings("error")

def moment4(t, v0, D, lambdaB, lambdaD):
    """
        Calculate the logarithm of the fourth moment of a stochastic process.

        Parameters:
        t (float): Time parameter.
        v0 (float): Initial velocity.
        D (float): Diffusion coefficient.
        lambdaB (float): Rate parameter B.
        lambdaD (float): Rate parameter D.

        Returns:
        float: Logarithm of the fourth moment.
        """
    C1 = 3 * lambdaB ** (-2) * (lambdaB + lambdaD) ** (-2) * (2 * D * lambdaB ** 2 + v0 ** 2 * lambdaD) ** 2
    C2 = 3 * lambdaB ** (-3) * lambdaD * (lambdaB + lambdaD) ** (-3) * (
            8 * D ** 2 * lambdaB ** 4 - 8 * D * v0 ** 2 * lambdaB ** 2 * (2 * lambdaB + lambdaD) + v0 ** 4 * (
            3 * lambdaB ** 2 - 2 * lambdaB * lambdaD - 3 * lambdaD ** 2))
    C3 = 3 * lambdaB ** (-4) * lambdaD * (lambdaB + lambdaD) ** (-4) * (
            -8 * D ** 2 * lambdaB ** 5 + 8 * D * v0 ** 2 * lambdaB ** 2 * (
            3 * lambdaB ** 2 + 3 * lambdaB * lambdaD + lambdaD ** 2) + v0 ** 4 * (
                    -9 * lambdaB ** 3 - 7 * lambdaB ** 2 * lambdaD + 3 * lambdaB * lambdaD ** 2 + 3 * lambdaD ** 3))
    C4 = 3 / 2 * v0 ** 4 * lambdaB ** (-2) * lambdaD * (lambdaB + lambdaD) ** (-1)
    C5 = 6 * v0 ** 4 * lambdaB ** (-2) * (lambdaB + lambdaD) ** (-1)
    C6 = 0.0
    C7 = -3 * v0 ** 2 / (lambdaB ** (4) * lambdaD * (lambdaB + lambdaD)) * (8 * D * lambdaB ** 2 * lambdaD + v0 ** 2 * (
            2 * lambdaB ** 2 - 6 * lambdaB * lambdaD + 3 * lambdaD ** 2))
    C8 = 6 * lambdaB / (lambdaD * (lambdaB + lambdaD) ** (4)) * (v0 ** 2 + 2 * D * lambdaD) ** 2
    expr = (C1 * t ** 2 + C2 * t + C3 + 
            C4 * t ** 2 * np.exp(-lambdaB * t) + 
            C5 * t * np.exp(-lambdaB * t) + 
            C6 * t * np.exp(-(lambdaB + lambdaD) * t) + 
            C7 * np.exp(-lambdaB * t) + 
            C8 * np.exp(-(lambdaB + lambdaD) * t))

    return (expr)
    

def to_optimize_mom4(params):
    try:
        model_result = momment4(tau_list, *params)
        return np.sum(np.abs(dx4 - safe_log(model_result)))
    except Exception as e:
        return 1e10

def mom22_4_diff(l_tau, v, D, l_lambdaB, l_lambdaD):
    """
        Calculate a modified version of the second moment in logarithmic form.

        Parameters:
        l_tau, v, D, l_lambdaB, l_lambdaD: Same as in mom2_serg_log.

        Returns:
        float: Modified logarithmic second moment.
        """
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB/(l_beta)    
    C1_2 = 2*((v/l_beta)**2) * (l_alpha - 1)/(l_alpha**2)
    C2_2 = 2*((1-l_alpha)/l_alpha) * ((v**2)/l_beta) + 4*D*l_alpha
    
    expr2  = 2*( (C1_2*(1-np.exp(-l_alpha*l_beta*l_tau)) + C2_2*l_tau )/2)
    
    lambdaB = l_lambdaB
    lambdaD = l_lambdaD
    v0 = v
    t = l_tau
    
    C1 = 3 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -2 ) * ( 2 * D * lambdaB**2 + v0**2 * lambdaD )**2
    C2 = 3 * lambdaB**( -3 ) * lambdaD * ( lambdaB + lambdaD )**( -3 ) * ( 8 * D**2 * lambdaB**4 - 8 * D * v0**2 * lambdaB**2 * ( 2 * lambdaB + lambdaD ) + v0**4 * ( 3 * lambdaB**2 - 2 * lambdaB * lambdaD - 3 * lambdaD**2 ) )
    C3 = 3* lambdaB**( -4 ) * lambdaD * ( lambdaB + lambdaD )**( -4 ) * ( -8 * D**2 * lambdaB**5 + 8 * D * v0**2 * lambdaB**2 * ( 3 * lambdaB**2 + 3 * lambdaB * lambdaD + lambdaD**2 ) + v0**4 * ( -9 * lambdaB**3 - 7 * lambdaB**2 * lambdaD + 3 * lambdaB * lambdaD**2 + 3 * lambdaD**3 ) )
    C4 = 3/2 * v0**4 * lambdaB**( -2 ) * lambdaD * ( lambdaB + lambdaD )**( -1 )
    C5 = 6 * v0**4 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -1 )
    C6 = 0.0
    C7 = -3 * v0**2 / (lambdaB**( 4 ) * lambdaD * ( lambdaB + lambdaD )) * ( 8 * D * lambdaB**2  * lambdaD  + v0**2 * ( 2 * lambdaB**2 - 6 * lambdaB * lambdaD + 3 * lambdaD**2 ) )
    C8 = 6 * lambdaB * lambdaD**( -1 ) * ( lambdaB + lambdaD )**( -4 ) * ( v0**2 + 2 * D * lambdaD )**2
    expr = (C1 * t**2 + C2 * t + C3 + C4 * t**2 * np.e**( -lambdaB * t ) + C5 * t * np.e**( -lambdaB * t ) + C6 * t * np.e**( -( lambdaB + lambdaD ) * t ) + C7 * np.e**( -lambdaB * t ) + C8 * np.e**( -( lambdaB + lambdaD ) * t ))

    return(expr - expr2)
def to_optimize_mom22_4_diff(params):
    try:
        model_result = mom22_4_diff(tau_list, *params)
        return np.sum(np.abs(difference - (model_result)))
    except Exception as e:
        #print("Error encountered:", e)
        return 1e10


def intermittent2(nt, dt, mean_bal_sac, diffusion, rate21, rate12):
    diffusion = np.sqrt(2 * diffusion)
    P1 = rate21 / (rate12 + rate21)
    if np.random.random() < P1:
        regime = 1
        waitt = -math.log(1.0 - np.random.uniform(0.0, 1.0, None)) / rate12
    else:
        regime = 2
        angle2 = np.random.randint(2) * math.pi
        waitt = -math.log(1.0 - np.random.uniform(0.0, 1.0, None)) / rate21

    dts = math.sqrt(dt)
    x = np.zeros(nt)
    y = np.zeros(nt)
    time_since_last_jump = 0
    for i in range(1, nt):
        angle = random.random() * 2 * math.pi
        time_since_last_jump += dt
        if regime == 1:
            # diffu = diffusion*np.random.normal(0,1)*dts
            dx = np.random.normal(0, diffusion) * dts
            dy = np.random.normal(0, diffusion) * dts
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1] + dy
            if time_since_last_jump > waitt:
                waitt = -math.log(1.0 - np.random.uniform(0.0, 1.0, None)) / rate21
                regime = 2
                angle2 = angle
                time_since_last_jump = 0
        if regime == 2:
            angle3 = angle2
            bal = mean_bal_sac * dt
            dx = bal * math.cos(angle3)
            dy = bal * math.sin(angle3)
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1] + dy

            if time_since_last_jump > waitt:
                waitt = -math.log(1.0 - np.random.uniform(0.0, 1.0, None)) / rate12
                time_since_last_jump = 0
                regime = 1

    return (x, y)


def levy_flight_2D_2(n_redirections, n_max, lalpha, tmin, measuring_dt):
    if lalpha <= 1:
        print("alpha should be larger than 1")
        return ("alpha should be larger than 1")

    t_redirection = tmin * (np.ones(n_redirections) - np.random.rand(n_redirections)) ** (1.0 / (-lalpha + 1))
    cum_t_redirection = np.cumsum(t_redirection)

    angle = np.random.rand(len(t_redirection)) * 2 * math.pi
    x_increments = t_redirection * np.cos(angle)
    y_increments = t_redirection * np.sin(angle)
    l_x_list = np.cumsum(x_increments)
    l_y_list = np.cumsum(y_increments)

    if n_max * measuring_dt < cum_t_redirection[-1]:
        x_measured = np.interp(np.arange(0, n_max * measuring_dt, measuring_dt), np.cumsum(t_redirection), l_x_list)
        y_measured = np.interp(np.arange(0, n_max * measuring_dt, measuring_dt), np.cumsum(t_redirection), l_y_list)
    else:
        n_max = int(cum_t_redirection[-1] / measuring_dt)
        # print("me<zasuring time greater than simulated time. n_max becomes " + str(n_max))
        x_measured = np.interp(np.arange(0, n_max * measuring_dt, measuring_dt), np.cumsum(t_redirection), l_x_list)
        y_measured = np.interp(np.arange(0, n_max * measuring_dt, measuring_dt), np.cumsum(t_redirection), l_y_list)
        # print("measuring time greater than simulated time.")
    return x_measured, y_measured, t_redirection



def form_groups(vector, threshold_array, x_axis_format):
    detectionfisher = []
    detection = []
    min_k = None
    min_fisher = None

    if len(threshold_array) > 0:  # Check if the array is not empty
        for i in threshold_array:
            matrix = frequency_matrix_2D(vector, i, False)
            fisher_result = scipy.stats.fisher_exact(matrix)
            detectionfisher.append(np.log(fisher_result[1]))
            p = matrix.sum(axis=1)[0] / matrix.sum()
            detection.append(1 if p in [1, 0] else (matrix[1][0]) / (matrix.sum() * p * (1 - p)))

        minim = min(vector)
        diff = max(vector) - minim

        min_k = np.argmin(detection) * diff / len(threshold_array) + minim
        min_fisher = np.argmin(detectionfisher) * diff / len(threshold_array) + minim

    return detection, detectionfisher, min_k, min_fisher, threshold_array, minim, diff

def adjusted_r_square(l_emp_points, l_emp_fit, degrees_freedom):
    """
    Calculate the adjusted R-squared, which is a modified version of R-squared that has been
    adjusted for the number of predictors in the model. It provides a more accurate measure in
    the context of multiple regression.

    Parameters:
    l_emp_points (list or array): The empirical data points (observed values).
    l_emp_fit (list or array): The values predicted by the regression model.
    degrees_freedom (int): The degrees of freedom in the model, typically the number of predictors.

    Returns:
    float: The adjusted R-squared value, which accounts for the number of predictors.
    """
    n = len(l_emp_points)
    l_num = np.mean((np.array(l_emp_points) - np.array(l_emp_fit)) ** 2)
    l_den = np.std(np.array(l_emp_points)) ** 2
    rsqu = 1 - l_num / l_den
    return 1 - (1 - rsqu) * ((n - 1) / (n - degrees_freedom))
    
def load_parameters(file_name):
    loc_params = np.swapaxes(np.loadtxt(file_name), 0, 1)
    mean_params = np.mean(np.log(loc_params), axis=1)
    std_params = np.std(np.log(loc_params), axis=1)
    return (np.swapaxes(np.log(loc_params), 0, 1) - mean_params) / std_params, mean_params, std_params

def mom2_model(tau, param1, param2):
    """
    Define the model for the second moment.

    Parameters:
    tau (float or array): The input value(s) for the model.
    param1 (float): The first parameter for the model.
    param2 (float): The second parameter for the model.

    Returns:
    float or array: The result of the model for the given input value(s).
    """
    return param1 * tau**param2


def mom4_model(tau, param1, param2):
    return param1 * tau**param2
    
    
    
def powerl_fit(tau, l_k, l_a):
    """
    Calculate the value of a power-law function for a given set of parameters. This type of function
    is commonly used in various scientific fields to model relationships where one quantity varies
    as a power of another.

    Parameters:
    l_tau (float or array): The input value(s) for the power-law function.
    l_k (float): The coefficient (scale factor) of the power-law function.
    l_a (float): The exponent of the power-law function.

    Returns:
    float or array: The result of the power-law function for the given input value(s).
    """
    return l_k * np.power(2, tau * l_a)

def setup_kde(normed_loc_params, bandwidth=0.2):
    return KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(normed_loc_params)



def perform_iterations(N_iter, N, integration_factor, g_tau, kde, std_params, mean_params, tau_list):
    og_params, lev_params_int, adj_r_square_int_lev, adj_r_square_int_int = [], [], [], []
    est_params, est_params2 = [], []

    for itera in range(N_iter):
        new_data = kde.sample()
        [[g_v0, g_D, g_lambda_B, g_lambda_D]] = np.exp(new_data * std_params + mean_params)
        og_params.append([g_v0, g_D, g_lambda_B, g_lambda_D])

        x_loc, y_loc = intermittent2(N, g_tau, g_v0, g_D, g_lambda_B, g_lambda_D)
        lev_params, int_params, adj_r_square_lev, adj_r_square_int = perform_estimation(x_loc, y_loc)
        lev_params_int.append(lev_params)
        adj_r_square_int_lev.append(adj_r_square_lev)
        adj_r_square_int_int.append(adj_r_square_int)

        # Logic for calculating est_params and est_params2
       
        #est_param = calculate_est_param(lev_params, int_params, x_loc, y_loc)  # Replace with your function
        #est_params.append(est_param)
        #est_param2 = calculate_est_param2(lev_params, int_params, x_loc, y_loc)  # Replace with your function
        #est_params2.append(est_param2)

    return og_params, lev_params_int, adj_r_square_int_lev, adj_r_square_int_int, est_params, est_params2


def perform_estimation(x_loc, y_loc):
    # Replace this with the actual estimation logic
    return [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()], \
           [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()], \
           np.random.rand(), np.random.rand()


def weighted_error_fourth_second(ltau, lv, ld, llambdab, llambdad, lemp_fourth, lemp_second):
    LTheo4 = np.array(mom4_serg_log(ltau, lv, ld, llambdab, llambdad))
    LTheo2 = np.array(mom4_serg_log(ltau, lv, ld, llambdab, llambdad))

def r_square(l_emp_points, l_emp_fit):
    """
    Calculate the coefficient of determination, R-squared, which is a statistical measure of how well
    the regression predictions approximate the real data points.

    Parameters:
    l_emp_points (list or array): The empirical data points (observed values).
    l_emp_fit (list or array): The values predicted by the regression model.

    Returns:
    float: The R-squared value, ranging from 0 to 1, where 1 indicates a perfect fit.
    """
    l_num = np.mean((np.array(l_emp_points) - np.array(l_emp_fit)) ** 2)
    l_den = np.std(np.array(l_emp_points)) ** 2
    return 1 - l_num / l_den



def optimized_funcPairs(larr):
    """
    Creates pairs of elements from a given array without repeating the same element.
    """
    n = len(larr)
    lpairs = []
    for i in range(n):
        for j in range(i + 1, n):
            lpairs.append([larr[i], larr[j]])
    return lpairs

def optimized_parse_trials(lparams_list, threshold_ratio):
    """
    Processes a list of parameters, filtering out those that do not meet a certain threshold ratio.
    """
    threshold_log = np.log(threshold_ratio)
    log_lparams = np.log(lparams_list)
    indices_del = set()

    for row in log_lparams:
        pairs = optimized_funcPairs(row)
        new_M = np.abs(np.diff(pairs, axis=1)) + (threshold_log + 1) * np.eye(len(row))

        # Using boolean indexing for efficiency
        index_del_array = np.sum(new_M > threshold_log, axis=0) == len(row)
        indices_to_delete = np.where(index_del_array)[0]
        indices_del.update(indices_to_delete)

    # Using advanced indexing to delete indices
    fin_log_lparams = np.delete(log_lparams, list(indices_del), axis=1)
    final_params = np.mean(np.exp(fin_log_lparams), axis=0)
    return final_params


def calculate_log_moments(x_loc, y_loc, tau_list, integration_factor):
    dx4_log, dy4_log, dx2_log, dy2_log = [], [], [], []
    for i in tau_list:
        dx = np.diff(x_loc[::i * integration_factor])
        dy = np.diff(y_loc[::i * integration_factor])
        dx4_log.append(np.log(np.mean(dx**4)))
        dy4_log.append(np.log(np.mean(dy**4)))
        dx2_log.append(np.log(np.mean(dx**2)))
        dy2_log.append(np.log(np.mean(dy**2)))
    return np.array(dx4_log), np.array(dy4_log), np.array(dx2_log), np
    
    
################################################################################Logarthimic of moments#########################
# Momentum function definitions

def safe_log(x, min_val=1e-10, max_val=1e30):
    """ Compute logarithm, replacing non-positive or extremely large values """
    x_safe = np.clip(x, min_val, max_val)
    return np.log(x_safe)



def mom4_log(t, v0, D, lambdaB, lambdaD):
    """
        Calculate the logarithm of the fourth moment of a stochastic process.

        Parameters:
        t (float): Time parameter.
        v0 (float): Initial velocity.
        D (float): Diffusion coefficient.
        lambdaB (float): Rate parameter B.
        lambdaD (float): Rate parameter D.

        Returns:
        float: Logarithm of the fourth moment.
        """
    C1 = 3 * lambdaB ** (-2) * (lambdaB + lambdaD) ** (-2) * (2 * D * lambdaB ** 2 + v0 ** 2 * lambdaD) ** 2
    C2 = 3 * lambdaB ** (-3) * lambdaD * (lambdaB + lambdaD) ** (-3) * (
            8 * D ** 2 * lambdaB ** 4 - 8 * D * v0 ** 2 * lambdaB ** 2 * (2 * lambdaB + lambdaD) + v0 ** 4 * (
            3 * lambdaB ** 2 - 2 * lambdaB * lambdaD - 3 * lambdaD ** 2))
    C3 = 3 * lambdaB ** (-4) * lambdaD * (lambdaB + lambdaD) ** (-4) * (
            -8 * D ** 2 * lambdaB ** 5 + 8 * D * v0 ** 2 * lambdaB ** 2 * (
            3 * lambdaB ** 2 + 3 * lambdaB * lambdaD + lambdaD ** 2) + v0 ** 4 * (
                    -9 * lambdaB ** 3 - 7 * lambdaB ** 2 * lambdaD + 3 * lambdaB * lambdaD ** 2 + 3 * lambdaD ** 3))
    C4 = 3 / 2 * v0 ** 4 * lambdaB ** (-2) * lambdaD * (lambdaB + lambdaD) ** (-1)
    C5 = 6 * v0 ** 4 * lambdaB ** (-2) * (lambdaB + lambdaD) ** (-1)
    C6 = 0.0
    C7 = -3 * v0 ** 2 / (lambdaB ** (4) * lambdaD * (lambdaB + lambdaD)) * (8 * D * lambdaB ** 2 * lambdaD + v0 ** 2 * (
            2 * lambdaB ** 2 - 6 * lambdaB * lambdaD + 3 * lambdaD ** 2))
    C8 = 6 * lambdaB / (lambdaD * (lambdaB + lambdaD) ** (4)) * (v0 ** 2 + 2 * D * lambdaD) ** 2
    expr = (C1 * t ** 2 + C2 * t + C3 +
            C4 * t ** 2 * np.exp(-lambdaB * t) +
            C5 * t * np.exp(-lambdaB * t) +
            C6 * t * np.exp(-(lambdaB + lambdaD) * t) +
            C7 * np.exp(-lambdaB * t) +
            C8 * np.exp(-(lambdaB + lambdaD) * t))

    # Debugging: Print values that are about to be logged
    # print("Debug: expr values before log:", expr)

    expr_safe = np.maximum(expr, 1e-10)

    # Debugging: Print values after ensuring positivity
    # print("Debug: expr_safe values after ensuring positivity:", expr_safe)

    return np.log(expr_safe)


def to_optimize_mom4_log(params):
    try:
        model_result = mom4_serg_log(tau_list, *params)
        return np.sum(np.abs(dx4_log - safe_log(model_result)))
    except Exception as e:
        # print("Error encountered:", e)
        return 1e10


def to_optimize_mom22_4_diff_log(params):
    try:
        model_result = mom22_4_diff_serg_log(tau_list, *params)
        return np.sum(np.abs(difference - safe_log(model_result)))
    except Exception as e:
        # print("Error encountered:", e)
        return 1e10
def mom22_4_diff_log(l_tau, v, D, l_lambdaB, l_lambdaD):
    """
        Calculate a modified version of the second moment in logarithmic form.

        Parameters:
        l_tau, v, D, l_lambdaB, l_lambdaD: Same as in mom2_serg_log.

        Returns:
        float: Modified logarithmic second moment.
        """
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB/(l_beta)    
    C1_2 = 2*((v/l_beta)**2) * (l_alpha - 1)/(l_alpha**2)
    C2_2 = 2*((1-l_alpha)/l_alpha) * ((v**2)/l_beta) + 4*D*l_alpha
    
    expr2  = 2*np.log( (C1_2*(1-np.exp(-l_alpha*l_beta*l_tau)) + C2_2*l_tau )/2)
    
    lambdaB = l_lambdaB
    lambdaD = l_lambdaD
    v0 = v
    t = l_tau
    
    C1 = 3 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -2 ) * ( 2 * D * lambdaB**2 + v0**2 * lambdaD )**2
    C2 = 3 * lambdaB**( -3 ) * lambdaD * ( lambdaB + lambdaD )**( -3 ) * ( 8 * D**2 * lambdaB**4 - 8 * D * v0**2 * lambdaB**2 * ( 2 * lambdaB + lambdaD ) + v0**4 * ( 3 * lambdaB**2 - 2 * lambdaB * lambdaD - 3 * lambdaD**2 ) )
    C3 = 3* lambdaB**( -4 ) * lambdaD * ( lambdaB + lambdaD )**( -4 ) * ( -8 * D**2 * lambdaB**5 + 8 * D * v0**2 * lambdaB**2 * ( 3 * lambdaB**2 + 3 * lambdaB * lambdaD + lambdaD**2 ) + v0**4 * ( -9 * lambdaB**3 - 7 * lambdaB**2 * lambdaD + 3 * lambdaB * lambdaD**2 + 3 * lambdaD**3 ) )
    C4 = 3/2 * v0**4 * lambdaB**( -2 ) * lambdaD * ( lambdaB + lambdaD )**( -1 )
    C5 = 6 * v0**4 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -1 )
    C6 = 0.0
    C7 = -3 * v0**2 / (lambdaB**( 4 ) * lambdaD * ( lambdaB + lambdaD )) * ( 8 * D * lambdaB**2  * lambdaD  + v0**2 * ( 2 * lambdaB**2 - 6 * lambdaB * lambdaD + 3 * lambdaD**2 ) )
    C8 = 6 * lambdaB * lambdaD**( -1 ) * ( lambdaB + lambdaD )**( -4 ) * ( v0**2 + 2 * D * lambdaD )**2
    expr = np.log(C1 * t**2 + C2 * t + C3 + C4 * t**2 * np.e**( -lambdaB * t ) + C5 * t * np.e**( -lambdaB * t ) + C6 * t * np.e**( -( lambdaB + lambdaD ) * t ) + C7 * np.e**( -lambdaB * t ) + C8 * np.e**( -( lambdaB + lambdaD ) * t ))
    #print('a',[C1,C2<,C3,C4,C5,C6,C7,C8])
    return(expr - expr2)


