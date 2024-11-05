# optimization.py

import numpy as np
from .moments import mom2_serg_log, mom4_serg_log


def to_optimize_mom4_serg_log(variables, tau_list, logdx4):
    v0, D, lambdaB, lambdaD = variables
    expr = mom4_serg_log(tau_list, v0, D, lambdaB, lambdaD)
    l_num = np.mean((logdx4 - expr) ** 2)
    return l_num


def to_optimize_mom2_serg_log(variables, tau_list, logdx2):
    v0, D, lambdaB, lambdaD = variables
    expr = mom2_serg_log(tau_list, v0, D, lambdaB, lambdaD)
    l_num = np.mean((logdx2 - expr) ** 2)
    return l_num


def to_optimize_mom4_and_2_serg_log(variables, tau_list, logdx2, logdx4):
    v0, D, lambdaB, lambdaD = variables
    expr4 = mom4_serg_log(tau_list, v0, D, lambdaB, lambdaD)
    expr2 = mom2_serg_log(tau_list, v0, D, lambdaB, lambdaD)
    l_num4 = np.mean((logdx4 - expr4) ** 2)
    l_num2 = np.mean((logdx2 - expr2) ** 2)
    l_num = l_num4 + 2 * l_num2
    return l_num


def to_optimize_mom4_serg_log_vl(variables, tau_list, logdx4, tos_D, tos_lambdaD):
    v0, lambdaB = variables
    D = tos_D
    lambdaD = tos_lambdaD
    expr = mom4_serg_log(tau_list, v0, D, lambdaB, lambdaD)
    l_num = np.mean(np.abs(logdx4 - expr) / logdx4)
    return l_num


def to_optimize_second_ll(variables, tau_list, logdx2, tos_v, tos_D):
    lambdaB, lambdaD = variables
    expr2 = mom2_serg_log(tau_list, tos_v, tos_D, lambdaB, lambdaD)
    l_num = np.mean(np.abs(logdx2 - expr2) / logdx2)
    return l_num

