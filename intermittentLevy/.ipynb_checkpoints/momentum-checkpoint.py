# Necessary library imports for the momentum functions
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.neighbors import KernelDensity
import functools
import math
#import functions_eye_tracker_project as funcs
#import separation_algorithm as sepa
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import random
import warnings
warnings.filterwarnings("error")

# Additional imports if needed

# Momentum function definitions

def mom22_4_diff_serg_log(l_tau, v, D, l_lambdaB, l_lambdaD):
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB / (l_beta)
    C1_2 = 2 * ((v / l_beta) ** 2) * (l_alpha - 1) / (l_alpha ** 2)
    C2_2 = 2 * ((1 - l_alpha) / l_alpha) * ((v ** 2) / l_beta) + 4 * D * l_alpha

    expr2 = 2 * np.log((C1_2 * (1 - np.exp(-l_alpha * l_beta * l_tau)) + C2_2 * l_tau) / 2)

    lambdaB = l_lambdaB
    lambdaD = l_lambdaD
    v0 = v
    t = l_tau

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
    C8 = 6 * lambdaB * lambdaD ** (-1) * (lambdaB + lambdaD) ** (-4) * (v0 ** 2 + 2 * D * lambdaD) ** 2
    expr = np.log(C1 * t ** 2 + C2 * t + C3 + C4 * t ** 2 * np.e ** (-lambdaB * t) + C5 * t * np.e ** (
                -lambdaB * t) + C6 * t * np.e ** (-(lambdaB + lambdaD) * t) + C7 * np.e ** (
                              -lambdaB * t) + C8 * np.e ** (-(lambdaB + lambdaD) * t))
    # print('a',[C1,C2<,C3,C4,C5,C6,C7,C8])
    return (expr - expr2)


def mom4_serg_log(t, v0, D, lambdaB, lambdaD):
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
    expr = C1 * t ** 2 + C2 * t + C3 + C4 * t ** 2 * np.e ** (-lambdaB * t) + C5 * t * np.e ** (
                -lambdaB * t) + C6 * t * np.e ** (-(lambdaB + lambdaD) * t) + C7 * np.e ** (
                       -lambdaB * t) + C8 * np.e ** (-(lambdaB + lambdaD) * t)
    print('a', [C1, C2, C3, C4, C5, C6, C7, C8])
    return (np.log(expr))


def frequency_matrix_2D(d__ss, threshold, normalized):
    d__ss = np.array(d__ss)
    d__ss = (d__ss - min(d__ss)) / ((max(d__ss) - min(d__ss)) * 1.000001)
    binary_vector = np.array(d__ss > threshold).astype(int)
    matrix = np.histogram2d(binary_vector[1:], binary_vector[:-1], [0, 1, 2])[0]
    if normalized:
        for j in range(2):
            matrix[j, :] = matrix[j, :] / matrix.sum(axis=1)[j]
    return (matrix)


# in threshold array, 0 corresponds to the minimum of the vector while 1 is the maximum of the vector
def form_groups(vector, threshold_array, graph, x_label, title, x_axis_format):
    detectionfisher = []
    detection = []
    for i in threshold_array:
        matrix = frequency_matrix_2D(vector, i, False)
        detectionfisher.append(np.log(scipy.stats.fisher_exact(matrix)[1]))
        p = matrix.sum(axis=1)[0] / matrix.sum()
        if p == 1 or p == 0:
            detection.append(1)
        else:
            detection.append((matrix[1][0]) / (matrix.sum() * (p * (1 - p))))

    minim = min(vector)
    diff = max(vector) - minim
    min_k = np.argmin(detection) * diff + minim
    min_fisher = np.argmin(detectionfisher) * diff + minim

    if graph:

        xticks_labels = [x_axis_format % (minim + diff * pipi) for pipi in threshold_array]

        plt.plot(detection)
        plt.xlabel(x_label)
        # plt.title(title +"k min = " +str(round(min_k,5)))
        plt.title(title)
        if len(threshold_array) > 40:
            plt.xticks(np.arange(len(threshold_array))[::int(len(threshold_array) / 10)],
                       xticks_labels[::int(len(threshold_array) / 10)])
        else:
            plt.xticks(np.arange(len(threshold_array))[::4], xticks_labels[::4])
        plt.ylabel("k")
        plt.savefig("group_detection - k " + x_label + title + ".png", dpi=500)
        plt.show()
        plt.close()

        plt.plot(detectionfisher)
        plt.xlabel(x_label)
        plt.title(title)
        if len(threshold_array) > 40:
            plt.xticks(np.arange(len(threshold_array))[::int(len(threshold_array) / 10)],
                       xticks_labels[::int(len(threshold_array) / 10)])
        else:
            plt.xticks(np.arange(len(threshold_array))[::4], xticks_labels[::4])
        plt.ylabel("log-fisher exact test")
        plt.savefig("group_detection - log-fisher " + x_label + title + ".png", dpi=500)
        plt.show()
        plt.close()

    return (detection, detectionfisher, min_k, min_fisher)


def real_k_and_fisher(binary_vector):
    matrix = np.zeros((2, 2))
    for i in range(len(binary_vector) - 1):
        matrix[int(binary_vector[i])][int(binary_vector[i + 1])] += 1

    detectionfisher = []
    detection = []

    detectionfisher.append(np.log(scipy.stats.fisher_exact(matrix)[1]))
    p = matrix.sum(axis=1)[0] / matrix.sum()
    detection.append((matrix[1][0]) / (matrix.sum() * (p * (1 - p))))

    return (matrix, detection, detectionfisher)


def mom4_serg_log(t, v0, D, lambdaB, lambdaD):
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
    expr = C1 * t ** 2 + C2 * t + C3 + C4 * t ** 2 * np.e ** (-lambdaB * t) + C5 * t * np.e ** (
                -lambdaB * t) + C6 * t * np.e ** (-(lambdaB + lambdaD) * t) + C7 * np.e ** (
                       -lambdaB * t) + C8 * np.e ** (-(lambdaB + lambdaD) * t)
    # print('a',[C1,C2<,C3,C4,C5,C6,C7,C8])
    return (np.log(expr))


def mom2_serg_log(l_tau, v, D, l_lambdaB, l_lambdaD):
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB / (l_beta)
    C1 = 2 * ((v / l_beta) ** 2) * (l_alpha - 1) / (l_alpha ** 2)
    C2 = 2 * ((1 - l_alpha) / l_alpha) * ((v ** 2) / l_beta) + 4 * D * l_alpha

    expr2 = 0.5 * (C1 * (1 - np.exp(-l_alpha * l_beta * l_tau)) + C2 * l_tau)
    return (np.log(expr2))


def to_optimize_mom4_serg_log(variables):
    v0, D, lambdaB, lambdaD = variables
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
    expr = C1 * np.array(global_tau_list) ** 2 + C2 * np.array(global_tau_list) + C3 + C4 * np.array(
        global_tau_list) ** 2 * np.e ** (-lambdaB * np.array(global_tau_list)) + C5 * np.array(
        global_tau_list) * np.e ** (-lambdaB * np.array(global_tau_list)) + C6 * np.array(global_tau_list) * np.e ** (
                       -(lambdaB + lambdaD) * np.array(global_tau_list)) + C7 * np.e ** (
                       -lambdaB * np.array(global_tau_list)) + C8 * np.e ** (
                       -(lambdaB + lambdaD) * np.array(global_tau_list))
    l_num = np.mean((np.array(global_logdx4) - np.log(np.array(expr))) ** 2)
    return (l_num)


def r_square(l_emp_points, l_emp_fit):
    l_num = np.mean((np.array(l_emp_points) - np.array(l_emp_fit)) ** 2)
    l_den = np.std(np.array(l_emp_points)) ** 2
    return (1 - l_num / l_den)


def adjusted_r_square(l_emp_points, l_emp_fit, degrees_freedom):
    n = len(l_emp_points)
    l_num = np.mean((np.array(l_emp_points) - np.array(l_emp_fit)) ** 2)
    l_den = np.std(np.array(l_emp_points)) ** 2
    rsqu = 1 - l_num / l_den
    return (1 - (1 - rsqu) * ((n - 1) / (n - degrees_freedom)))


def powerl_fit(l_tau, l_k, l_a):
    return (l_k * np.power(2, l_tau * l_a))


def to_optimize_mom4_serg_log_vdl(variables):
    v0, D, lambdaB = variables
    lambdaD = tos_lambdaD

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
    expr = C1 * np.array(global_tau_list) ** 2 + C2 * np.array(global_tau_list) + C3 + C4 * np.array(
        global_tau_list) ** 2 * np.e ** (-lambdaB * np.array(global_tau_list)) + C5 * np.array(
        global_tau_list) * np.e ** (-lambdaB * np.array(global_tau_list)) + C6 * np.array(global_tau_list) * np.e ** (
                       -(lambdaB + lambdaD) * np.array(global_tau_list)) + C7 * np.e ** (
                       -lambdaB * np.array(global_tau_list)) + C8 * np.e ** (
                       -(lambdaB + lambdaD) * np.array(global_tau_list))
    l_num = np.mean((np.array(global_logdx4) - np.log(np.array(expr))) ** 2)
    l_num = np.mean(np.abs(np.array(global_logdx4) - np.log(np.array(expr))) / np.array(global_logdx4))
    return (l_num)


def to_optimize_mom4_serg_log_vl(variables):
    v0, lambdaB = variables
    lambdaD = tos_lambdaD
    D = tos_D

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
    expr = C1 * np.array(global_tau_list) ** 2 + C2 * np.array(global_tau_list) + C3 + C4 * np.array(
        global_tau_list) ** 2 * np.e ** (-lambdaB * np.array(global_tau_list)) + C5 * np.array(
        global_tau_list) * np.e ** (-lambdaB * np.array(global_tau_list)) + C6 * np.array(global_tau_list) * np.e ** (
                       -(lambdaB + lambdaD) * np.array(global_tau_list)) + C7 * np.e ** (
                       -lambdaB * np.array(global_tau_list)) + C8 * np.e ** (
                       -(lambdaB + lambdaD) * np.array(global_tau_list))
    l_num = np.mean((np.array(global_logdx4) - np.log(np.array(expr))) ** 2)
    l_num = np.mean(np.abs(np.array(global_logdx4) - np.log(np.array(expr))) / np.array(global_logdx4))
    return (l_num)


def Sergyi_expr_simp1(l_tau, v, D, l_lambdaB, l_lambdaD):
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB / (l_beta)
    C1 = 2 * ((v / l_beta) ** 2) * (l_alpha - 1) / (l_alpha ** 2)
    C2 = 2 * ((1 - l_alpha) / l_alpha) * ((v ** 2) / l_beta) + 4 * D * l_alpha

    return (C1 * (1 - np.exp(-l_alpha * l_beta * l_tau)) + C2 * l_tau)


def to_optimize_second_l(l_lambdaD):
    l_beta = tos_lambdaB + l_lambdaD
    l_alpha = tos_lambdaB / (l_beta)
    C1 = 2 * ((tos_v / l_beta) ** 2) * (l_alpha - 1) / (l_alpha ** 2)
    C2 = 2 * ((1 - l_alpha) / l_alpha) * ((tos_v ** 2) / l_beta) + 4 * tos_D * l_alpha
    expr22 = C1 * (1 - np.exp(-l_alpha * l_beta * global_tau_list)) + C2 * global_tau_list
    expr2 = 0.5 * expr22
    l_num = np.mean(np.abs(np.array(global_logdx2) - np.log(np.array(expr2))) / np.array(global_logdx2))
    return (l_num)


def to_optimize_second_ld(lvariables):
    l_D, l_lambdaD = lvariables
    l_beta = tos_lambdaB + l_lambdaD
    l_alpha = tos_lambdaB / (l_beta)
    C1 = 2 * ((tos_v / l_beta) ** 2) * (l_alpha - 1) / (l_alpha ** 2)
    C2 = 2 * ((1 - l_alpha) / l_alpha) * ((tos_v ** 2) / l_beta) + 4 * l_D * l_alpha
    expr22 = C1 * (1 - np.exp(-l_alpha * l_beta * global_tau_list)) + C2 * global_tau_list
    expr2 = 0.5 * expr22
    l_num = np.mean(np.abs(np.array(global_logdx2) - np.log(np.array(expr2))) / np.array(global_logdx2))
    return (l_num)


def to_optimize_second_ll(lvariables):
    l_lambdaB, l_lambdaD = lvariables
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB / (l_beta)
    C1 = 2 * ((tos_v / l_beta) ** 2) * (l_alpha - 1) / (l_alpha ** 2)
    C2 = 2 * ((1 - l_alpha) / l_alpha) * ((tos_v ** 2) / l_beta) + 4 * tos_D * l_alpha
    expr22 = C1 * (1 - np.exp(-l_alpha * l_beta * global_tau_list)) + C2 * global_tau_list
    expr2 = 0.5 * expr22
    l_num = np.mean(np.abs(np.array(global_logdx2) - np.log(np.array(expr2))) / np.array(global_logdx2))
    return (l_num)


def weighted_error_fourth_second(ltau, lv, ld, llambdab, llambdad, lemp_fourth, lemp_second):
    LTheo4 = np.array(mom4_serg_log(ltau, lv, ld, llambdab, llambdad))
    LTheo2 = np.array(mom4_serg_log(ltau, lv, ld, llambdab, llambdad))

# additional functions or code here if needed
