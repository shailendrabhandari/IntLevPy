# moments.py

import numpy as np


def mom4_serg_log(t, v0, D, lambdaB, lambdaD):
    t = np.array(np.hstack([t]), dtype=np.longdouble)
    C1 = (
        3
        * lambdaB ** (-2)
        * (lambdaB + lambdaD) ** (-2)
        * (2 * D * lambdaB ** 2 + v0 ** 2 * lambdaD) ** 2
    )
    C2 = (
        3
        * lambdaB ** (-3)
        * lambdaD
        * (lambdaB + lambdaD) ** (-3)
        * (
            8 * D ** 2 * lambdaB ** 4
            - 8 * D * v0 ** 2 * lambdaB ** 2 * (2 * lambdaB + lambdaD)
            + v0 ** 4 * (3 * lambdaB ** 2 - 2 * lambdaB * lambdaD - 3 * lambdaD ** 2)
        )
    )
    C3 = (
        3
        * lambdaB ** (-4)
        * lambdaD
        * (lambdaB + lambdaD) ** (-4)
        * (
            -8 * D ** 2 * lambdaB ** 5
            + 8 * D * v0 ** 2 * lambdaB ** 2 * (3 * lambdaB ** 2 + 3 * lambdaB * lambdaD + lambdaD ** 2)
            + v0 ** 4 * (-9 * lambdaB ** 3 - 7 * lambdaB ** 2 * lambdaD + 3 * lambdaB * lambdaD ** 2 + 3 * lambdaD ** 3)
        )
    )
    C4 = 3 / 2 * v0 ** 4 * lambdaB ** (-2) * lambdaD * (lambdaB + lambdaD) ** (-1)
    C5 = 6 * v0 ** 4 * lambdaB ** (-2) * (lambdaB + lambdaD) ** (-1)
    C6 = 0.0
    C7 = (
        -3
        * v0 ** 2
        / (lambdaB ** 4 * lambdaD * (lambdaB + lambdaD))
        * (
            8 * D * lambdaB ** 2 * lambdaD
            + v0 ** 2 * (2 * lambdaB ** 2 - 6 * lambdaB * lambdaD + 3 * lambdaD ** 2)
        )
    )
    C8 = (
        6
        * lambdaB
        / (lambdaD * (lambdaB + lambdaD) ** 4)
        * (v0 ** 2 + 2 * D * lambdaD) ** 2
    )
    expr = (
        C1 * t ** 2
        + C2 * t
        + C3
        + C4 * t ** 2 * np.exp(-lambdaB * t)
        + C5 * t * np.exp(-lambdaB * t)
        + C6 * t * np.exp(-(lambdaB + lambdaD) * t)
        + C7 * np.exp(-lambdaB * t)
        + C8 * np.exp(-(lambdaB + lambdaD) * t)
    )
    return np.log(8 * expr / 3)


def mom2_serg_log(tau, v, D, lambdaB, lambdaD):
    beta = lambdaB + lambdaD
    alpha = lambdaB / beta
    DI = (v ** 2) / beta
    C1 = -2 * (1 / (beta * alpha)) * ((1 - alpha) / alpha) * DI
    C2 = 2 * ((1 - alpha) / alpha) * DI + 4 * D * alpha
    expr2 = C1 * (1 - np.exp(-lambdaB * tau)) + C2 * tau
    return np.log(expr2)


def theor_levy_moment(n_mom, alpha, v, t, tmin):
    num = tmin ** alpha * (v ** n_mom) * n_mom * (alpha - 2) * t ** (2 + n_mom - alpha)
    den = (tmin ** 2) * (2 + n_mom - alpha) * (1 + n_mom - alpha)
    return num / den
