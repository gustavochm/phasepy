from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from ..math import gdem
from .equilibriumresult import EquilibriumResult


# ELV phi-phi
def dew_sus(P_T, Y, T_P, type, x_guess, eos, vl0, vv0):

    if type == 'T':
        P = P_T
        temp_aux = T_P
    elif type == 'P':
        T = P_T
        temp_aux = eos.temperature_aux(T)
        P = T_P

    # Vapour fugacities
    lnphiv, vv0 = eos.logfugef_aux(Y, temp_aux, P, 'V', vv0)

    tol = 1e-8
    error = 1
    itacc = 0
    niter = 0
    n = 5
    X_calc = x_guess
    X = x_guess

    while error > tol and itacc < 3:
        niter += 1

        # Liquid fugacitiies
        lnphil, vl0 = eos.logfugef_aux(X, temp_aux, P, 'L', vl0)

        lnK = lnphil-lnphiv
        K = np.exp(lnK)
        X_calc_old = X_calc
        X_calc = Y/K

        if niter == (n-3):
            X3 = X_calc
        elif niter == (n-2):
            X2 = X_calc
        elif niter == (n-1):
            X1 = X_calc
        elif niter == n:
            niter = 0
            itacc += 1
            dacc = gdem(X_calc, X1, X2, X3)
            X_calc += dacc
        error = np.linalg.norm(X_calc - X_calc_old)
        X = X_calc/X_calc.sum()

    if type == 'T':
        f0 = X_calc.sum() - 1
    elif type == 'P':
        f0 = np.log(X_calc.sum())

    return f0, X, lnK, vl0, vv0


def dew_newton(inc, Y, T_P, type, eos):

    global vl, vv

    f = np.zeros_like(inc)
    lnK = inc[:-1]
    K = np.exp(lnK)

    if type == 'T':
        P = inc[-1]
        temp_aux = T_P
    elif type == 'P':
        T = inc[-1]
        temp_aux = eos.temperature_aux(T)
        P = T_P

    X = Y/K

    # Liquid fugacities
    lnphil, vl = eos.logfugef_aux(X, temp_aux, P, 'L', vl)
    # Vapor fugacities
    lnphiv, vv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv)

    f[:-1] = lnK + lnphiv - lnphil
    f[-1] = (Y-X).sum()

    return f


def dewPx(x_guess, P_guess, y, T, model, good_initial=False,
          v0=[None, None], full_output=False):
    """
    Dew point (y, T) -> (x, P)

    Solves dew point (liquid phase composition and pressure) at given
    temperature and vapor composition.

    Parameters
    ----------
    x_guess : array
        Initial guess of liquid phase molar fractions
    P_guess : float
        Initial guess of equilibrium pressure [bar]
    y : array
        Vapor phase molar fractions
    T : float
        Temperature [K]
    model : object
        Phase equilibrium model object
    good_initial: bool, optional
        If True uses only phase envelope method in solution
    v0 : list, optional
        Liquid and vapor phase molar volume used as initial values to
        compute fugacities
    full_output: bool, optional
        Flag to return a dictionary of all calculation info

    Returns
    -------
    x : array
        Liquid molar fractions
    P : float
        Equilibrium pressure [bar]
    """
    nc = model.nc
    if len(x_guess) != nc or len(y) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vl, vv
    vl0, vv0 = v0

    temp_aux = model.temperature_aux(T)

    it = 0
    itmax = 10
    tol = 1e-8

    P = P_guess
    f, X, lnK, vl, vv = dew_sus(P, y, temp_aux, 'T', x_guess, model, vl0, vv0)
    error = np.abs(f)
    h = 1e-3

    while error > tol and it <= itmax and not good_initial:
        it += 1
        f1, X1, lnK1, vl, vv = dew_sus(P+h, y, temp_aux, 'T', X, model, vl, vv)
        f, X, lnK, vl, vv = dew_sus(P, y, temp_aux, 'T', X, model, vl, vv)
        df = (f1-f)/h
        dP = f / df
        if dP > P:
            dP = 0.4 * P
        elif np.isnan(dP):
            dP = 0.0
            it = 1.*itmax
        P -= dP
        error = np.abs(f)

    if error > tol:
        inc0 = np.hstack([lnK, P])
        sol1 = root(dew_newton, inc0, args=(y, temp_aux, 'T', model))
        sol = sol1.x
        lnK = sol[:-1]
        error = np.linalg.norm(sol1.fun)
        it += sol1.nfev
        lnK = sol[:-1]
        X = y / np.exp(lnK)
        P = sol[-1]

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'state1': 'Liquid',
               'Y': y, 'v2': vv, 'state2': 'Vapor'}
        out = EquilibriumResult(sol)
        return out

    return X, P


def dewTx(x_guess, T_guess, y, P, model, good_initial=False,
          v0=[None, None], full_output=False):
    """
    Dew point (y, P) -> (x, T)

    Solves dew point (liquid phase composition and temperature) at given
    pressure and vapor phase composition.

    Parameters
    ----------
    x_guess : array
        Initial guess of liquid phase molar fractions
    T_guess : float
        Initial guess of equilibrium temperature [K]
    y : array
        Vapor phase molar fractions
    P : float
        Pressure [bar]
    model : object
        Phase equilibrium model object
    good_initial: bool, optional
        If True uses only phase envelope method in solution
    v0 : list, optional
        Liquid and vapor phase molar volume used as initial values to
        compute fugacities
    full_output: bool, optional
        Flag to return a dictionary of all calculation info

    Returns
    -------
    x : array
        Liquid molar fractions
    T : float
        Equilibrium temperature [K]
    """

    nc = model.nc
    if len(x_guess) != nc or len(y) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vl, vv
    vl0, vv0 = v0

    it = 0
    itmax = 10
    tol = 1e-8

    T = T_guess
    f, X, lnK, vl, vv = dew_sus(T, y, P, 'P', x_guess, model, vl0, vv0)
    error = np.abs(f)
    h = 1e-4

    while error > tol and it <= itmax and not good_initial:
        it += 1
        f1, X1, lnK1, vl, vv = dew_sus(T+h, y, P, 'P', X, model, vl, vv)
        f, X, lnK, vl, vv = dew_sus(T, y, P, 'P', X, model, vl, vv)
        df = (f1-f)/h
        if np.isnan(df):
            df = 0.0
            it = 1.*itmax
        T -= f/df
        error = np.abs(f)

    if error > tol:
        inc0 = np.hstack([lnK, T])
        sol1 = root(dew_newton, inc0, args=(y, P, 'P', model))
        sol = sol1.x
        lnK = sol[:-1]
        error = np.linalg.norm(sol1.fun)
        it += sol1.nfev
        lnK = sol[:-1]
        X = y / np.exp(lnK)
        T = sol[-1]

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'state1': 'Liquid',
               'Y': y, 'v2': vv, 'state2': 'Vapor'}
        out = EquilibriumResult(sol)
        return out

    return X, T


__all__ = ['dewTx', 'dewPx']
