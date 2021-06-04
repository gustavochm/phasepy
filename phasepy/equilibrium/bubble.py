from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from ..math import gdem
from .equilibriumresult import EquilibriumResult


def bubble_sus(P_T, X, T_P, type, y_guess, eos, vl0, vv0):

    if type == 'T':
        P = P_T
        temp_aux = T_P
    elif type == 'P':
        T = P_T
        temp_aux = eos.temperature_aux(T)
        P = T_P

    # Liquid fugacities
    lnphil, vl = eos.logfugef_aux(X, temp_aux, P, 'L', vl0)

    tol = 1e-8
    error = 1
    itacc = 0
    niter = 0
    n = 5
    Y_calc = y_guess
    Y = y_guess

    # Vapor fugacities
    lnphiv, vv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv0)

    while error > tol and itacc < 3:
        niter += 1

        lnK = lnphil-lnphiv
        K = np.exp(lnK)
        Y_calc_old = Y_calc
        Y_calc = X*K

        if niter == (n-3):
            Y3 = Y_calc
        elif niter == (n-2):
            Y2 = Y_calc
        elif niter == (n-1):
            Y1 = Y_calc
        elif niter == n:
            niter = 0
            itacc += 1
            dacc = gdem(Y_calc, Y1, Y2, Y3)
            Y_calc += dacc
        error = np.linalg.norm(Y_calc-Y_calc_old)
        Y = Y_calc/Y_calc.sum()
        # Vapor fugacities
        lnphiv, vv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv)

    if type == 'T':
        f0 = Y_calc.sum() - 1
    elif type == 'P':
        f0 = np.log(Y_calc.sum())

    return f0, Y, lnK, vl, vv


def bubble_newton(inc, X, T_P, type, eos):
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

    Y = X*K

    # Liquid fugacities
    lnphil, vl = eos.logfugef_aux(X, temp_aux, P, 'L', vl)
    # Vapor fugacities
    lnphiv, vv = eos.logfugef_aux(Y, temp_aux, P, 'V', vv)

    f[:-1] = lnK + lnphiv - lnphil
    f[-1] = (Y-X).sum()

    return f


def bubblePy(y_guess, P_guess, X, T, model, good_initial=False,
             v0=[None, None], full_output=False):
    """
    Bubble point (X, T) -> (Y, P).

    Solves bubble point (vapor phase composition and pressure) at given
    temperature and liquid composition.

    Parameters
    ----------
    y_guess : array
        Initial guess of vapor phase molar fractions
    P_guess : float
        Initial guess for equilibrium pressure [bar]
    X : array
        Liquid phase molar fractions
    T : float
        Absolute temperature [K]
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
    Y : array
        Vapor molar fractions
    P : float
        Equilibrium pressure [bar]
    """
    nc = model.nc
    if len(y_guess) != nc or len(X) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vl, vv
    vl0, vv0 = v0

    temp_aux = model.temperature_aux(T)

    it = 0
    itmax = 10
    tol = 1e-8

    P = P_guess
    f, Y, lnK, vl, vv = bubble_sus(P, X, temp_aux, 'T', y_guess, model,
                                   vl0, vv0)
    error = np.abs(f)
    h = 1e-4

    while error > tol and it <= itmax and not good_initial:
        it += 1
        f1, Y1, lnK1, vl, vv = bubble_sus(P+h, X, temp_aux, 'T', Y, model,
                                          vl, vv)
        f, Y, lnK, vl, vv = bubble_sus(P, X, temp_aux, 'T', Y, model, vl, vv)
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
        sol1 = root(bubble_newton, inc0, args=(X, temp_aux, 'T', model))
        sol = sol1.x
        lnK = sol[:-1]
        error = np.linalg.norm(sol1.fun)
        it += sol1.nfev
        Y = np.exp(lnK)*X
        Y /= Y.sum()
        P = sol[-1]

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'state1': 'Liquid',
               'Y': Y, 'v2': vv, 'state2': 'Vapor'}
        out = EquilibriumResult(sol)
        return out

    return Y, P


def bubbleTy(y_guess, T_guess, X, P, model, good_initial=False,
             v0=[None, None], full_output=False):
    """
    Bubble point (X, P) -> (Y, T).

    Solves bubble point (vapor phase composition and temperature) at given
    pressure and liquid phase composition.

    Parameters
    ----------
    y_guess : array
        Initial guess of vapor phase molar fractions
    T_guess : float
        Initial guess of equilibrium temperature [K]
    X : array
        Liquid phase molar fractions
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
    Y : array
        Vapor molar fractions
    T : float
        Equilibrium temperature [K]
    """

    nc = model.nc
    if len(y_guess) != nc or len(X) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vl, vv

    vl0, vv0 = v0

    it = 0
    itmax = 10
    tol = 1e-8

    T = T_guess
    f, Y, lnK, vl, vv = bubble_sus(T, X, P, 'P', y_guess, model, vl0, vv0)
    error = np.abs(f)
    h = 1e-4

    while error > tol and it <= itmax and not good_initial:
        it += 1
        f1, Y1, lnK1, vl, vv = bubble_sus(T+h, X, P, 'P', Y, model, vl, vv)
        f, Y, lnK, vl, vv = bubble_sus(T, X, P, 'P', Y, model, vl, vv)
        df = (f1-f)/(h)
        T -= f/df
        error = np.abs(f)

    if error > tol:
        inc0 = np.hstack([lnK, T])
        sol1 = root(bubble_newton, inc0, args=(X, P, 'P', model))
        sol = sol1.x
        lnK = sol[:-1]
        error = np.linalg.norm(sol1.fun)
        it += sol1.nfev
        Y = np.exp(lnK)*X
        Y /= Y.sum()
        T = sol[-1]

    if full_output:
        sol = {'T': T, 'P': P, 'error': error, 'iter': it,
               'X': X, 'v1': vl, 'state1': 'Liquid',
               'Y': Y, 'v2': vv, 'state2': 'Vapor'}
        out = EquilibriumResult(sol)
        return out

    return Y, T


__all__ = ['bubbleTy', 'bubblePy']
