from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from ..math import gdem
from .equilibriumresult import EquilibriumResult


def bubble_sus(P_T, X, T_P, tipo, y_guess, eos, vl0, vv0):

    if tipo == 'T':
        P = P_T
        T = T_P
    elif tipo == 'P':
        T = P_T
        P = T_P

    # Liquid fugacities
    lnphil, vl = eos.logfugef(X, T, P, 'L', vl0)

    tol = 1e-8
    error = 1
    itacc = 0
    niter = 0
    n = 5
    Y_calc = y_guess
    Y = y_guess

    # Vapour fugacities
    lnphiv, vv = eos.logfugef(Y, T, P, 'V', vv0)

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
        lnphiv, vv = eos.logfugef(Y, T, P, 'V', vv)

    if tipo == 'T':
        f0 = Y_calc.sum() - 1
    elif tipo == 'P':
        f0 = np.log(Y_calc.sum())

    return f0, Y, lnK, vl, vv


def bubble_newton(inc, X, T_P, tipo, eos, vl0, vv0):
    global vl, vv
    f = np.zeros_like(inc)
    lnK = inc[:-1]
    K = np.exp(lnK)

    if tipo == 'T':
        P = inc[-1]
        T = T_P
    elif tipo == 'P':
        T = inc[-1]
        P = T_P

    Y = X*K

    # Liquid fugacities
    lnphil, vl = eos.logfugef(X, T, P, 'L', vl0)
    # Vapour fugacities
    lnphiv, vv = eos.logfugef(Y, T, P, 'V', vv0)

    f[:-1] = lnK + lnphiv - lnphil
    f[-1] = (Y-X).sum()

    return f


def bubblePy(y_guess, P_guess, X, T, model, good_initial=False,
             v0=[None, None], full_output=False):
    """
    Bubble point (T, x) -> (P, y)

    Solves bubble point at given liquid composition and temperature. It uses a
    combination of accelerated successive sustitution with quasi Newton Method
    in regular cases and when good initial it's provided the full system of
    equations of the phase envelope method is used as objective function.

    Parameters
    ----------
    y_guess : array_like
        guess of vapour phase composition
    P_guess : float
        guess of equilibrium pressure in bar.
    x : array_like
        liquid phase composition
    T : float
        absolute temperature of the liquid in K.
    model : object
        create from mixture, eos and mixrule
    good_initial: bool, optional
        if True skip succesive substitution and solves by Newton's Method.
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    Y : array_like, vector of vapour fraction moles
    P : float, equilibrium pressure in bar

    """
    nc = model.nc
    if len(y_guess) != nc or len(X) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vl, vv
    vl0, vv0 = v0

    it = 0
    itmax = 10
    tol = 1e-8

    P = P_guess
    f, Y, lnK, vl, vv = bubble_sus(P, X, T, 'T', y_guess, model, vl0, vv0)
    error = np.abs(f)
    h = 1e-4

    while error > tol and it <= itmax and not good_initial:
        it += 1
        f1, Y1, lnK1, vl, vv = bubble_sus(P+h, X, T, 'T', Y, model, vl, vv)
        f, Y, lnK, vl, vv = bubble_sus(P, X, T, 'T', Y, model, vl, vv)
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
        sol1 = root(bubble_newton, inc0, args=(X, T, 'T', model, vl, vv))
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
    Bubble point (P, x) -> (T, y)

    Solves bubble point at given liquid composition and pressure. It uses a
    combination of accelerated successive sustitution with quasi Newton Method
    in regular cases and when good initial it's provided the full system of
    equations of the phase envelope method is used as objective function.

    Parameters
    ----------
    y_guess : array_like
        guess of vapour phase composition
    T_guess : float
        guess of equilibrium temperature of the liquid in K.

    x : array_like
        liquid phase composition
    P : float
        pressure of the liquid in bar
    model : object
        create from mixture, eos and mixrule
    good_initial: bool, optional
        if True skip succesive substitution and solves by Newton's Method.
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    full_output: bool, optional
        wheter to outputs all calculation info


    Returns
    -------
    Y : array_like
        vector of vapour fraction moles
    T : float
        equilibrium temperature in K
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
        sol1 = root(bubble_newton, inc0, args=(X, P, 'P', model, vl, vv))
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
