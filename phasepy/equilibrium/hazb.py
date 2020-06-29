from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from .equilibriumresult import EquilibriumResult


def haz_objb(inc, T_P, tipo, modelo, v0):

    X, W, Y, P_T = np.array_split(inc, 4)

    if tipo == 'T':
        P = P_T
        T = T_P
    elif tipo == 'P':
        T = P_T
        P = T_P

    global vx, vw, vy
    vx, vw, vy = v0

    fugX, vx = modelo.logfugef(X, T, P, 'L', vx)
    fugW, vw = modelo.logfugef(W, T, P, 'L', vw)
    fugY, vy = modelo.logfugef(Y, T, P, 'V', vy)

    K1 = np.exp(fugX-fugY)
    K2 = np.exp(fugX-fugW)
    return np.hstack([K1*X-Y, K2*X-W, X.sum()-1, Y.sum()-1, W.sum()-1])


def vlleb(X0, W0, Y0, P_T, T_P, spec, model,
          v0=[None, None, None], full_output=False):
    '''
    Solves liquid liquid vapour equilibrium for binary mixtures.
    (T,P) -> (x,w,y)

    Parameters
    ----------

    X0 : array_like
        guess composition of phase 1
    W0 : array_like
        guess composition of phase 1
    Y0 : array_like
        guess composition of phase 2
    P_T : float
        absolute temperature or pressure
    T_P : floar
        absolute temperature or pressure
    spec: string
        'T' if T_P is temperature or 'P' if pressure.
    model : object
        created from mixture, eos and mixrule
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------

    X : array_like
        liquid1 mole fraction vector
    W : array_like
        liquid2 mole fraction vector
    Y : array_like
        vapour mole fraction fector
    var: float
        temperature or pressure, depending of specification

    '''

    nc = model.nc

    if nc != 2:
        raise Exception('3 phase equilibra for binary mixtures')

    if len(X0) != nc or len(W0) != nc or len(Y0) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vx, vw, vy

    sol1 = root(haz_objb, np.hstack([X0, W0, Y0, P_T]),
                args=(T_P, spec, model, v0))
    error = np.linalg.norm(sol1.fun)
    nfev = sol1.nfev
    sol = sol1.x
    if np.any(sol < 0):
        raise Exception('negative Composition or T/P  founded')
    X, W, Y, var = np.array_split(sol, 4)

    if full_output:
        if spec == 'T':
            P = var
            T = T_P
        elif spec == 'P':
            T = var
            P = T_P
        inc = {'T': T, 'P': P, 'error': error, 'nfev': nfev,
               'X': X, 'vx': vx, 'statex': 'Liquid',
               'W': W, 'vw': vw, 'statew': 'Liquid',
               'Y': Y, 'vy': vy, 'statey': 'Vapor'}
        out = EquilibriumResult(inc)
        return out

    return X, W, Y, var
