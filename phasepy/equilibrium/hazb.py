from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from .equilibriumresult import EquilibriumResult

'''
def haz_objb(inc, T_P, type, model):

    X, W, Y, P_T = np.array_split(inc, 4)

    if type == 'T':
        P = P_T
        T = T_P
    elif type == 'P':
        T = P_T
        P = T_P

    temp_aux = model.temperature_aux(T)

    global vx, vw, vy

    fugX, vx = model.logfugef_aux(X, temp_aux, P, 'L')
    fugW, vw = model.logfugef_aux(W, temp_aux, P, 'L')
    fugY, vy = model.logfugef_aux(Y, temp_aux, P, 'V')

    K1 = np.exp(fugX-fugY)
    K2 = np.exp(fugX-fugW)
    return np.hstack([K1*X-Y, K2*X-W, X.sum()-1, Y.sum()-1, W.sum()-1])
'''

def haz_objb(inc, T_P, type, model):

    X, W, Y, P_T = np.array_split(inc, 4)
    # P_T = P_T[0]
    if type == 'T':
        P = P_T
        temp_aux = T_P
    elif type == 'P':
        T = P_T
        temp_aux = model.temperature_aux(T)
        P = T_P

    global vx, vw, vy

    fugX, vx = model.logfugef_aux(X, temp_aux, P, 'L', vx)
    fugW, vw = model.logfugef_aux(W, temp_aux, P, 'L', vw)
    fugY, vy = model.logfugef_aux(Y, temp_aux, P, 'V', vy)

    K1 = np.exp(fugX-fugY)
    K2 = np.exp(fugX-fugW)
    return np.hstack([K1*X-Y, K2*X-W, X.sum()-1, Y.sum()-1, W.sum()-1])


def vlleb(X0, W0, Y0, P_T, T_P, spec, model, v0=[None, None, None],
          full_output=False):
    '''
    Solves component molar fractions in each phase and either
    temperature or pressure in vapor-liquid-liquid equilibrium (VLLE)
    of binary (two component) mixture: (T or P) -> (X, W, Y, and P or T)

    Parameters
    ----------
    X0 : array
        Initial guess molar fractions of liquid phase 1
    W0 : array
        Initial guess molar fractions of liquid phase 2
    Y0 : array
        Initial guess molar fractions of vapor phase
    P_T : float
        Absolute temperature [K] or pressure [bar] (see *spec*)
    T_P : float
        Absolute temperature [K] or pressure [bar] (see *spec*)
    spec: string
        'T' if T_P is temperature or 'P' if T_P is pressure.
    model : object
        Phase equilibrium model object
    v0 : list, optional
        Liquid phase 1 and 2 and vapor phase molar volume used as initial
        values to compute fugacities
    full_output: bool, optional
        Flag to return a dictionary of all calculation info

    Returns
    -------
    X : array
        Liquid phase 1 molar fractions
    W : array
        Liquid phase 2 molar fractions
    Y : array
        Vapor phase molar fractions
    var: float
        Temperature [K] or pressure [bar], opposite of *spec*

    '''

    nc = model.nc
    if nc != 2:
        raise Exception('vlleb() requires a binary mixture')

    if len(X0) != nc or len(W0) != nc or len(Y0) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    global vx, vw, vy
    vx, vw, vy = v0
    '''
    sol1 = root(haz_objb, np.hstack([X0, W0, Y0, P_T]),
                args=(T_P, spec, model, v0))
    '''
    if spec == 'T':
        temp_aux = model.temperature_aux(T_P)
        sol1 = root(haz_objb, np.hstack([X0, W0, Y0, P_T]),
                    args=(temp_aux, spec, model))
    elif spec == 'P':
        sol1 = root(haz_objb, np.hstack([X0, W0, Y0, P_T]),
                    args=(T_P, spec, model))
    else:
        raise Exception('Specification not known')

    error = np.linalg.norm(sol1.fun)
    nfev = sol1.nfev
    sol = sol1.x
    if np.any(sol < 0):
        raise Exception('composition, T or P is negative')
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
