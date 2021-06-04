from __future__ import division, print_function, absolute_import
import numpy as np
from ..math import gdem
from scipy.optimize import minimize
from .equilibriumresult import EquilibriumResult


def rachfordrice(beta, K, Z):
    '''
    Solves Rachford Rice equation by Halley's method
    '''
    K1 = K-1.
    g0 = np.dot(Z, K) - 1.
    g1 = 1. - np.dot(Z, 1/K)
    singlephase = False

    if g0 < 0:
        beta = 0.
        D = np.ones_like(Z)
        singlephase = True
    elif g1 > 0:
        beta = 1.
        D = 1 + K1
        singlephase = True
    it = 0
    e = 1.
    while e > 1e-8 and it < 20 and not singlephase:
        it += 1
        D = 1 + beta*K1
        KD = K1/D
        fo = np.dot(Z, KD)
        dfo = - np.dot(Z, KD**2)
        d2fo = 2*np.dot(Z, KD**3)
        dbeta = - (2*fo*dfo)/(2*dfo**2-fo*d2fo)
        beta += dbeta
        e = np.abs(dbeta)

    return beta, D, singlephase


def Gibbs_obj(v, phases, Z, temp_aux, P, model):
    '''
    Objective function to minimize Gibbs energy in biphasic flash
    '''
    l = Z-v
    v[v < 1e-8] = 1e-8
    l[l < 1e-8] = 1e-8
    X = l/l.sum()
    Y = v/v.sum()
    global v1, v2
    lnfugl, v1 = model.logfugef_aux(X, temp_aux, P, phases[0], v1)
    lnfugv, v2 = model.logfugef_aux(Y, temp_aux, P, phases[1], v2)
    fugl = np.log(X) + lnfugl
    fugv = np.log(Y) + lnfugv
    fo = v*fugv + l*fugl
    f = np.sum(fo)
    df = fugv - fugl
    return f, df


def dGibbs_obj(v, phases, Z, temp_aux, P, model):
    '''
    Objective function to minimize Gibbs energy in biphasic flash when second
    order derivatives are available
    '''

    l = Z - v
    v[v < 1e-8] = 1e-8
    l[l < 1e-8] = 1e-8
    vt = np.sum(v)
    lt = np.sum(l)
    X = l/lt
    Y = v/vt
    nc = len(l)
    eye = np.eye(nc)

    global v1, v2
    lnfugl, dlnfugl, v1 = model.dlogfugef_aux(X, temp_aux, P, phases[0], v1)
    lnfugv, dlnfugv, v2 = model.dlogfugef_aux(Y, temp_aux, P, phases[1], v2)

    fugl = np.log(X) + lnfugl
    fugv = np.log(Y) + lnfugv
    fo = v*fugv + l*fugl
    f = np.sum(fo)
    df = fugv - fugl

    global dfugv, dfugl
    dfugv = eye/v - 1/vt + dlnfugv/vt
    dfugl = eye/l - 1/lt + dlnfugl/lt

    return f, df


def dGibbs_hess(v, phases, Z, temp_aux, P, model):
    '''
    Hessian to minimize Gibbs energy in biphasic flash when second
    order derivatives are available
    '''
    global dfugv, dfugl
    d2fo = dfugv + dfugl
    return d2fo


def flash(x_guess, y_guess, equilibrium, Z, T, P, model,
          v0=[None, None], K_tol=1e-8, nacc=5, full_output=False):
    """
    Isobaric isothermic (PT) flash: (Z, T, P) -> (x, y, beta)

    Parameters
    ----------
    x_guess : array
        Initial guess for molar fractions of phase 1 (liquid)
    y_guess : array
        Initial guess for molar fractions of phase 2 (gas or liquid)
    equilibrium : string
        Two-phase system definition: 'LL' (liquid-liquid) or
        'LV' (liquid-vapor)
    Z : array
        Overall molar fractions of components
    T : float
        Absolute temperature [K]
    P : float
        Pressure [bar]
    model : object
        Phase equilibrium model object
    v0 : list, optional
        Liquid and vapor phase molar volume used as initial values to compute
        fugacities
    K_tol : float, optional
        Tolerance for equilibrium constant values
    nacc : int, optional
        number of accelerated successive substitution cycles to perform
    full_output: bool, optional
        Flag to return a dictionary of all calculation info

    Returns
    -------
    x : array
        Phase 1 molar fractions of components
    y : array
        Phase 2 molar fractions of components
    beta : float
        Phase 2 phase fraction
    """
    nc = model.nc
    if len(x_guess) != nc or len(y_guess) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    temp_aux = model.temperature_aux(T)

    v10, v20 = v0

    e1 = 1
    itacc = 0
    it = 0
    it2 = 0
    n = 5
    X = x_guess
    Y = y_guess

    global v1, v2
    fugl, v1 = model.logfugef_aux(X, temp_aux, P, equilibrium[0], v10)
    fugv, v2 = model.logfugef_aux(Y, temp_aux, P, equilibrium[1], v20)
    lnK = fugl - fugv
    K = np.exp(lnK)

    bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], 0.]))
    bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
    beta = (bmin + bmax)/2

    while e1 > K_tol and itacc < nacc:
        it += 1
        it2 += 1
        lnK_old = lnK
        beta, D, singlephase = rachfordrice(beta, K, Z)

        X = Z/D
        Y = X*K
        X /= X.sum()
        Y /= Y.sum()
        fugl, v1 = model.logfugef_aux(X, temp_aux, P, equilibrium[0], v1)
        fugv, v2 = model.logfugef_aux(Y, temp_aux, P, equilibrium[1], v2)

        lnK = fugl-fugv
        if it == (n-3):
            lnK3 = lnK
        elif it == (n-2):
            lnK2 = lnK
        elif it == (n-1):
            lnK1 = lnK
        elif it == n:
            it = 0
            itacc += 1
            dacc = gdem(lnK, lnK1, lnK2, lnK3)
            lnK += dacc
        K = np.exp(lnK)
        e1 = ((lnK-lnK_old)**2).sum()

    if e1 > K_tol and itacc == nacc and not singlephase:
        if model.secondorder:
            fobj = dGibbs_obj
            jac = True
            hess = dGibbs_hess
            method = 'trust-ncg'
        else:
            fobj = Gibbs_obj
            jac = True
            hess = None
            method = 'BFGS'

        vsol = minimize(fobj, beta*Y, jac=jac, method=method, hess=hess,
                        tol=K_tol, args=(equilibrium, Z, temp_aux, P, model))

        it2 += vsol.nit
        e1 = np.linalg.norm(vsol.jac)
        v = vsol.x
        l = Z - v
        beta = v.sum()
        v[v <= 1e-8] = 0
        l[l <= 1e-8] = 0
        Y = v / beta
        Y /= Y.sum()
        X = l/l.sum()

    if beta == 1.0:
        X = Y.copy()
    elif beta == 0.:
        Y = X.copy()

    if full_output:
        sol = {'T': T, 'P': P, 'beta': beta, 'error': e1, 'iter': it2,
               'X': X, 'v1': v1, 'state1': equilibrium[0],
               'Y': Y, 'v2': v2, 'state2': equilibrium[1]}
        out = EquilibriumResult(sol)
        return out

    return X, Y, beta
