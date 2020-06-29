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


def Gibbs_obj(v, phases, Z, T, P, model, v10, v20):
    '''
    Objective function to minimize Gibbs energy in biphasic flash
    '''
    l = Z-v
    v[v < 1e-8] = 1e-8
    l[l < 1e-8] = 1e-8
    X = l/l.sum()
    Y = v/v.sum()

    lnfugl, v1 = model.logfugef(X, T, P, phases[0], v10)
    lnfugv, v2 = model.logfugef(Y, T, P, phases[1], v20)
    fugl = np.log(X) + lnfugl
    fugv = np.log(Y) + lnfugv
    fo = v*fugv + l*fugl
    f = np.sum(fo)
    df = fugv - fugl
    return f, df


def dGibbs_obj(v, phases, Z, T, P, model, v10, v20):
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

    lnfugl, dlnfugl, v1 = model.dlogfugef(X, T, P, phases[0], v10)
    lnfugv, dlnfugv, v2 = model.dlogfugef(Y, T, P, phases[1], v20)

    fugl = np.log(X) + lnfugl
    fugv = np.log(Y) + lnfugv
    fo = v*fugv + l*fugl
    f = np.sum(fo)
    df = fugv - fugl

    global dfugv, dfugl
    dfugv = eye/v - 1/vt + dlnfugv/vt
    dfugl = eye/l - 1/lt + dlnfugl/lt

    return f, df


def dGibbs_hess(v, phases, Z, T, P, model, v10, v20):
    '''
    Hessian to minimize Gibbs energy in biphasic flash when second
    order derivatives are available
    '''
    global dfugv, dfugl
    d2fo = dfugv + dfugl
    return d2fo


def flash(x_guess, y_guess, equilibrium, Z, T, P, model,
          v0=[None, None], K_tol=1e-8, full_output=False):
    """
    Isothermic isobaric flash (z,T,P) -> (x,y,beta)

    Parameters
    ----------

    x_guess : array_like
        guess composition of phase 1
    y_guess : array_like
        guess composition of phase 2
    equilibrium : string
        'LL' for ELL, 'LV' for ELV
    z : array_like
        overall system composition
    T : float
        absolute temperature in K.
    P : float
        pressure in bar

    model : object
        created from mixture, eos and mixrule
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    K_tol : float, optional
        Desired accuracy of K (= Y/X) vector
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    X : array_like
        phase 1 composition
    Y : array_like
        phase 2 composition
    beta : float
        phase 2 phase fraction
    """
    nc = model.nc
    if len(x_guess) != nc or len(y_guess) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    v10, v20 = v0

    e1 = 1
    itacc = 0
    it = 0
    it2 = 0
    n = 4
    nacc = 3
    X = x_guess
    Y = y_guess

    fugl, v1 = model.logfugef(X, T, P, equilibrium[0], v10)
    fugv, v2 = model.logfugef(Y, T, P, equilibrium[1], v20)
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
        fugl, v1 = model.logfugef(X, T, P, equilibrium[0], v1)
        fugv, v2 = model.logfugef(Y, T, P, equilibrium[1], v2)

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

        vsol = minimize(fobj, beta*Y, args=(equilibrium, Z, T, P, model,
                        v1, v2), jac=jac, method=method, hess=hess, tol=K_tol)

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
