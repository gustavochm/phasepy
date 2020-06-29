from __future__ import division, print_function, absolute_import
import numpy as np
from .stability import tpd_minimas
from .multiflash import multiflash


def lle(x0, w0, Z, T, P, model, v0=[None, None],
        K_tol=1e-8, full_output=False):
    """
    Liquid liquid equilibrium (z,T,P) -> (x,w,beta)

    Solves liquid liquid equilibrium from multicomponent mixtures at given
    pressure, temperature and overall composition.

    Parameters
    ----------

    x0 : array_like
        initial guess for liquid phase 1
    w0 : array_like
        initial guess for liquid phase 2
    z : array_like
        overal composition of mix
    T : float
        absolute temperature in K.
    P : float
        pressure in en bar
    model : object
        created from mixture, eos and mixrule
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    K_tol : float, optional
        Desired accuracy of K (= W/X) vector
    full_output: bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    X : array_like
        liquid 1 mole fraction vector
    W : array_like
        liquid 2 mole fraction vector
    beta : float
        phase fraction of liquid 2

    """
    nc = model.nc
    if len(x0) != nc or len(w0) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    '''
    equilibrio = ['L', 'L']
    out = flash(x0, w0, equilibrio, Z, T, P, model, v0, K_tol , True)
    X, W, beta = out.X, out.Y, out.beta
    v1, v2 = out.v1, out.v2
    '''

    equilibrio = ['L', 'L']
    fugx, v1 = model.logfugef(x0, T, P, equilibrio[0], v0[0])
    fugw, v2 = model.logfugef(w0, T, P, equilibrio[1], v0[1])
    lnK = fugx - fugw
    K = np.exp(lnK)

    bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], 0.]))
    bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
    beta = (bmin + bmax)/2
    X0 = np.array([x0, w0])

    beta0 = np.array([1-beta, beta, 0.])

    out = multiflash(X0, beta0, equilibrio, Z, T, P, model,
                     [v1, v2], K_tol, True)
    Xm, beta, tetha, v = out.X, out.beta, out.tetha, out.v

    if tetha > 0:
        xes, tpd_min2 = tpd_minimas(2, Xm[0], T, P, model, 'L', 'L',
                                    v[0], v[0])
        X0 = np.asarray(xes)
        beta0 = np.hstack([beta, 0.])
        out = multiflash(X0, beta0, equilibrio, Z, T, P, model, v, K_tol, True)
        Xm, beta, tetha, v = out.X, out.beta, out.tetha, out.v

    X, W = Xm
    if tetha > 0:
        W = X.copy()

    if full_output:
        return out

    return X, W, beta[1]


__all__ = ['lle']
