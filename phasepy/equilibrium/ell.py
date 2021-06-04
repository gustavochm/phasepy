from __future__ import division, print_function, absolute_import
import numpy as np
from .stability import tpd_minimas
from .multiflash import multiflash


def lle(x0, w0, Z, T, P, model, v0=[None, None],
        K_tol=1e-8, nacc=5, full_output=False):
    """
    Isobaric isothermic (PT) flash for multicomponent liquid-liquid
    systems: (Z, T, P) -> (x, w, beta)

    Parameters
    ----------
    x0 : array
        Initial guess for molar fractions of liquid phase 1
    w0 : array
        Initial guess for molar fractions of liquid phase 2
    Z : array
        Overall molar fractions of components
    T : float
        Absolute temperature [K]
    P : float
        Pressure [bar]
    model : object
        Phase equilibrium model object
    v0 : list, optional
        Liquid phase 1 and 2 molar volumes used as initial values to compute
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
    w : array
        Phase 2 molar fractions of components
    beta : float
        Phase 2 phase fraction
    """
    nc = model.nc
    if len(x0) != nc or len(w0) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    equilibrium = ['L', 'L']

    temp_aux = model.temperature_aux(T)

    fugx, v1 = model.logfugef_aux(x0, temp_aux, P, equilibrium[0], v0[0])
    fugw, v2 = model.logfugef_aux(w0, temp_aux, P, equilibrium[1], v0[1])
    lnK = fugx - fugw
    K = np.exp(lnK)

    bmin = max(np.hstack([((K*Z-1.)/(K-1.))[K > 1], 0.]))
    bmax = min(np.hstack([((1.-Z)/(1.-K))[K < 1], 1.]))
    beta = (bmin + bmax)/2
    X0 = np.array([x0, w0])

    beta0 = np.array([1-beta, beta, 0.])

    out = multiflash(X0, beta0, equilibrium, Z, T, P, model,
                     [v1, v2], K_tol, nacc, True)
    Xm, beta, tetha, v = out.X, out.beta, out.tetha, out.v

    if tetha > 0:
        xes, tpd_min2 = tpd_minimas(2, Xm[0], T, P, model, 'L', 'L',
                                    v[0], v[0])
        X0 = np.asarray(xes)
        beta0 = np.hstack([beta, 0.])
        out = multiflash(X0, beta0, equilibrium, Z, T, P, model, v, K_tol,
                         nacc, True)
        Xm, beta, tetha, v = out.X, out.beta, out.tetha, out.v

    X, W = Xm
    if tetha > 0:
        W = X.copy()

    if full_output:
        return out

    return X, W, beta[1]


__all__ = ['lle']
