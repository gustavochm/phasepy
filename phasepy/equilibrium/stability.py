from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import minimize


def tpd(X, state, Z, T, P, model, v0=[None, None]):
    """
    Michelsen's Adimentional tangent plane function

    tpd (x, state, z, T, P, model)


    Parameters
    ----------
    X : array_like
        mole fraction array of trial fase
    state : string
        'L' for liquid phase, 'V' for vapour phase
    Z : array_like
        mole fraction array of overall mixture
    T :  float
        absolute temperature, in K
    P:  float
        absolute pressure in bar
    model : object
        create from mixture, eos and mixrule
    v0 : list, optional
        values to solve fugacity, if supplied

    Returns
    -------
    tpd: float
        tpd distance

    """
    v1, v2 = v0
    logfugX, v1 = model.logfugef(X, T, P, state, v1)
    logfugZ, v2 = model.logfugef(Z, T, P, 'L', v2)
    di = np.log(Z) + logfugZ
    tpdi = X*(np.log(X) + logfugX - di)
    return np.sum(np.nan_to_num(tpdi))


def tpd_obj(a, T, P, di, model, state, v0):

    W = a**2/4.  # change from alpha to mole numbers
    w = W/W.sum()  # change to mole fraction

    logfugW, _ = model.logfugef(w, T, P, state, v0)

    dtpd = np.log(W) + logfugW - di
    tpdi = np.nan_to_num(W*(dtpd-1.))
    tpd = 1. + tpdi.sum()
    dtpd *= a/2
    return tpd, dtpd


def tpd_min(W, Z, T, P, model, stateW, stateZ, vw=None, vz=None):
    """

    Found a minimiun of Michelsen's Adimentional tangent plane function

    tpd_min (W, Z, T, P, model, stateW, stateZ)

    Parameters
    ----------
    W : array_like
        mole fraction array of trial fase
    Z : array_like
        mole fraction array of overall mixture
    T :  absolute temperature, in K
    P:  absolute pressure in bar
    model : object create from mixture, eos and mixrule
    stateW : string
        'L' for liquid phase, 'V' for vapor phase
    stateZ : string
        'L' for liquid phase, 'V' for vapor phase
    vw, vz: float, optional
        initial volume value to compute fugacities of phases

    Returns
    -------
    w : array_like
        molar fraction of minimum
    f : float
        minimized tpd distance

    """
    nc = model.nc
    if len(W) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')
    # Fugacity of phase Zl
    Z[Z < 1e-8] = 1e-8
    logfugZ, vz = model.logfugef(Z, T, P, stateZ, vz)
    di = np.log(Z) + logfugZ

    alpha0 = 2*W**0.5
    alpha0[alpha0 < 1e-8] = 1e-8  # To avoid negative compositions

    alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw),
                     jac=True, method='BFGS')

    W = alpha.x**2/2
    w = W/W.sum()
    tpd = alpha.fun
    return w, tpd


def tpd_minimas(nmin, Z, T, P, model, stateW, stateZ, vw=None, vz=None):
    """

    Found nmin minimuns of Michelsen's Adimentional tangent plane function

    tpd_minimas (nmin, Z, T, P, model, stateW, stateZ)

    Parameters
    ----------
    nmin: int
        number of minimiuns to be founded
    Z : array_like
        mole fraction array of overall mixture
    T : float
        absolute temperature, in K
    P:  float
        absolute pressure in bar
    model : object
        create from mixture, eos and mixrule
    stateW : string
        'L' for liquid phase, 'V' for vapour phase
    stateZ : string
        'L' for liquid phase, 'V' for vapour phase
    vw, vz : float, optional
        if supplied volume used as initial value to compute fugacities

    Returns
    -------
    all_minima: tuple
        molar fractions arrays of minimums
    f_minima: tuple
        minimized tpd distance

    """
    nc = model.nc
    if len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    Z[Z < 1e-8] = 1e-8
    logfugZ, vz = model.logfugef(Z, T, P, stateZ, vz)
    di = np.log(Z) + logfugZ

    nc = model.nc
    all_minima = []
    f_minima = []

    # search from pures
    Id = np.eye(nc)
    alpha0 = 2*Id[0]**0.5
    alpha0[alpha0 < 1e-5] = 1e-5  # no negative or zero compositions
    alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw),
                     jac=True, method='BFGS')
    W = alpha.x**2/4
    w = W/W.sum()  # normalized composition
    tpd = alpha.fun
    all_minima.append(w)
    f_minima.append(tpd)

    for i in range(1, nc):
        alpha0 = 2*Id[i]**0.5
        alpha0[alpha0 < 1e-5] = 1e-5
        alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw),
                         jac=True, method='BFGS')
        W = alpha.x**2/4
        w = W/W.sum()  # normalized composition
        tpd = alpha.fun
        if alpha.success:
            add = np.any(np.all(np.isclose(all_minima, w, atol=1e-3), axis=1))
            if not add:
                f_minima.append(tpd)
                all_minima.append(w)
                if len(f_minima) == nmin:
                    return tuple(all_minima), np.array(f_minima)

    # random seach
    niter = 0
    while len(f_minima) < nmin and niter < (nmin+1):
        niter += 1
        Al = np.random.rand(nc)
        Al = Al/np.sum(Al)
        alpha0 = 2*Al**0.5
        alpha0[alpha0 < 1e-5] = 1e-5
        alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw),
                         jac=True, method='BFGS')
        W = alpha.x**2/4
        w = W/W.sum()  # normalized composition
        tpd = alpha.fun
        if alpha.success:
            add = np.any(np.all(np.isclose(all_minima, w, atol=1e-3), axis=1))
            if not add:
                f_minima.append(tpd)
                all_minima.append(w)
                if len(f_minima) == nmin:
                    return tuple(all_minima), np.array(f_minima)

    while len(f_minima) < nmin:
        all_minima.append(all_minima[0])
        f_minima.append(f_minima[0])

    return tuple(all_minima), np.array(f_minima)


def lle_init(Z, T, P, model, v0=None):
    """
    Minimize tpd function to initiate ELL at fixed T and P.

    Parameters
    ----------
    z : array_like
        overall molar fraction array
    T : float
        absolute temperature in K
    P : float
        absolute pressure in bar
    model : object
        created from eos and mixture
    v0 : float, optional
        if supplied volume used as initial value to compute fugacities

    Returns
    -------
    x0s: tuple
        Contains two mol fractions arrays
    """
    x0s, tpd0 = tpd_minimas(2, Z, T, P, model, 'L', 'L', v0, v0)
    return x0s


def gmix(X, T, P, state, lnphi0, model, v0=None):
    lnphi, v = model.logfugmix(X, T, P, state, v0)
    gmix = lnphi
    gmix -= np.sum(X*lnphi0)
    gmix += np.sum(np.nan_to_num(X*np.log(X)))
    return gmix
