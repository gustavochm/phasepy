from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import fsolve, root
from .multiflash import multiflash
from .equilibriumresult import EquilibriumResult
from warnings import warn


def haz_objb(inc, T_P, type, model, index, equilibrium):

    X0 = inc[:-1].reshape(3, 2)
    P_T = inc[-1]

    if type == 'T':
        P = P_T
        # T = T_P
        temp_aux = T_P
    elif type == 'P':
        T = P_T
        temp_aux = model.temperature_aux(T)
        P = T_P

    nc = model.nc
    X = np.zeros((3, nc))
    X[:, index] = X0
    lnphi = np.zeros_like(X)

    global vg

    for i, state in enumerate(equilibrium):
        lnphi[i], vg[i] = model.logfugef_aux(X[i], temp_aux, P, state, vg[i])

    lnK = lnphi[0] - lnphi[1:]
    K = np.exp(lnK)
    return np.hstack([(K[:, index]*X0[0]-X0[1:]).flatten(), X0.sum(axis=1)-1.])

'''
def haz_pb(X0, P_T, T_P, tipo, modelo, index, equilibrio,
           v0=[None, None, None]):
    sol = fsolve(haz_objb, np.hstack([X0.flatten(), P_T]),
                 args=(T_P, tipo, modelo, index, equilibrio, v0))

    var = sol[-1]
    X = sol[:-1].reshape(3, 2)

    return X, var
'''


def haz_objt(inc, temp_aux, P, model):

    X, W, Y = np.split(inc, 3)

    global vg

    fugX, vg[0] = model.logfugef_aux(X, temp_aux, P, 'L', vg[0])
    fugW, vg[1] = model.logfugef_aux(W, temp_aux, P, 'L', vg[1])
    fugY, vg[2] = model.logfugef_aux(Y, temp_aux, P, 'V', vg[2])

    K1 = np.exp(fugX-fugY)
    K2 = np.exp(fugX-fugW)

    return np.hstack([K1*X-Y, K2*X-W, X.sum()-1, Y.sum()-1, W.sum()-1])


def haz(X0, W0, Y0, T, P, model, good_initial=False,
        v0=[None, None, None], K_tol=1e-10, nacc=5, full_output=False):
    """
    Liquid liquid vapor (T,P) -> (x, w, y)

    Computes liquid liquid vapor equilibrium of multicomponent mixtures at
    given temperature. This functions uses a simultaneous method to check
    stability and equilibrium, when slow convergence is noted, minimization of
    Gibbs free energy is performed with BFGS.

    Parameters
    ----------

    X0 : array_like
         guess composition of liquid 1
    W0 : array_like
         guess composition of liquid 2
    Y0 : array_like
         guess composition of vapour 1
    T : float
        absolute temperature in K.
    P : float
        pressure in bar
    model : object
        Created from mixture, eos and mixrule
    good_initial: bool, optional
        if True skip Gupta's method and solves full system of equations.
    v0 : list, optional
         if supplied volume used as initial value to compute fugacities
    K_tol : float, optional
            Desired accuracy of K (= X/Xr) vector
    nacc : int, optional
        number of accelerated successive substitution cycles to perform
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

    """

    nc = model.nc
    if len(X0) != nc or len(W0) != nc or len(Y0) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    Z0 = (X0+Y0+W0)/3
    nonzero = np.count_nonzero(Z0)
    x0 = np.vstack([X0, W0, Y0])
    b0 = np.array([0.33, 0.33, 0.33, 0., 0.])

    global vg
    vg = v0.copy()

    # check for binary mixture
    if nonzero == 2:
        warn('Global mixture is a binary mixture, updating temperature')
        index = np.nonzero(Z0)[0]
        sol = np.zeros_like(x0)

        inc0 = np.hstack([x0[:, index].flatten(), T])
        sol1 = root(haz_objb, inc0, args=(P, 'P', model, index, 'LLV'))
        T = sol1.x[-1]
        Xs = sol1.x[:-1].reshape(3, 2)
        sol[:, index] = Xs
        X, W, Y = sol

        if full_output:
            info = {'T': T, 'P': P, 'X': sol, 'v': vg,
                    'states': ['L', 'L', 'V']}
            out = EquilibriumResult(info)
            return out
        return X, W, Y, T

    if not good_initial:
        out = multiflash(x0, b0, ['L', 'L', 'V'], Z0, T, P, model, vg,
                         K_tol, nacc, True)
    else:
        temp_aux = model.temperature_aux(T)
        sol = fsolve(haz_objt, x0.flatten(), args=(temp_aux, P, model))
        x0 = sol.reshape([model.nc, 3])
        Z0 = x0.mean(axis=0)
        out = multiflash(x0, b0, ['L', 'L', 'V'], Z0, T, P, model,
                         vg, K_tol, nacc, True)

    Xm, beta, tetha, equilibrio = out.X, out.beta, out.tetha, out.states
    error_inner = out.error_inner
    v = out.v

    if error_inner > 1e-6:
        order = [2, 0, 1]  # Y, X, W
        Xm = Xm[order]
        betatetha = np.hstack([beta[order], tetha])
        equilibrio = np.asarray(equilibrio)[order]
        v0 = np.asarray(v)[order]
        out = multiflash(Xm, betatetha, equilibrio, Z0, T, P, model, v0,
                         K_tol, nacc, full_output=True)
        order = [1, 2, 0]
        Xm, beta, tetha, equilibrio = out.X, out.beta, out.tetha, out.states
        error_inner = out.error_inner
        if error_inner > 1e-6:
            order = [2, 1, 0]  # W, X, Y
            Xm = Xm[order]
            betatetha = np.hstack([beta[order], tetha])
            equilibrio = np.asarray(equilibrio)[order]
            v0 = np.asarray(out.v)[order]
            out = multiflash(Xm, betatetha, equilibrio, Z0, T, P, model, v0,
                             K_tol, nacc, full_output=True)
            order = [1, 0, 2]
            Xm, beta, tetha = out.X, out.beta, out.tetha
            equilibrio = out.states
            error_inner = out.error_inner
        Xm = Xm[order]
        beta = beta[order]
        tetha = np.hstack([0., tetha])
        tetha = tetha[order]
        v = (out.v)[order]
    else:
        tetha = np.hstack([0., tetha])

    if full_output:
        info = {'T': T, 'P': P, 'error_outer': out.error_outer,
                'error_inner': error_inner, 'iter': out.iter,
                'beta': beta, 'tetha': tetha, 'X': Xm, 'v': v,
                'states': ['L', 'L', 'V']}
        out = EquilibriumResult(info)
        return out

    tethainestable = tetha > 0.
    Xm[tethainestable] = None
    X, W, Y = Xm

    return X, W, Y


def vlle(X0, W0, Y0, Z, T, P, model, v0=[None, None, None], K_tol=1e-10,
         nacc=5, full_output=False):
    """
    Solves liquid-liquid-vapor equilibrium (VLLE) multicomponent flash:
    (Z, T, P) -> (X, W, Y)

    Parameters
    ----------
    X0 : array
        Initial guess molar fractions of liquid phase 1
    W0 : array
        Initial guess molar fractions of liquid phase 2
    Y0 : array
        Initial guess molar fractions of vapor phase
    T : float
        Absolute temperature [K]
    P : float
        Pressure [bar]
    model : object
        Phase equilibrium model object
    good_initial: bool, optional
        if True skip Gupta's method and solve full system of equations
    v0 : list, optional
        Liquid phase 1 and 2 and vapor phase molar volume used as initial
        values to compute fugacities
    K_tol : float, optional
        Tolerance for equilibrium constant values
    nacc : int, optional
        number of accelerated successive substitution cycles to perform
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
    """

    nc = model.nc
    if len(X0) != nc or len(W0) != nc or len(Y0) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    nonzero = np.count_nonzero(Z)
    x0 = np.vstack([X0, W0, Y0])
    b0 = np.array([0.33, 0.33, 0.33, 0., 0.])

    # check for binary mixture
    if nonzero == 2:
        warn('Global mixture is a binary mixture, updating temperature')
        index = np.nonzero(Z)[0]
        sol = np.zeros_like(x0)
        global vg
        vg = v0.copy()
        inc0 = np.hstack([x0[:, index].flatten(), T])
        sol1 = root(haz_objb, inc0, args=(P, 'P', model, index, 'LLV'))
        T = sol1.x[-1]
        Xs = sol1.x[:-1].reshape(3, 2)
        sol[:, index] = Xs
        X, W, Y = sol
        if full_output:
            info = {'T': T, 'P': P, 'X': sol, 'v': vg,
                    'states': ['L', 'L', 'V']}
            out = EquilibriumResult(info)
            return out
        return X, W, Y, T

    out = multiflash(x0, b0, ['L', 'L', 'V'], Z, T, P, model, v0, K_tol,
                     nacc, True)

    Xm, beta, tetha, equilibrio = out.X, out.beta, out.tetha, out.states
    error_inner = out.error_inner
    v = out.v

    if error_inner > 1e-6:
        order = [2, 0, 1]  # Y, X, W
        Xm = Xm[order]
        betatetha = np.hstack([beta[order], tetha])
        equilibrio = np.asarray(equilibrio)[order]
        v0 = np.asarray(v)[order]
        out = multiflash(Xm, betatetha, equilibrio, Z, T, P, model, v0,
                         K_tol, nacc, full_output=True)
        order = [1, 2, 0]
        Xm, beta, tetha, equilibrio = out.X, out.beta, out.tetha, out.states
        error_inner = out.error_inner
        if error_inner > 1e-6:
            order = [2, 1, 0]  # W, X, Y
            Xm = Xm[order]
            betatetha = np.hstack([beta[order], tetha])
            equilibrio = np.asarray(equilibrio)[order]
            v0 = np.asarray(out.v)[order]
            out = multiflash(Xm, betatetha, equilibrio, Z, T, P, model, v0,
                             K_tol, nacc, full_output=True)
            order = [1, 0, 2]
            Xm, beta, tetha = out.X, out.beta, out.tetha
            equilibrio = out.states
            error_inner = out.error_inner
        Xm = Xm[order]
        beta = beta[order]
        tetha = np.hstack([0., tetha])
        tetha = tetha[order]
        v = (out.v)[order]
    else:
        tetha = np.hstack([0., tetha])

    if full_output:
        info = {'T': T, 'P': P, 'error_outer': out.error_outer,
                'error_inner': error_inner, 'iter': out.iter, 'beta': beta,
                'tetha': tetha, 'X': Xm, 'v': v, 'states': ['L', 'L', 'V']}
        out = EquilibriumResult(info)
        return out

    tethainestable = tetha > 0.
    Xm[tethainestable] = None
    X, W, Y = Xm

    return X, W, Y


__all__ = ['haz', 'vlle']
