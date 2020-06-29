from __future__ import division, print_function, absolute_import
import numpy as np
from ..equilibrium import bubblePy, lle, tpd_min, vlleb, haz


def fobj_elv(model, Xexp, Yexp, Texp, Pexp):
    """
    Objective function to fit parameters for ELV in multicomponent mixtures
    """

    n = Pexp.shape[0]
    n1, n2 = Xexp.shape
    if n2 == n:
        Xexp = Xexp.T
        Yexp = Yexp.T

    P = np.zeros(n)  # n,
    Y = np.zeros_like(Xexp)  # nc,n

    for i in range(n):
        Y[i], P[i] = bubblePy(Yexp[i], Pexp[i], Xexp[i], Texp[i], model)

    error = ((Y-Yexp)**2).sum()
    error += ((P/Pexp-1)**2).sum()
    error /= n
    return error


def fobj_ell(model, Xexp, Wexp, Texp, Pexp, tpd=True):
    """
    Objective function to fit parameters for ELL in multicomponent mixtures
    """
    n = Pexp.shape[0]
    n1, n2 = Xexp.shape
    if n2 == n:
        Xexp = Xexp.T
        Wexp = Wexp.T

    X = np.zeros_like(Xexp)
    W = np.zeros_like(Wexp)
    Z = (Xexp+Wexp)/2

    for i in range(n):
        if tpd:
            X0, tpd = tpd_min(Xexp[i], Z[i], Texp[i], Pexp[i], model, 'L', 'L')
            W0, tpd = tpd_min(Wexp[i], Z[i], Texp[i], Pexp[i], model, 'L', 'L')
        else:
            X0 = Xexp[i]
            W0 = Wexp[i]
        X[i], W[i], beta = lle(X0, W0, Z[i], Texp[i], Pexp[i], model)

    error = ((X-Xexp)**2).sum()
    error += ((W-Wexp)**2).sum()
    error /= n
    return error


def fobj_hazb(model, Xellv, Wellv, Yellv, Tellv, Pellv, info=[1, 1, 1]):
    """
    Objective function to fit parameters for ELLV in binary mixtures
    """
    n = len(Tellv)
    n1, n2 = Xellv.shape
    if n2 == n:
        Xellv = Xellv.T
        Wellv = Wellv.T
        Yellv = Yellv.T

    X = np.zeros_like(Xellv)
    W = np.zeros_like(Wellv)
    Y = np.zeros_like(Yellv)
    P = np.zeros(n)
    Zll = (Xellv + Wellv) / 2

    for i in range(n):
        try:
            X0, tpd = tpd_min(Xellv[i], Zll[i], Tellv[i], Pellv[i],
                              model, 'L', 'L')
            W0, tpd = tpd_min(Wellv[i], Zll[i], Tellv[i], Pellv[i],
                              model, 'L', 'L')
            X[i], W[i], Y[i], P[i] = vlleb(X0, W0, Yellv[i], Pellv[i],
                                           Tellv[i], 'T', model)
        except:
            pass

    error = info[0]*((X-Xellv)**2).sum()
    error += info[1]*((W-Wellv)**2).sum()
    error += info[2]*((Y-Yellv)**2).sum()
    error += ((P/Pellv-1)**2).sum()
    error /= n

    return error


def fobj_hazt(model, Xellv, Wellv, Yellv, Tellv, Pellv):
    """
    Objective function to fit parameters for ELLV in multicomponent mixtures
    """

    n = len(Tellv)
    n1, n2 = Xellv.shape
    if n2 == n:
        Xellv = Xellv.T
        Wellv = Wellv.T
        Yellv = Yellv.T

    X = np.zeros_like(Xellv)
    W = np.zeros_like(Wellv)
    Y = np.zeros_like(Yellv)

    error = 0
    for i in range(n):
        try:
            X[i], W[i], Y[i] = haz(Xellv[i], Wellv[i], Yellv[i],
                                   Tellv[i], Pellv[i], model, True)
        except ValueError:
            X[i], W[i], Y[i], T = haz(Xellv[i], Wellv[i], Yellv[i], Tellv[i],
                                      Pellv[i], model, True)
            error += (T/Tellv[i]-1)**2
        except:
            pass

    error += ((np.nan_to_num(X)-Xellv)**2).sum()
    error += ((np.nan_to_num(Y)-Yellv)**2).sum()
    error += ((np.nan_to_num(W)-Wellv)**2).sum()
    error /= n
    return error
