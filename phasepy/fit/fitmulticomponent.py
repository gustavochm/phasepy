from __future__ import division, print_function, absolute_import
import numpy as np
from ..equilibrium import bubblePy, lle, tpd_min, vlleb, haz


def fobj_vle(model, Xexp, Yexp, Texp, Pexp, weights_vle=[1., 1.]):
    """
    Objective function to fit parameters for VLE in multicomponent mixtures
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

    error = weights_vle[0] * np.sum((Y-Yexp)**2)
    error += weights_vle[1] * np.sum((P/Pexp-1)**2)
    error /= n
    return error


def fobj_lle(model, Xexp, Wexp, Texp, Pexp, tpd=True, weights_lle=[1., 1.]):
    """
    Objective function to fit parameters for LLE in multicomponent mixtures
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

    error = weights_lle[0] * np.sum((X-Xexp)**2)
    error += weights_lle[1] * np.sum((W-Wexp)**2)
    error /= n
    return error


def fobj_vlleb(model, Xellv, Wellv, Yellv, Tellv, Pellv,
               weights_vlleb=[1., 1., 1., 1.]):
    """
    Objective function to fit parameters for VLLE in binary mixtures
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

    error = weights_vlleb[0] * np.sum((X-Xellv)**2)
    error += weights_vlleb[1] * np.sum((W-Wellv)**2)
    error += weights_vlleb[2] * np.sum((Y-Yellv)**2)
    error += weights_vlleb[3] * np.sum((P/Pellv-1)**2)
    error /= n

    return error


def fobj_vllet(model, Xellv, Wellv, Yellv, Tellv, Pellv,
               weights_vlle=[1., 1., 1.]):
    """
    Objective function to fit parameters for VLLE in multicomponent mixtures
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

    error += weights_vlle[0] * np.sum((np.nan_to_num(X)-Xellv)**2)
    error += weights_vlle[1] * np.sum((np.nan_to_num(Y)-Yellv)**2)
    error += weights_vlle[2] * np.sum((np.nan_to_num(W)-Wellv)**2)
    error /= n
    return error
