from __future__ import division, print_function, absolute_import
import numpy as np
from ..equilibrium import haz
from ..actmodels import virialgamma
from .fitmulticomponent import fobj_elv, fobj_ell, fobj_hazt


def fobj_nrtlrkt(D, Xexp, Wexp, Yexp, Texp, Pexp, mezcla, good_initial=True):

    n = len(Pexp)
    mezcla.rkt(D)

    vg = virialgamma(mezcla, actmodel='nrtlter')
    x = np.zeros_like(Xexp)
    w = np.zeros_like(Wexp)
    y = np.zeros_like(Yexp)

    for i in range(n):
        x[:, i], w[:, i], y[:, i] = haz(Xexp[:, i], Wexp[:, i], Yexp[:, i],
                                        Texp[i], Pexp[i], vg, good_initial)

    error = ((np.nan_to_num(x)-Xexp)**2).sum()
    error += ((np.nan_to_num(w)-Wexp)**2).sum()
    error += ((np.nan_to_num(y)-Yexp)**2).sum()

    return error


def fobj_nrtlt(inc, mezcla, datavle=None, datalle=None, datavlle=None,
               alpha_fixed=False, Tdep=False):

    if alpha_fixed:
        a12 = a13 = a23 = 0.2
        if Tdep:
            g12, g21, g13, g31, g23, g32, g12T, g21T, g13T, g31T, g23T, g32T = inc
            gT = np.array([[0, g12T, g13T],
                           [g21T, 0, g23T],
                           [g31T, g32T, 0]])
        else:
            g12, g21, g13, g31, g23, g32 = inc
            gT = None
    else:
        if Tdep:
            g12, g21, g13, g31, g23, g32, g12T, g21T, g13T, g31T, g23T, g32T, a12, a13, a23 = inc
            gT = np.array([[0, g12T, g13T],
                           [g21T, 0, g23T],
                           [g31T, g32T, 0]])
        else:
            g12, g21, g13, g31, g23, g32, a12, a13, a23 = inc
            gT = None

    g = np.array([[0, g12, g13],
                  [g21, 0, g23],
                  [g31, g32, 0]])

    alpha = np.array([[0, a12, a13],
                      [a12, 0, a23],
                      [a13, a23, 0]])

    mezcla.NRTL(alpha, g, gT)
    modelo = virialgamma(mezcla)

    error = 0

    if datavle is not None:
        error += fobj_elv(modelo, *datavle)
    if datalle is not None:
        error += fobj_ell(modelo, *datalle)
    if datavlle is not None:
        error += fobj_hazt(modelo, *datavlle)
    return error


def fobj_kijt(inc, eos, mezcla, datavle=None, datalle=None, datavlle=None):

    k12, k13, k23 = inc
    Kij = np.array([[0, k12, k13],
                   [k12, 0, k23],
                   [k13, k23, 0]])
    mezcla.kij_cubica(Kij)
    modelo = eos(mezcla)

    error = 0

    if datavle is not None:
        error += fobj_elv(modelo, *datavle)
    if datalle is not None:
        error += fobj_ell(modelo, *datalle)
    if datavlle is not None:
        error += fobj_hazt(modelo, *datavlle)
    return error
