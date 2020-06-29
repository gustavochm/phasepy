from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import nrtl, wilson, nrtlter, rk, unifac
from ..actmodels import dnrtl, dwilson, dnrtlter, drk, dunifac
from ..constants import R


def ws(X, T, ai, bi, C, Kij, ActModel, parameter):

    # Acivity coefficient
    lngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama, X)

    RT = R*T
    # Mixrule parameters
    ei = ai/(bi*RT)
    biaiRT = bi - ai/RT
    abij = np.add.outer(biaiRT, biaiRT)/2
    abij *= (1. - Kij)
    xbi_ai = X*abij
    Q = np.sum(xbi_ai.T*X)
    # dQ = 2*np.sum(xbi_ai, axis=1)
    D = Gex/C + np.dot(X, ei)
    D1 = 1. - D
    # dD = ei + lngama/C

    # Mixture parameters
    bm = Q/D1
    am = bm*D*RT
    return am, bm


def dws(X, T, ai, bi, C, Kij, ActModel, parameter):

    # Acivity coefficient
    lngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama, X)

    RT = R*T
    # Mixrule parameters
    ei = ai/(bi*RT)
    biaiRT = bi - ai/RT
    abij = np.add.outer(biaiRT, biaiRT)/2
    abij *= (1. - Kij)
    xbi_ai = X*abij
    Q = np.sum(xbi_ai.T*X)
    dQ = 2*np.sum(xbi_ai, axis=1)
    D = Gex/C + np.dot(X, ei)
    D1 = 1. - D
    dD = ei + lngama/C

    # Mixture parameters
    bm = Q/D1
    am = bm*D*RT
    # Partial Molar properties
    Bi = dQ/D1 - Q/D1**2 * (1 - dD)
    Di = RT*(D*Bi + bm*dD)

    return am, Di, bm, Bi


def d2ws(X, T, ai, bi, C, Kij, ActModel, parameter):

    # Acivity coefficient
    lngama, dlngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama, X)

    RT = R*T
    # Mixrule parameters

    ei = ai/(bi*RT)
    biaiRT = bi - ai/RT
    abij = np.add.outer(biaiRT, biaiRT)/2
    xbi_ai = X*abij
    Q = np.sum(xbi_ai.T*X)
    D = Gex/C + np.dot(X, ei)
    D1 = 1. - D

    bm = Q/D1
    am = bm*D*RT

    dQ_dn = 2*np.sum(xbi_ai, axis=1)
    dD_dn = ei + lngama/C

    db_dn = dQ_dn/D1 - Q/D1**2 * (1 - dD_dn)
    da_dn = RT*(D*db_dn + bm*dD_dn)

    dQ_dnij = 2. * abij
    nc = len(X)
    dxidnj = np.eye(nc) - X
    dD_dnij = dlngama@dxidnj.T / C
    dQ_D1 = dQ_dn / D1**2 - 2*Q*(1-dD_dn) / D1**3

    db_dnij = np.outer(dQ_dn, -1./D1**2 * (1. - dD_dn))
    db_dnij += 1/D1 * dQ_dnij
    db_dnij += Q/D1**2 * dD_dnij
    db_dnij -= np.outer((1. - dD_dn), dQ_D1)

    da_dnij = np.outer(db_dn, dD_dn) + D*db_dnij
    da_dnij += np.outer(dD_dn, db_dn) + bm*dD_dnij
    da_dnij *= RT

    return am, da_dn, da_dnij, bm, db_dn, db_dnij


def ws_nrtl(X, T, ai, bi, order, C, Kij, alpha, g, g1):
    parameter = (alpha, g, g1)
    if order == 0:
        mixparameters = ws(X, T, ai, bi, C, Kij, nrtl, parameter)
    elif order == 1:
        mixparameters = dws(X, T, ai, bi, C, Kij, nrtl, parameter)
    elif order == 2:
        mixparameters = dws(X, T, ai, bi, C, Kij, dnrtl, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_nrtlt(X, T, ai, bi, order, C, Kij, alpha, g, g1, D):
    parameter = (alpha, g, g1, D)
    if order == 0:
        mixparameters = ws(X, T, ai, bi, C, Kij, nrtlter, parameter)
    elif order == 1:
        mixparameters = dws(X, T, ai, bi, C, Kij, nrtlter, parameter)
    elif order == 2:
        mixparameters = dws(X, T, ai, bi, C, Kij, dnrtlter, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_wilson(X, T, ai, bi, order, C, Kij, Aij, vl):
    parameter = (Aij, vl)
    if order == 0:
        mixparameters = ws(X, T, ai, bi, C, Kij, wilson, parameter)
    elif order == 1:
        mixparameters = dws(X, T, ai, bi, C, Kij, wilson, parameter)
    elif order == 2:
        mixparameters = dws(X, T, ai, bi, C, Kij, dwilson, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_rk(X, T, ai, bi, order, C, Kij, Crk, Crk1, combinatory):
    parameter = (Crk, Crk1, combinatory)
    if order == 0:
        mixparameters = ws(X, T, ai, bi, C, Kij, rk, parameter)
    elif order == 1:
        mixparameters = dws(X, T, ai, bi, C, Kij, rk, parameter)
    elif order == 2:
        mixparameters = d2ws(X, T, ai, bi, C, Kij, drk, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_unifac(X, T, ai, bi, order, C, Kij, qi, ri, ri34, Vk, Qk,
              tethai, a0, a1, a2):
    parameter = (qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2)
    if order == 0:
        mixparameters = ws(X, T, ai, bi, C, Kij, unifac, parameter)
    elif order == 1:
        mixparameters = dws(X, T, ai, bi, C, Kij, unifac, parameter)
    elif order == 2:
        mixparameters = d2ws(X, T, ai, bi, C, Kij, dunifac, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters
