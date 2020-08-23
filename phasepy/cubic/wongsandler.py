from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import nrtl_aux, dnrtl_aux, nrtlter_aux, dnrtlter_aux
from ..actmodels import wilson_aux, dwilson_aux
from ..actmodels import rk_aux, drk_aux
from ..actmodels import unifac_aux, dunifac_aux


def ws(X, RT, ai, bi, C, Kij, ActModel, parameter):

    # Acivity coefficient
    lngama = ActModel(X, *parameter)
    Gex = np.dot(lngama, X)

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


def dws(X, RT, ai, bi, C, Kij, ActModel, parameter):

    # Acivity coefficient
    lngama = ActModel(X, *parameter)
    Gex = np.dot(lngama, X)

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


def d2ws(X, RT, ai, bi, C, Kij, ActModel, parameter):

    # Acivity coefficient
    lngama, dlngama = ActModel(X, *parameter)
    Gex = np.dot(lngama, X)

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


def ws_nrtl(X, T, ai, bi, order, C, Kij, tau, G):
    parameter = (tau, G)
    if order == 0:
        mixparameters = ws(X, T, ai, bi, C, Kij, nrtl_aux, parameter)
    elif order == 1:
        mixparameters = dws(X, T, ai, bi, C, Kij, nrtl_aux, parameter)
    elif order == 2:
        mixparameters = d2ws(X, T, ai, bi, C, Kij, dnrtl_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_nrtlt(X, RT, ai, bi, order, C, Kij, tau, G, D):
    parameter = (tau, G, D)
    if order == 0:
        mixparameters = ws(X, RT, ai, bi, C, Kij, nrtlter_aux, parameter)
    elif order == 1:
        mixparameters = dws(X, RT, ai, bi, C, Kij, nrtlter_aux, parameter)
    elif order == 2:
        mixparameters = d2ws(X, RT, ai, bi, C, Kij, dnrtlter_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_wilson(X, RT, ai, bi, order, C, Kij, M):
    parameter = (M, )
    if order == 0:
        mixparameters = ws(X, RT, ai, bi, C, Kij, wilson_aux, parameter)
    elif order == 1:
        mixparameters = dws(X, RT, ai, bi, C, Kij, wilson_aux, parameter)
    elif order == 2:
        mixparameters = d2ws(X, RT, ai, bi, C, Kij, dwilson_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_rk(X, RT, ai, bi, order, C, Kij, G, combinatory):
    parameter = (G, combinatory)
    if order == 0:
        mixparameters = ws(X, RT, ai, bi, C, Kij, rk_aux, parameter)
    elif order == 1:
        mixparameters = dws(X, RT, ai, bi, C, Kij, rk_aux, parameter)
    elif order == 2:
        mixparameters = d2ws(X, RT, ai, bi, C, Kij, drk_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_unifac(X, RT, ai, bi, order, C, Kij, qi, ri, ri34, Vk, Qk,
              tethai, amn, psi):
    parameter = (qi, ri, ri34, Vk, Qk, tethai, amn, psi)
    if order == 0:
        mixparameters = ws(X, RT, ai, bi, C, Kij, unifac_aux, parameter)
    elif order == 1:
        mixparameters = dws(X, RT, ai, bi, C, Kij, unifac_aux, parameter)
    elif order == 2:
        mixparameters = d2ws(X, RT, ai, bi, C, Kij, dunifac_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters
