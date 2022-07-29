from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import nrtl_aux, dnrtl_aux, nrtlter_aux, dnrtlter_aux
from ..actmodels import wilson_aux, dwilson_aux
from ..actmodels import rk_aux, drk_aux
from ..actmodels import unifac_aux, dunifac_aux
from ..actmodels import uniquac_aux, duniquac_aux
from ..actmodels import unifac_original_aux, dunifac_original_aux


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
    D = Gex/C + np.dot(X, ei)
    D1 = 1. - D

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
    abij *= (1. - Kij)
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


def ws_nrtl(X, RT, ai, bi, order, C, Kij, tau, G):
    '''
    Wons-Sandler mixrule with nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    C: cubic eos constant, -np.log((1+c1)/(1+c2))/(c1-c2)
    Kij: array_like, correction to cross (b - a/RT)ij
    tau, G : array_like, parameters to evaluate nrtl model


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter = (tau, G)
    if order == 0:
        mixparameters = ws(X, RT, ai, bi, C, Kij, nrtl_aux, parameter)
    elif order == 1:
        mixparameters = dws(X, RT, ai, bi, C, Kij, nrtl_aux, parameter)
    elif order == 2:
        mixparameters = d2ws(X, RT, ai, bi, C, Kij, dnrtl_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_nrtlt(X, RT, ai, bi, order, C, Kij, tau, G, D):
    '''
    Wons-Sandler mixrule with modified ternary nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    C: cubic eos constant, -np.log((1+c1)/(1+c2))/(c1-c2)
    Kij: array_like, correction to cross (b - a/RT)ij
    tau, G : array_like, parameters to evaluate nrtl model
    D : array_like, parameter to evaluate ternary term.


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
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
    '''
    Wons-Sandler mixrule with wilson model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    C: cubic eos constant, -np.log((1+c1)/(1+c2))/(c1-c2)
    Kij: array_like, correction to cross (b - a/RT)ij
    M : array_like, parameters to evaluate wilson model


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
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
    '''
    Wons-Sandler mixrule with Redlich Kister model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    C: cubic eos constant, -np.log((1+c1)/(1+c2))/(c1-c2)
    Kij: array_like, correction to cross (b - a/RT)ij
    G : array_like, parameters to evaluate Redlich Kister polynomial
    combinatory: array_like, array_like, contains info of the order of
                 polynomial coefficients by pairs.


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
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
    '''
    Wons-Sandler mixrule with UNIFAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    C: cubic eos constant, -np.log((1+c1)/(1+c2))/(c1-c2)
    Kij: array_like, correction to cross (b - a/RT)ij
    qi, ri, ri34, Vk, Qk, tethai, amn, psi: parameters to evaluae modified
                                            Dortmund UNIFAC.


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
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


def ws_uniquac(X, RT, ai, bi, order, C, Kij, ri, qi, tau):
    '''
    Wons-Sandler mixrule with UNIQUAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    C: cubic eos constant, -np.log((1+c1)/(1+c2))/(c1-c2)
    Kij: array_like, correction to cross (b - a/RT)ij
    qi, ri, tau : array_like, parameters to evaluate UNIQUAC model


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter = (ri, qi, tau)
    if order == 0:
        mixparameters = ws(X, RT, ai, bi, C, Kij, uniquac_aux, parameter)
    elif order == 1:
        mixparameters = dws(X, RT, ai, bi, C, Kij, uniquac_aux, parameter)
    elif order == 2:
        mixparameters = d2ws(X, RT, ai, bi, C, Kij, duniquac_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def ws_unifac_original(X, RT, ai, bi, order, C, Kij, qi, ri, Vk, Qk,
                       tethai, psi):
    '''
    Wons-Sandler mixrule with Original-UNIFAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    C: cubic eos constant, -np.log((1+c1)/(1+c2))/(c1-c2)
    Kij: array_like, correction to cross (b - a/RT)ij
    qi, ri, Vk, Qk, tethai, amn: parameters to evaluate original UNIFAC.

    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter = (qi, ri, Vk, Qk, tethai, psi)
    if order == 0:
        mixparameters = ws(X, RT, ai, bi, C, Kij, unifac_original_aux,
                           parameter)
    elif order == 1:
        mixparameters = dws(X, RT, ai, bi, C, Kij, unifac_original_aux,
                            parameter)
    elif order == 2:
        mixparameters = d2ws(X, RT, ai, bi, C, Kij, dunifac_original_aux,
                             parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters
    