from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import nrtl_aux, dnrtl_aux, nrtlter_aux, dnrtlter_aux
from ..actmodels import wilson_aux, dwilson_aux
from ..actmodels import rk_aux, drk_aux
from ..actmodels import unifac_aux, dunifac_aux
from ..actmodels import uniquac_aux, duniquac_aux
from ..actmodels import unifac_original_aux, dunifac_original_aux


def mhv1(X, RT, ai, bi, q1, ActModel, parameter):
    '''
    Modified Huron vidal mixrule-1

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    q1: MHV1 cubic eos constants
    ActModel: function, activity coefficient model.
    parameter : tuple of parameters to evaluate ActModel.


    Out :
    am (mixture a term)
    bm (mixture b term)
    '''
    # Pure component dimensionless parameter
    ei = ai/(bi*RT)

    # Mixture covolume
    bm = np.dot(bi, X)
    # Acivity coefficient
    lngama = ActModel(X, *parameter)
    Gex = np.dot(lngama, X)

    bibm = bi/bm
    logbibm = np.log(bibm)

    em = (Gex - np.dot(logbibm, X)) / q1 + np.dot(X, ei)
    am = em*bm*RT

    return am, bm


def dmhv1(X, RT, ai, bi, q1, ActModel, parameter):
    '''
    Modified Huron vidal mixrule-1

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    q1: MHV1 cubic eos constants
    ActModel: function, activity coefficient model.
    parameter : tuple of parameters to evaluate ActModel.


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    '''
    # Pure component dimensionless parameter
    ei = ai/(bi*RT)

    # Mixture covolume
    bm = np.dot(bi, X)
    # Acivity coefficient
    lngama = ActModel(X, *parameter)
    Gex = np.dot(lngama, X)

    bibm = bi/bm
    logbibm = np.log(bibm)

    em = (Gex - np.dot(logbibm, X)) / q1 + np.dot(X, ei)
    am = em*bm*RT

    # partial derivatives
    dedx = (lngama - logbibm + bibm)/q1 + ei
    dnem_dn = em + dedx - np.dot(dedx, X)
    # Partial attractive term
    da_dn = em*(bi-bm)*RT + dnem_dn*bm*RT

    D = am
    Di = da_dn + am
    B = bm
    Bi = bi

    return D, Di, B, Bi


def d2mhv1(X, RT, ai, bi, q1, ActModel, parameter):
    '''
    Modified Huron vidal mixrule-1

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    q1: MHV1 cubic eos constants
    ActModel: function, activity coefficient model.
    parameter : tuple of parameters to evaluate ActModel.


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    nc = len(X)
    dxjdni = np.eye(nc) - X

    # Pure component dimensionless parameter
    ei = ai/(bi*RT)

    # Mixture covolume
    bm = np.dot(bi, X)
    bibm = bi/bm
    logbibm = np.log(bibm)
    dbm_dn = bi - bm

    lngama, dlng_dx = ActModel(X, *parameter)
    dlngama = dlng_dx@dxjdni.T
    Gex = np.dot(lngama, X)

    em = (Gex - np.dot(logbibm, X)) / q1 + np.dot(X, ei)
    am = em*bm*RT

    # partial derivatives
    dedx = (lngama - logbibm + bibm)/q1 + ei
    dnem_dn = em + dedx - np.dot(dedx, X)
    # Partial attractive term
    da_dn = em*(bi-bm)*RT + dnem_dn*bm*RT

    # second order derivatives
    dnem_dnij = (dlngama + np.outer(1/bm - bibm/bm, dbm_dn)) / q1

    dnam_dnij = dnem_dnij*bm
    dnam_dnij += np.outer(dnem_dn, dbm_dn)
    dnam_dnij += np.outer(dbm_dn, dnem_dn)
    dnam_dnij -= em*np.add.outer(dbm_dn, dbm_dn)
    dnam_dnij *= RT

    D = am
    Di = da_dn + am
    Dij = dnam_dnij + np.add.outer(da_dn, da_dn)
    B = bm
    Bi = bi
    Bij = 0.
    return D, Di, Dij, B, Bi, Bij


def mhv1_nrtl(X, RT, ai, bi, order, q1, tau, G):
    '''
    Modified Huron vidal mixrule-1 with nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    q1: MHV1 cubic eos constants
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
        mixparameters = mhv1(X, RT, ai, bi, q1, nrtl_aux, parameter)
    elif order == 1:
        mixparameters = dmhv1(X, RT, ai, bi, q1, nrtl_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv1(X, RT, ai, bi, q1, dnrtl_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv1_wilson(X, RT, ai, bi, order, q1, M):
    '''
    Modified Huron vidal mixrule-1 with wilson model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    q1: MHV1 cubic eos constants
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
        mixparameters = mhv1(X, RT, ai, bi, q1, wilson_aux, parameter)
    elif order == 1:
        mixparameters = dmhv1(X, RT, ai, bi, q1, wilson_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv1(X, RT, ai, bi, q1, dwilson_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv1_nrtlt(X, RT, ai, bi, order, q1, tau, G, D):
    '''
    Modified Huron vidal mixrule-1 with modified ternary nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    q1: MHV1 cubic eos constants
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
        mixparameters = mhv1(X, RT, ai, bi, q1, nrtlter_aux, parameter)
    elif order == 1:
        mixparameters = dmhv1(X, RT, ai, bi, q1, nrtlter_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv1(X, RT, ai, bi, q1, dnrtlter_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv1_rk(X, RT, ai, bi, order, q1, G, combinatory):
    '''
    Modified Huron vidal mixrule-1 with Redlich Kister model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    q1: MHV1 cubic eos constants
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
        mixparameters = mhv1(X, RT, ai, bi, q1, rk_aux, parameter)
    elif order == 1:
        mixparameters = dmhv1(X, RT, ai, bi, q1, rk_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv1(X, RT, ai, bi, q1, drk_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv1_unifac(X, RT, ai, bi, order, q1, qi, ri, ri34, Vk, Qk,
                tethai, amn, psi):
    '''
    Modified Huron vidal mixrule-1 with UNIFAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    q1: MHV1 cubic eos constants
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
        mixparameters = mhv1(X, RT, ai, bi, q1, unifac_aux, parameter)
    elif order == 1:
        mixparameters = dmhv1(X, RT, ai, bi, q1, unifac_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv1(X, RT, ai, bi, q1, dunifac_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv1_uniquac(X, RT, ai, bi, order, q1, ri, qi, tau):
    '''
    Modified Huron vidal mixrule-1 with UNIQUAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    q1: MHV1 cubic eos constants
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
        mixparameters = mhv1(X, RT, ai, bi, q1, uniquac_aux, parameter)
    elif order == 1:
        mixparameters = dmhv1(X, RT, ai, bi, q1, uniquac_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv1(X, RT, ai, bi, q1, duniquac_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv1_unifac_original(X, RT, ai, bi, order, q1, qi, ri, Vk, Qk,
                         tethai, psi):
    '''
    Modified Huron vidal mixrule-1 with UNIFAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    q1: MHV1 cubic eos constants
    qi, ri, Vk, Qk, tethai, psi: parameters to evaluae original UNIFAC.

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
        mixparameters = mhv1(X, RT, ai, bi, q1, unifac_original_aux,
                             parameter)
    elif order == 1:
        mixparameters = dmhv1(X, RT, ai, bi, q1, unifac_original_aux,
                              parameter)
    elif order == 2:
        mixparameters = d2mhv1(X, RT, ai, bi, q1, dunifac_original_aux,
                               parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters
