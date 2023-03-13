from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import nrtl_aux, dnrtl_aux, nrtlter_aux, dnrtlter_aux
from ..actmodels import wilson_aux, dwilson_aux
from ..actmodels import rk_aux, drk_aux
from ..actmodels import unifac_aux, dunifac_aux
from ..actmodels import uniquac_aux, duniquac_aux
from ..actmodels import unifac_original_aux, dunifac_original_aux


# Modified Huron Vidal Mixrule
# Adimentional volume of mixture and first derivative
def U_mhv(em, c1, c2):
    ter1 = em-c1-c2
    ter2 = c1*c2+em
    ter3 = np.sqrt(ter1**2 - 4*ter2)
    Umhv = ter1 - ter3
    Umhv /= 2

    dUmhv = 1 + (2 - ter1)/ter3
    dUmhv /= 2

    return Umhv, dUmhv


# Adimentional volume of mixture and first  and second derivative
def dU_mhv(em, c1, c2):
    ter1 = em - c1 - c2
    ter2 = c1 * c2 + em
    ter3 = np.sqrt(ter1**2 - 4*ter2)

    Umhv = ter1 - ter3
    Umhv /= 2

    dUmhv = 1 + (- ter1 + 2) / ter3
    dUmhv /= 2

    d2Umhv = 2 * (1 + c1) * (1 + c2) / ter3**1.5
    return Umhv, dUmhv, d2Umhv


# Objetive function MHV and its first derivatives
def f0_mhv(em, zm, c1, c2):

    Umhv, dUmhv = U_mhv(em, c1, c2)

    Umhv_1 = Umhv - 1
    Umhvc1 = Umhv + c1
    Umhvc2 = Umhv + c2
    logUmhv = np.log(Umhvc1/Umhvc2)
    Umhvc1c2 = Umhvc1 * Umhvc2

    f0 = (- 1. - np.log(Umhv_1) - (em/(c1-c2)) * logUmhv)
    f0 -= zm

    df0 = -dUmhv / Umhv_1 - (1/(c1-c2)) * logUmhv
    df0 += dUmhv * em / Umhvc1c2

    return f0, df0


# Objetive function MHV and its derivatives
def df0_mhv(em, zm, c1, c2):

    Umhv, dUmhv, d2Umhv = dU_mhv(em, c1, c2)

    Umhv_1 = Umhv - 1
    Umhvc1 = Umhv + c1
    Umhvc2 = Umhv + c2
    logUmhv = np.log(Umhvc1/Umhvc2)
    Umhvc1c2 = Umhvc1 * Umhvc2

    f0 = (- 1. - np.log(Umhv_1) - (em/(c1-c2)) * logUmhv)
    f0 -= zm

    df0 = -dUmhv / Umhv_1 - (1/(c1-c2)) * logUmhv
    df0 += dUmhv * em / Umhvc1c2

    d2f0 = 2 * dUmhv / Umhvc1c2
    d2f0 += dUmhv**2 * (1./Umhv_1**2-em*(c1+c2+2.*Umhv)/Umhvc1c2**2)
    d2f0 += d2Umhv * (-1./Umhv_1+em/Umhvc1c2)

    return f0, df0, d2f0


#  Adimentional paramter solver with newton method
def em_solver(X, e, zm, c1, c2):
    em = np.dot(X, e)
    it = 0.
    f0, df0 = f0_mhv(em, zm, c1, c2)
    error = np.abs(f0)
    while error > 1e-8 and it < 10:
        it += 1
        de = f0 / df0
        em -= de
        error = np.abs(de)
        f0, df0 = f0_mhv(em, zm, c1, c2)
    return em, df0


# Adimentional paramater solver with Halleys method
def dem_solver(X, e, zm, c1, c2):
    em = np.dot(X, e)
    it = 0.
    f0, df0, d2f0 = df0_mhv(em, zm, c1, c2)
    error = np.abs(f0)
    while error > 1e-8 and it < 10:
        it += 1
        de = - (2*f0*df0)/(2*df0**2-f0*d2f0)
        em += de
        error = np.abs(de)
        f0, df0, d2f0 = df0_mhv(em, zm, c1, c2)
    return em, df0, d2f0


def mhv(X, RT, ai, bi, c1, c2, ActModel, parameter):
    '''
    Modified Huron vidal mixrule

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    ActModel: function, activity coefficient model.
    parameter : tuple of parameters to evaluate ActModel.


    Out :
    am (mixture a term)
    bm (mixture b term)
    '''
    e = ai/(bi*RT)
    # Pure component reduced volume
    U = (e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2
    # Pure component fugacity at zero pressure
    z = -1.-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))
    # Mixture Fugacity
    bm = np.dot(bi, X)
    # Acivity coefficient
    lngama = ActModel(X, *parameter)
    Gex = np.dot(lngama, X)

    bibm = bi/bm
    logbibm = np.log(bibm)

    zm = Gex + np.dot(z, X) - np.dot(logbibm, X)
    em, der = em_solver(X, e, zm, c1, c2)
    am = em*bm*RT

    return am, bm


def dmhv(X, RT, ai, bi, c1, c2, ActModel, parameter):
    '''
    Modified Huron vidal mixrule

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
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
    e = ai/(bi*RT)
    # Pure component reduced volume
    U = (e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2
    # Pure component fugacity at zero pressure
    z = -1-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))
    # Mixture Fugacity
    bm = np.dot(bi, X)
    # Acivity coefficient
    lngama = ActModel(X, *parameter)
    Gex = np.dot(lngama, X)

    bibm = bi/bm
    logbibm = np.log(bibm)

    zm = Gex + np.dot(z, X) - np.dot(logbibm, X)
    em, der = em_solver(X, e, zm, c1, c2)
    am = em*bm*RT

    # Partial fugacity
    zp = lngama + z - logbibm + bibm - 1.
    dedn = (zp-zm)/der
    # Partial attractive term
    ap = am + em*(bi-bm)*RT + dedn*bm*RT

    D = am
    Di = am + ap
    B = bm
    Bi = bi

    return D, Di, B, Bi


def d2mhv(X, RT, ai, bi, c1, c2, ActModel, parameter):
    '''
    Modified Huron vidal mixrule

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
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
    e = ai/(bi*RT)
    U = (e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2.

    # Pure component fugacity at zero pressure
    z = -1-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))
    # Mixture Fugacity
    bm = np.dot(bi, X)
    # Acivity coefficient
    lngama, dlng_dx = ActModel(X, *parameter)
    dlngama = dlng_dx@dxjdni.T
    Gex = np.dot(lngama, X)

    bibm = bi / bm
    logbibm = np.log(bibm)
    dbm_dn = bi - bm

    zm = Gex + np.dot(z, X) - np.dot(logbibm, X)
    em, der, der2 = dem_solver(X, e, zm, c1, c2)
    am = em*bm*RT

    # Partial fugacity
    zp = lngama + z - logbibm + bibm - 1.
    dem_dn = (zp-zm)/der
    # Partial attractive term
    ap = am + em*dbm_dn*RT + dem_dn*bm*RT

    # ep = em + dem_dn
    # dam_dn = ap - am
    dzm_dn = zp - zm

    dzp_dnij = dlngama + np.outer(1/bm - bibm/bm, dbm_dn)

    dem_dnij = der * (dzp_dnij - dzm_dn)
    dem_dnij -= np.outer(dzm_dn, der2 * dem_dn)
    dem_dnij /= der**2
    dem_dnij += dem_dn

    dap_dnij = dem_dnij * bm * RT
    dap_dnij += RT*np.outer(dbm_dn, dem_dn)
    dap_dnij += RT*np.outer(dem_dn, dbm_dn)

    D = am
    Di = am + ap
    Dij = dap_dnij + np.add.outer(ap, ap)
    B = bm
    Bi = bi
    Bij = 0.
    return D, Di, Dij, B, Bi, Bij


def mhv_nrtl(X, RT, ai, bi, order, c1, c2, tau, G):
    '''
    Modified Huron vidal mixrule with nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
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
        mixparameters = mhv(X, RT, ai, bi, c1, c2, nrtl_aux, parameter)
    elif order == 1:
        mixparameters = dmhv(X, RT, ai, bi, c1, c2, nrtl_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, RT, ai, bi, c1, c2, dnrtl_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv_wilson(X, RT, ai, bi, order, c1, c2, M):
    '''
    Modified Huron vidal mixrule with wilson model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
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
        mixparameters = mhv(X, RT, ai, bi, c1, c2, wilson_aux, parameter)
    elif order == 1:
        mixparameters = dmhv(X, RT, ai, bi, c1, c2, wilson_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, RT, ai, bi, c1, c2, dwilson_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv_nrtlt(X, RT, ai, bi, order, c1, c2, tau, G, D):
    '''
    Modified Huron vidal mixrule with modified ternary nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
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
        mixparameters = mhv(X, RT, ai, bi, c1, c2, nrtlter_aux, parameter)
    elif order == 1:
        mixparameters = dmhv(X, RT, ai, bi, c1, c2, nrtlter_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, RT, ai, bi, c1, c2, dnrtlter_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv_rk(X, RT, ai, bi, order, c1, c2, G, combinatory):
    '''
    Modified Huron vidal mixrule with Redlich Kister model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
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
        mixparameters = mhv(X, RT, ai, bi, c1, c2, rk_aux, parameter)
    elif order == 1:
        mixparameters = dmhv(X, RT, ai, bi, c1, c2, rk_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, RT, ai, bi, c1, c2, drk_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv_unifac(X, RT, ai, bi, order, c1, c2, qi, ri, ri34, Vk, Qk,
               tethai, amn, psi):
    '''
    Modified Huron vidal mixrule with Dortmund-UNIFAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
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
        mixparameters = mhv(X, RT, ai, bi, c1, c2, unifac_aux, parameter)
    elif order == 1:
        mixparameters = dmhv(X, RT, ai, bi, c1, c2, unifac_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, RT, ai, bi, c1, c2, dunifac_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv_uniquac(X, RT, ai, bi, order, c1, c2, ri, qi, tau):
    '''
    Modified Huron vidal mixrule with UNIQUAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition
            derivatives and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
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
        mixparameters = mhv(X, RT, ai, bi, c1, c2, uniquac_aux, parameter)
    elif order == 1:
        mixparameters = dmhv(X, RT, ai, bi, c1, c2, uniquac_aux, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, RT, ai, bi, c1, c2, duniquac_aux, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters


def mhv_unifac_original(X, RT, ai, bi, order, c1, c2, qi, ri, Vk, Qk,
                        tethai, psi):
    '''
    Modified Huron vidal mixrule with Original-UNIFAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    qi, ri, Vk, Qk, tethai, psi: parameters to evaluate original UNIFAC

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
        mixparameters = mhv(X, RT, ai, bi, c1, c2, unifac_original_aux,
                            parameter)
    elif order == 1:
        mixparameters = dmhv(X, RT, ai, bi, c1, c2, unifac_original_aux,
                             parameter)
    elif order == 2:
        mixparameters = d2mhv(X, RT, ai, bi, c1, c2, dunifac_original_aux,
                              parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters
