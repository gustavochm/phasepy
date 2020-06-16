from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import nrtl, wilson, nrtlter, rk, unifac
from ..actmodels import dnrtl, dwilson, dnrtlter, drk, dunifac
from ..constants import R


#Modified Huron Vidal Mixrule

#adimentional volume of mixture and first derivative
def U_mhv(em,c1,c2):
    ter1 = em-c1-c2
    ter2 = c1*c2+em
    ter3 = np.sqrt(ter1**2 - 4*ter2)
    Umhv = ter1 - ter3
    Umhv /= 2

    dUmhv = 1 + (2 - ter1) /ter3
    dUmhv /= 2

    return Umhv, dUmhv
#adimentional volume of mixture and first  and second derivative
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

#objetive function MHV and its first derivatives
def f0_mhv(em,zm,c1,c2):

    Umhv, dUmhv = U_mhv(em,c1,c2)

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

#objetive function MHV and its derivatives
def df0_mhv(em, zm, c1, c2):

    Umhv, dUmhv, d2Umhv = dU_mhv(em,c1,c2)

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
    d2f0 += dUmhv**2 * (1. / Umhv_1**2 - em * (c1 + c2 + 2. * Umhv) / Umhvc1c2**2 )
    d2f0 += d2Umhv * (-1. / Umhv_1 + em / Umhvc1c2)

    return f0, df0, d2f0

#adimentional paramter solver with newton method
def em_solver(X, e, zm,c1,c2):
    em = np.dot(X, e)
    it = 0.
    f0, df0 = f0_mhv(em,zm,c1,c2)
    error = np.abs(f0)
    while error > 1e-8 and it < 10:
        it += 1
        de = f0 / df0
        em -= de
        error = np.abs(de)
        f0, df0 = f0_mhv(em,zm,c1,c2)
    return em, df0

#adimentional paramater solver with Halleys method
def dem_solver(X, e, zm,c1,c2):
    em = np.dot(X, e)
    it = 0.
    f0, df0, d2f0 = df0_mhv(em,zm,c1,c2)
    error = np.abs(f0)
    while error > 1e-8 and it < 10:
        it += 1
        de = - (2*f0*df0)/(2*df0**2-f0*d2f0)
        em += de
        error = np.abs(de)
        f0, df0, d2f0 = df0_mhv(em,zm,c1,c2)
    return em, df0, d2f0

def mhv(X, T, ai, bi, c1, c2, ActModel, parameter):
    '''
    Modified Huron vidal mixrule

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    ActModel: function, activity coefficient model.
    parameter : tuple of parameters to evaluate ActModel.


    Out :
    am (mixture a term)
    bm (mixture b term)
    '''
    e = ai/(bi*R*T)
    #Pure component reduced volume
    U=(e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2
    #Pure component fugacity at zero pressure
    z= -1.-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))
    # Mixture Fugacity
    bm = np.dot(bi, X)
    #Acivity coefficient
    lngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama,X)

    bibm = bi/bm
    logbibm = np.log(bibm)

    zm = Gex + np.dot(z, X) - np.dot(logbibm,X)
    em, der = em_solver(X, e, zm, c1, c2)
    am=em*bm*R*T

    return am, bm


def dmhv(X, T, ai, bi, c1, c2, ActModel, parameter):
    '''
    Modified Huron vidal mixrule

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
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
    e = ai/(bi*R*T)
    #Pure component reduced volume
    U=(e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2
    #Pure component fugacity at zero pressure
    z=-1-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))
    # Mixture Fugacity
    bm = np.dot(bi, X)
    #Acivity coefficient
    lngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama,X)

    bibm = bi/bm
    logbibm = np.log(bibm)

    zm = Gex + np.dot(z, X) - np.dot(logbibm,X)
    em, der = em_solver(X, e, zm, c1, c2)
    am=em*bm*R*T

    #partial fugacity
    zp = lngama + z - logbibm + bibm - 1.
    dedn = (zp-zm)/der
    #partial attractive term
    ap = am + em*(bi-bm)*R*T + dedn*bm*R*T

    D = am
    Di = am + ap
    B = bm
    Bi = bi

    return D, Di, B, Bi

def d2mhv(X, T, ai, bi, c1, c2, ActModel, parameter):
    '''
    Modified Huron vidal mixrule

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
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
    RT = R*T
    e = ai/(bi*R*T)
    U = (e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2.

    #Pure component fugacity at zero pressure
    z = -1-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))

    # Mixture Fugacity
    bm = np.dot(bi, X)
    #Acivity coefficient
    #lngama = ActModel(X, T, *parameter)
    lngama, dlng_dx = ActModel(X, T, *parameter)
    dlngama = dlng_dx@dxjdni.T
    Gex = np.dot(lngama,X)

    bibm = bi / bm
    logbibm = np.log(bibm)
    dbm_dn = bi - bm

    zm = Gex + np.dot(z, X) - np.dot(logbibm,X)
    em, der, der2 = dem_solver(X, e, zm, c1, c2)
    am = em*bm*RT

    #partial fugacity
    zp = lngama + z - logbibm + bibm - 1.
    dem_dn = (zp-zm)/der
    #partial attractive term
    ap = am + em*dbm_dn*RT + dem_dn*bm*RT

    ep = em + dem_dn
    dam_dn = ap - am
    dzm_dn = zp - zm

    dzp_dnij = dlngama + np.outer(1/bm - bibm/bm, dbm_dn)

    dem_dnij = der * (dzp_dnij - dzm_dn)
    dem_dnij -= np.outer(dzm_dn , der2 * dem_dn)
    dem_dnij /= der**2
    dem_dnij += dem_dn

    dap_dnij = dem_dnij * bm * RT
    dap_dnij += R*T*np.outer(dbm_dn, dem_dn)
    dap_dnij += R*T*np.outer(dem_dn, dbm_dn)

    D = am
    Di = am + ap
    Dij = dap_dnij +  np.add.outer(ap, ap)
    B = bm
    Bi = bi
    Bij = 0.
    return D, Di, Dij, B, Bi, Bij

def mhv_nrtl(X, T, ai, bi, order, c1, c2, alpha, g, g1):
    '''
    Modified Huron vidal mixrule with nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition derivatives
            and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
    alpha, g, g1 : array_like, parameters to evaluate nrtl model


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter = (alpha, g, g1)
    if order == 0:
        mixparameters = mhv(X, T, ai, bi, c1, c2, nrtl, parameter)
    elif order == 1:
        mixparameters = dmhv(X, T, ai, bi, c1, c2, nrtl, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, T, ai, bi, c1, c2, dnrtl, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters

def mhv_wilson(X, T, ai, bi, order, c1, c2, Aij, vl):
    '''
    Modified Huron vidal mixrule with wilson model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition derivatives
            and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
    Aij : array_like, parameters to evaluate wilson model
    vl : function to evaluate pure liquid volumes


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter=(Aij,vl)
    if order == 0:
        mixparameters = mhv(X, T, ai, bi, c1, c2, wilson, parameter)
    elif order == 1:
        mixparameters = dmhv(X, T, ai, bi, c1, c2, wilson, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, T, ai, bi, c1, c2, dwilson, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters

def mhv_nrtlt(X,T, ai, bi, order, c1, c2, alpha, g, g1, D):
    '''
    Modified Huron vidal mixrule with modified ternary nrtl model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition derivatives
            and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
    alpha, g, g1 : array_like, parameters to evaluate nrtl model
    D : array_like, parameter to evaluate ternary term.


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter=(alpha, g, g1, D)

    if order == 0:
        mixparameters = mhv(X, T, ai, bi, c1, c2, nrtlter, parameter)
    elif order == 1:
        mixparameters = dmhv(X, T, ai, bi, c1, c2, nrtlter, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, T, ai, bi, c1, c2, dnrtlter, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters



def mhv_rk(X, T, ai, bi, order, c1, c2, C, C1, combinatory):
    '''
    Modified Huron vidal mixrule with Redlich Kister model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    order : 0 for mixture a, b, 1 for a and b and its first composition derivatives
            and 2 for a and b and its first a second derivatives.
    c1, c2: cubic eos constants
    C, C1 : array_like, parameters to evaluate Redlich Kister polynomial
    combinatory: array_like, array_like, contains info of the order of polynomial
            coefficients by pairs.


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter=(C, C1, combinatory)

    if order == 0:
        mixparameters = mhv(X, T, ai, bi, c1, c2, rk, parameter)
    elif order == 1:
        mixparameters = dmhv(X, T, ai, bi, c1, c2, rk, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, T, ai, bi, c1, c2, drk, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters

def mhv_unifac(X,T,ai,bi, order, c1, c2, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    '''
    Modified Huron vidal mixrule with UNIFAC model

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2: parameters to evaluae modified
        Dortmund UNIFAC.

    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''
    parameter = (qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2)
    if order == 0:
        mixparameters = mhv(X, T, ai, bi, c1, c2, unifac, parameter)
    elif order == 1:
        mixparameters = dmhv(X, T, ai, bi, c1, c2, unifac, parameter)
    elif order == 2:
        mixparameters = d2mhv(X, T, ai, bi, c1, c2, dunifac, parameter)
    else:
        raise Exception('Derivative order not valid')
    return mixparameters
