from __future__ import division, print_function, absolute_import
import numpy as np


def unifac_original_aux(x, qi, ri, Vk, Qk, tethai, psi):

    # Combinatory part
    rx = np.dot(x, ri)
    qx = np.dot(x, qi)
    phi_x = ri/rx
    tetha = x*qi / qx
    phi_tetha = (ri*qx) / (qi*rx)

    lngamac = np.log(phi_x) + 1. - phi_x
    lngamac -= 5.*qi*(np.log(phi_tetha) + 1. - phi_tetha)

    # Residual part
    Xm = x@Vk
    Xm = Xm/Xm.sum()
    tetha = Xm*Qk
    tetha /= tetha.sum()

    SumA = tetha@psi
    SumB = (psi*tetha)@(1./SumA)
    Gm = Qk * (1 - np.log(SumA) - SumB)

    SumAi = tethai@psi
    SumBi = np.tensordot((tethai/SumAi), psi, axes=(1, 1))
    Gi = Qk * (1 - np.log(SumAi) - SumBi)

    lngamar = (Vk*(Gm - Gi)).sum(axis=1)

    lngama = lngamac + lngamar
    return lngama


def dunifac_original_aux(x, qi, ri, Vk, Qk, tethai, psi):

    # Combinatory part
    rx = np.dot(x, ri)
    qx = np.dot(x, qi)
    phi_x = ri/rx

    phi_tetha = (ri*qx) / (qi*rx)

    dphi_x = - np.outer(ri, ri)/rx**2
    dphi_tetha = np.outer(ri/qi, rx*qi - ri*qx) / rx**2

    lngamac = np.log(phi_x) + 1. - phi_x
    lngamac -= 5.*qi*(np.log(phi_tetha) + 1. - phi_tetha)

    dlngamac = (dphi_x * (1./phi_x - 1.)).T
    dlngamac -= 5*qi*(dphi_tetha.T * (1/phi_tetha - 1))

    # Residual part
    Xm1 = x@Vk
    Xm1s = Xm1.sum()
    dXm1s = np.sum(Vk, axis=1)
    Xm = Xm1 / Xm1s
    dXm = (Vk * Xm1s - np.outer(dXm1s, Xm1))/Xm1s**2

    tetha1 = Xm*Qk
    tetha1s = tetha1.sum()
    tetha = tetha1/tetha1s
    dtetha = ((Qk*dXm)*tetha1s-np.outer(dXm@Qk, tetha1))/tetha1s**2

    SumA = tetha@psi
    dSumA = dtetha@psi
    dter1 = (dSumA / SumA).T

    SumB = (psi*tetha)@(1./SumA)
    dSumB = (psi/SumA)@dtetha.T - (tetha*psi/SumA**2)@dSumA.T

    Gm = Qk * (1 - np.log(SumA) - SumB)
    dlnGk = (Qk * (- dter1 - dSumB).T).T

    SumAi = tethai@psi
    SumBi = np.tensordot((tethai/SumAi), psi, axes=(1, 1))
    Gi = Qk * (1 - np.log(SumAi) - SumBi)

    lngamar = (Vk*(Gm - Gi)).sum(axis=1)
    dlngamar = Vk@dlnGk

    lngama = lngamac + lngamar
    dlngama = dlngamac + dlngamar

    return lngama, dlngama


def unifac_original(x, T, qi, ri, Vk, Qk, tethai, amn):
    '''
    Derivatives of Dortmund UNIFAC activity coefficient model
    for multicomponent mixtures. Group definitions and parameter values from
    `Dortmund public database <http://www.ddbst.com/published-parameters-unifac.html>`_.
    Function returns array of natural logarithm of activity

    Parameters
    ----------
    X: array like
        vector of molar fractions
    T: float
        absolute temperature in K
    qi: array like
        component surface array
    ri: array_like
        component volumes arrays
    Vk : array_like
        group volumes array
    Qk : array_like
        group surface arrays
    tethai : array_like
        surface fraction array
    amn : array_like
        energy interactions coefficient

    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    '''
    # Combinatory part
    rx = np.dot(x, ri)
    qx = np.dot(x, qi)
    phi_x = ri/rx
    tetha = x*qi / qx
    phi_tetha = (ri*qx) / (qi*rx)

    lngamac = np.log(phi_x) + 1. - phi_x
    lngamac -= 5.*qi*(np.log(phi_tetha) + 1. - phi_tetha)

    # Residual part
    psi = np.exp(-amn/T)
    Xm = x@Vk
    Xm = Xm/Xm.sum()
    tetha = Xm*Qk
    tetha /= tetha.sum()

    SumA = tetha@psi
    SumB = (psi*tetha)@(1./SumA)
    Gm = Qk * (1 - np.log(SumA) - SumB)

    SumAi = tethai@psi
    SumBi = np.tensordot((tethai/SumAi), psi, axes=(1, 1))
    Gi = Qk * (1 - np.log(SumAi) - SumBi)

    lngamar = (Vk*(Gm - Gi)).sum(axis=1)

    lngama = lngamac + lngamar
    return lngama


def dunifac_original(x, T, qi, ri, Vk, Qk, tethai, amn):
    '''
    Derivatives of Dortmund UNIFAC activity coefficient model
    for multicomponent mixtures. Group definitions and parameter values from
    `Dortmund public database <http://www.ddbst.com/published-parameters-unifac.html>`_.
    Function returns array of natural logarithm of activity

    Parameters
    ----------
    X: array like
        vector of molar fractions
    T: float
        absolute temperature in K
    qi: array like
        component surface array
    ri: array_like
        component volumes arrays
    Vk : array_like
        group volumes array
    Qk : array_like
        group surface arrays
    tethai : array_like
        surface fraction array
    amn : array_like
        energy interactions coefficient

    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    dlngama: array_like
        derivative of natural logarithm of activify coefficient respect to molar fraction
    '''

    # Combinatory part
    rx = np.dot(x, ri)
    qx = np.dot(x, qi)
    phi_x = ri/rx

    phi_tetha = (ri*qx) / (qi*rx)

    dphi_x = - np.outer(ri, ri)/rx**2
    dphi_tetha = np.outer(ri/qi, rx*qi - ri*qx) / rx**2

    lngamac = np.log(phi_x) + 1. - phi_x
    lngamac -= 5.*qi*(np.log(phi_tetha) + 1. - phi_tetha)

    dlngamac = (dphi_x * (1./phi_x - 1.)).T
    dlngamac -= 5*qi*(dphi_tetha.T * (1/phi_tetha - 1))

    # Residual part
    psi = np.exp(-amn/T)

    Xm1 = x@Vk
    Xm1s = Xm1.sum()
    dXm1s = np.sum(Vk, axis=1)
    Xm = Xm1 / Xm1s
    dXm = (Vk * Xm1s - np.outer(dXm1s, Xm1))/Xm1s**2

    tetha1 = Xm*Qk
    tetha1s = tetha1.sum()
    tetha = tetha1/tetha1s
    dtetha = ((Qk*dXm)*tetha1s-np.outer(dXm@Qk, tetha1))/tetha1s**2

    SumA = tetha@psi
    dSumA = dtetha@psi
    dter1 = (dSumA / SumA).T

    SumB = (psi*tetha)@(1./SumA)
    dSumB = (psi/SumA)@dtetha.T - (tetha*psi/SumA**2)@dSumA.T

    Gm = Qk * (1 - np.log(SumA) - SumB)
    dlnGk = (Qk * (- dter1 - dSumB).T).T

    SumAi = tethai@psi
    SumBi = np.tensordot((tethai/SumAi), psi, axes=(1, 1))
    Gi = Qk * (1 - np.log(SumAi) - SumBi)

    lngamar = (Vk*(Gm - Gi)).sum(axis=1)
    dlngamar = Vk@dlnGk

    lngama = lngamac + lngamar
    dlngama = dlngamac + dlngamar

    return lngama, dlngama
