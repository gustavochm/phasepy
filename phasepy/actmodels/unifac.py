from __future__ import division, print_function, absolute_import
import numpy as np


def unifac_aux(x, qi, ri, ri34, Vk, Qk, tethai, amn, psi):

    # Combinatory part
    rx = np.dot(x, ri)
    r34x = np.dot(x, ri34)
    qx = np.dot(x, qi)
    phi = ri34/r34x
    phi_tetha = (ri*qx) / (qi*rx)
    lngamac = np.log(phi)
    lngamac += 1 - phi
    lngamac -= 5*qi*(np.log(phi_tetha)+1-phi_tetha)

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

    return lngamac + lngamar


def dunifac_aux(x, qi, ri, ri34, Vk, Qk, tethai, amn, psi):

    # nc = len(x)
    # ng = len(Qk)

    # Combinatory part
    rx = np.dot(x, ri)
    r34x = np.dot(x, ri34)
    qx = np.dot(x, qi)
    phi = ri34/r34x
    phi_tetha = (ri*qx) / (qi*rx)
    lngamac = np.log(phi)
    lngamac += 1 - phi
    lngamac -= 5*qi*(np.log(phi_tetha)+1-phi_tetha)

    dphi = - np.outer(ri34, ri34)/r34x**2
    dphi_tetha = np.outer(ri/qi, rx*qi - ri*qx) / rx**2
    dlngamac = (dphi * (1/phi - 1)).T
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


def unifac(x, T, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    r'''
    Dortmund Modified-UNIFAC activity coefficient model for
    multicomponent mixtures is a group contribution method, which uses
    group definitions and parameter values from
    `Dortmund public database <http://www.ddbst.com/PublishedParametersUNIFACDO.html>`_.
    Function returns array of natural logarithm of activity
    coefficients.

    .. math::
	\ln \gamma_i = \ln \gamma_i^{comb} + \ln \gamma_i^{res}

    Energy interaction equation is

    .. math::
        a_{mn} = a_0 + a_1 T + a_2 T^2

    Parameters
    ----------
    X: array
        Molar fractions
    T: float
        Absolute temperature [K]
    qi: array
        Component surface array
    ri: array
        Component volumes array
    ri34 : array
        Component volume array, exponent 3/4
    Vk : array
        Group volumes
    Qk : array
        Group surface array
    tethai : array
        Surface fractions
    a0 : array
        Energy interactions polynomial coefficients
    a1 : array
        Energy interactions polynomial coefficients
    a2 : array
        Energy interactions polynomial coefficients
    '''

    # nc = len(x)
    # ng = len(Qk)

    # Combinatory part
    rx = np.dot(x, ri)
    r34x = np.dot(x, ri34)
    qx = np.dot(x, qi)
    phi = ri34/r34x
    phi_tetha = (ri*qx) / (qi*rx)
    lngamac = np.log(phi)
    lngamac += 1 - phi
    lngamac -= 5*qi*(np.log(phi_tetha)+1-phi_tetha)

    amn = a0 + a1 * T + a2 * T**2

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

    return lngamac + lngamar


def dunifac(x, T, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    '''
    Derivatives of Dortmund UNIFAC activity coefficient model
    for multicomponent mixtures.

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
    ri34 : array_like
        component volumen arrays power to 3/4
    Vk : array_like
        group volumes array
    Qk : array_like
        group surface arrays
    tethai : array_like
        surface fraction array
    a0 : array_like
        energy interactions polynomial coefficient
    a1 : array_like
        energy interactions polynomial coefficient
    a2 : array_like
        energy interactions polynomial coefficient

    Notes
    -----
    Energy interaction arrays: amn = a0 + a1 * T + a2 * T**2

    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    dlngama: array_like
        derivative of natural logarithm of activify coefficient
    '''

    # nc = len(x)
    # ng = len(Qk)

    # Combinatory part
    rx = np.dot(x, ri)
    r34x = np.dot(x, ri34)
    qx = np.dot(x, qi)
    phi = ri34/r34x
    phi_tetha = (ri*qx) / (qi*rx)
    lngamac = np.log(phi)
    lngamac += 1 - phi
    lngamac -= 5*qi*(np.log(phi_tetha)+1-phi_tetha)

    dphi = - np.outer(ri34, ri34)/r34x**2
    dphi_tetha = np.outer(ri/qi, rx*qi - ri*qx) / rx**2
    dlngamac = (dphi * (1/phi - 1)).T
    dlngamac -= 5*qi*(dphi_tetha.T * (1/phi_tetha - 1))

    amn = a0 + a1 * T + a2 * T**2

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
