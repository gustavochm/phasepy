from __future__ import division, print_function, absolute_import
import numpy as np


def uniquac_aux(x, ri, qi, tau):

    # Combinatory part
    rx = np.dot(x, ri)
    qx = np.dot(x, qi)
    phi_x = ri/rx
    tetha = x*qi / qx
    phi_tetha = (ri*qx) / (qi*rx)

    lngamac = np.log(phi_x) + 1. - phi_x
    lngamac -= 5.*qi*(np.log(phi_tetha) + 1. - phi_tetha)

    # residual part
    Sj = np.matmul(tetha, tau)
    SumA = np.matmul(tau, (tetha/Sj))
    lngamar = 1. - np.log(Sj) - SumA
    lngamar *= qi

    lngama = lngamac + lngamar
    return lngama


def duniquac_aux(x, ri, qi, tau):

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
    nc = len(x)
    dx = np.eye(nc)
    tethai = x*qi
    dtethai = dx*qi
    tetha = tethai / qx
    Sj = np.matmul(tetha, tau)
    SumA = np.matmul(tau, (tetha/Sj))

    lngamar = 1. - np.log(Sj) - SumA
    lngamar *= qi

    dtetha = ((dtethai*qx - np.outer(tethai, qi)) / qx**2).T
    dSj = np.matmul(dtetha, tau)
    dSumA_aux = ((dtetha*Sj - tetha*dSj)/Sj**2).T
    dSumA = np.matmul(tau, dSumA_aux)

    dlngamar = (-dSj/Sj - dSumA.T)
    dlngamar *= qi

    lngama = lngamac + lngamar
    dlngama = dlngamac + dlngamar
    return lngama, dlngama


def uniquac(x, T, ri, qi, a0, a1):
    r"""
    UNIQUAC activity coefficient model. This function returns array of natural
    logarithm of activity coefficients.

    .. math::
	\ln \gamma_i = \ln \gamma_i^{comb} + \ln \gamma_i^{res}

    Energy interaction equation is:

    .. math::
        a_{ij} = a_0 + a_1 T

    Parameters
    ----------
    x: array
        Molar fractions
    T: float
        Absolute temperature [K]
    ri: array
        Component volumes array
    qi: array
        Component surface array
    a0 : array
        Energy interactions polynomial coefficients [K]
    a1 : array
        Energy interactions polynomial coefficients [Adim]

    Returns
    -------
    lngama: array
        natural logarithm of activity coefficients.
    dlngama: array
        composition derivatives of activity coefficients natural logarithm.
    """
    # Combinatory part
    rx = np.dot(x, ri)
    qx = np.dot(x, qi)
    phi_x = ri/rx
    tetha = x*qi / qx
    phi_tetha = (ri*qx) / (qi*rx)

    lngamac = np.log(phi_x) + 1. - phi_x
    lngamac -= 5.*qi*(np.log(phi_tetha) + 1. - phi_tetha)

    # residual part
    Aij = a0 + a1 * T
    tau = np.exp(-Aij/T)
    Sj = np.matmul(tetha, tau)
    SumA = np.matmul(tau, (tetha/Sj))
    lngamar = 1. - np.log(Sj) - SumA
    lngamar *= qi

    lngama = lngamac + lngamar
    return lngama


def duniquac(x, T, ri, qi, a0, a1):
    r"""
    UNIQUAC activity coefficient model. This function returns array of natural
    logarithm of activity coefficients and its composition derivates matrix.

    .. math::
	\ln \gamma_i = \ln \gamma_i^{comb} + \ln \gamma_i^{res}

    Energy interaction equation is:

    .. math::
        a_{ij} = a_0 + a_1 T

    Parameters
    ----------
    x: array
        Molar fractions
    T: float
        Absolute temperature [K]
    ri: array
        Component volumes array
    qi: array
        Component surface array
    a0 : array
        Energy interactions polynomial coefficients [K]
    a1 : array
        Energy interactions polynomial coefficients [Adim]

    Returns
    -------
    lngama: array
        natural logarithm of activity coefficients.
    """
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
    nc = len(x)
    dx = np.eye(nc)
    tethai = x*qi
    dtethai = dx*qi
    tetha = tethai / qx

    Aij = a0 + a1 * T
    tau = np.exp(-Aij/T)

    Sj = np.matmul(tetha, tau)
    SumA = np.matmul(tau, (tetha/Sj))

    lngamar = 1. - np.log(Sj) - SumA
    lngamar *= qi

    dtetha = ((dtethai*qx - np.outer(tethai, qi)) / qx**2).T
    dSj = np.matmul(dtetha, tau)
    dSumA_aux = ((dtetha*Sj - tetha*dSj)/Sj**2).T
    dSumA = np.matmul(tau, dSumA_aux)

    dlngamar = (-dSj/Sj - dSumA.T)
    dlngamar *= qi

    lngama = lngamac + lngamar
    dlngama = dlngamac + dlngamar
    return lngama, dlngama
