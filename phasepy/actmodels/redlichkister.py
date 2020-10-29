from __future__ import division, print_function, absolute_import
import numpy as np
from .actmodels_cy import rkb_cy, rk_cy, drk_cy


def rkb_aux(x, G):
    x = np.asarray(x, dtype=np.float64)
    Mp = rkb_cy(x, G)
    return Mp


def rk_aux(x, G, combinatory):
    x = np.asarray(x, dtype=np.float64)
    ge, dge = rk_cy(x, G, combinatory)
    Mp = ge + dge - np.dot(dge, x)
    return Mp


def drk_aux(x, G, combinatory):
    x = np.asarray(x, dtype=np.float64)
    ge, dge, d2ge = drk_cy(x, G, combinatory)
    Mp = ge + dge - np.dot(dge, x)
    dMp = d2ge - d2ge@x
    return Mp, dMp


def rkb(x, T, C, C1):
    '''
    Redlich-Kister activity coefficient model for multicomponent mixtures.

    Parameters
    ----------
    X: array_like
        vector of molar fractions
    T: float
        absolute temperature in K
    C: array_like
        polynomial values adim
    C1: array_like
        polynomial values in K

    Notes
    -----

    G = C + C1/T

    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    '''
    x = np.asarray(x, dtype=np.float64)

    G = C + C1 / T
    Mp = rkb_cy(x, G)
    return Mp


def rk(x, T, C, C1, combinatory):
    r'''
    Redlich-Kister activity coefficient model for multicomponent
    mixtures. This method uses a polynomial fit of Gibbs excess
    energy. It is not recommended to use more than 5 terms of the
    polynomial expansion. Function returns array of natural logarithm
    of activity coefficients.

    .. math::
	g^e_{ij} = x_ix_j \sum_{k=0}^m C_k (x_i - x_j)^k

    .. math::
        G = C + C_1/T

    Parameters
    ----------
    X: array
        Molar fractions
    T: float
        Absolute temperature [K]
    C: array
        Polynomial coefficient values adim
    C1: array
        Polynomial coefficient values [K]
    '''
    x = np.asarray(x, dtype=np.float64)

    G = C + C1 / T
    ge, dge = rk_cy(x, G, combinatory)
    Mp = ge + dge - np.dot(dge, x)
    return Mp


def drk(x, T, C, C1, combinatory):
    '''
    Derivatives of Redlich-Kister activity coefficient model
    for multicomponent mixtures.

    Parameters
    ----------
    X: array_like
        vector of molar fractions
    T: float
        absolute temperature in K
    C: array_like
        polynomial values adim
    C1: array_like
        polynomial values in K
    combinatory: array_like
        index by pairs of Redlich Kister Expansion

    Notes
    -----

    G = C + C1/T

    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    dlngama: array_like
        derivative of natural logarithm of activify coefficient
    '''
    x = np.asarray(x, dtype=np.float64)

    G = C + C1 / T
    ge, dge, d2ge = drk_cy(x, G, combinatory)
    Mp = ge + dge - np.dot(dge, x)
    dMp = d2ge - d2ge@x
    return Mp, dMp
