from __future__ import division, print_function, absolute_import
import numpy as np
from .actmodels_cy import wilson_cy, dwilson_cy


def wilson_aux(X, M):
    X = np.asarray(X, dtype=np.float64)
    lngama = wilson_cy(X, M)
    return lngama


def dwilson_aux(X, M):
    X = np.asarray(X, dtype=np.float64)
    lngama, dlngama = dwilson_cy(X, M)
    return lngama, dlngama


def wilson(X, T, A, vl):
    r'''Wilson activity coefficient model is a local composition model
    recommended for vapor-liquid equilibria calculation. It can't
    predict liquid liquid equilibrium. Function returns array of
    natural logarithm of activity coefficients.

    .. math::
	g^e = \sum_{i=1}^c x_i \ln ( \sum_{j=1}^c x_j 	\Lambda_{ij})

    Parameters
    ----------
    X: array
        Molar fractions
    T: float
        Absolute temperature [K]
    A: array like
        Matrix of energy interactions [K]
    vl: function
        Returns liquid volume of species [:math:`\mathrm{cm^3/mol}`]
        given temperature [K] as argument.
    '''
    X = np.asarray(X, dtype=np.float64)
    v = vl(T)
    M = np.divide.outer(v, v).T * np.exp(-A/T)
    lngama = wilson_cy(X, M)
    return lngama


def dwilson(X, T, A, vl):
    '''
    Derivatives of Wilson activity coefficient model

    Parameters
    ----------
    X: array like
        vector of molar fractions
    T: float
        absolute temperature in K
    A: array like
        matrix of energy interactions in K
    vl: function
        liquid volume of species in cm3/mol

    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    dlngama: array_like
        derivative of natural logarithm of activify coefficient
    '''
    X = np.asarray(X, dtype=np.float64)
    v = vl(T)
    M = np.divide.outer(v, v).T * np.exp(-A/T)
    lngama, dlngama = dwilson_cy(X, M)

    return lngama, dlngama
