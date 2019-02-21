import numpy as np
from .actmodels_cy import wilson_cy


def wilson(X, T, A, vl):
    '''
    Wilson activity coefficient model
    
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
    '''
    X = np.asarray(X,dtype = np.float64)
    v = vl(T)
    M = np.divide.outer(v, v).T * np.exp(-A/T)
    lngama = wilson_cy(X, M)
    return lngama