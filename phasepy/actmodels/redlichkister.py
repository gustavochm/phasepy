from __future__ import division, print_function, absolute_import
import numpy as np
from .actmodels_cy import rk_cy

def rk(x, T, C, C1):
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
    x = np.asarray(x,dtype = np.float64)
    
    G = C + C1 / T
    Mp = rk_cy(x, G)
    return Mp
    
