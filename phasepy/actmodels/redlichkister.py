from __future__ import division, print_function, absolute_import
import numpy as np
from .actmodels_cy import rkb_cy, rk_cy

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
    x = np.asarray(x,dtype = np.float64)
    
    G = C + C1 / T
    Mp = rkb_cy(x, G)
    return Mp
    
def rk(x, T, C, C1, combinatory):
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
    combinatory: array_like
        index by pairs of Redlich Kister Expansion
        
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
    ge, dge = rk_cy(x, G, combinatory)
    Mp = ge + dge - np.dot(dge,x)
    return Mp
    