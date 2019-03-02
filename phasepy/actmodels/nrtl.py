from __future__ import division, print_function, absolute_import
import numpy as np
from .actmodels_cy import nrtl_cy, rkter_nrtl_cy

    
def nrtl(X, T, alpha, g, g1):
    '''
    NRTL activity coefficient model.
    
    Parameters
    ----------
    X: array like
        vector of molar fractions
    T: float
        absolute temperature in K
    g: array like
        matrix of energy interactions in K
    g1: array_like
        matrix of energy interactions in 1/K
    alpha: array_like
        aleatory factor.
    
    tau = ((g + g1*T)/T)
    
    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    '''
    X = np.asarray(X, dtype = np.float64)
    tau = g/T + g1
    G = np.exp(-alpha*tau)
    lngama = nrtl_cy(X, tau, G)
    
    return lngama


def nrtlter(X,T, alpha,g, g1, D):
    '''
    NRTL activity coefficient model.
    
    Parameters
    ----------
    X: array like
        vector of molar fractions
    T: float
        absolute temperature in K
    g: array like
        matrix of energy interactions in K
    g1: array_like
        matrix of energy interactions in 1/K
    alpha: array_like
        aleatory factor
    D : array_like
        ternary contribution parameters
    
    tau = ((g + g1*T)/T)
    
    Returns
    -------
    lngama: array_like
        natural logarithm of activify coefficient
    '''
    xd = X*D
    lngama = rkter_nrtl_cy(X, xd)
    lngama += nrtl(X, T,alpha, g , g1)
    return lngama