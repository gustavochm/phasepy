import numpy as np
from .actmodels_cy import rk_cy, rkb_cy 

def rkb(x, T, C, C1):
    '''
    Redlich-Kister activity coefficient model for multicomponent mixtures.
    
    input
    X: array like, vector of molar fractions
    T: float, absolute temperature in K.
    C: array like, polynomial values adim..
    C1: array_like, polynomial values in K.
    
    G = C + C1/T
    
    output
    lngama: array_like, natural logarithm of activify coefficient
    '''
    x = np.asarray(x,dtype = np.float64)
    
    G = C + C1 / T
    Mp = rkb_cy(x, G)
    return Mp

def rk(x, T, C, C1, combinatoria):
    '''
    Redlich-Kister activity coefficient model for multicomponent mixtures.
    
    input
    X: array like, vector of molar fractions
    T: float, absolute temperature in K.
    C: array like, polynomial values adim..
    C1: array_like, polynomial values in K.
    
    G = C + C1/T
    
    output
    lngama: array_like, natural logarithm of activify coefficient
    '''
    x = np.asarray(x,dtype = np.float64)
    combinatoria = np.asarray(combinatoria, dtype = np.float64)
    G = C + C1 / T
    Mp = rk_cy(x, G, combinatoria)
    return Mp
    
