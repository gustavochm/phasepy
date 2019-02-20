import numpy as np
from .actmodels_cy import nrtl_cy

    
def nrtl(X, T, alpha, g, g1):
    '''
    NRTL activity coefficient model.
    
    input
    X: array like, vector of molar fractions
    T: float, absolute temperature in K.
    g: array like, matrix of energy interactions in K.
    g1: array_like, matrix of energy interactions in K^2
    alpha: float, aleatory factor.
    
    tau = ((g + g1/T)/T)
    
    output
    lngama: array_like, natural logarithm of activify coefficient
    '''
    X = np.asarray(X, dtype = np.float64)
    tau = g/T + g1
    G = np.exp(-alpha*tau)
    lngama = nrtl_cy(X, tau, G)
    
    return lngama

def rkter_nrtl(x,d):
    n = len(x)
    q = np.zeros_like(x)
    for i in range(n):
        x2 = x.copy()
        if x2[i] != 0.:
            x2[i] = 1.
        for k in range(n):
            if k != i:
                q[i] -= (-1+3*x[i])*x[k]*d[k]
            else:
                q[i] += (2-3*x[i])*x[k]*d[k]
        q[i] *= np.prod(x2)
    return q

def nrtlter(X,T, alpha,g, g1, D):
    return nrtl(X, T,alpha, g , g1) + rkter_nrtl(X, D)