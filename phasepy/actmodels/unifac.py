from __future__ import division, print_function, absolute_import
import numpy as np
from .actmodels_cy import lnG_cy

def unifac(x, T, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    '''
    Dortmund UNIFAC activity coefficient model for multicomponent mixtures.
    
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
    '''
    
    nc = len(x)
    ng = len(Qk)

    #combinatory part
    rx = np.dot(x, ri)
    r34x = np.dot(x, ri34)
    qx = np.dot(x,qi)
    phi = ri34/r34x
    phi_tetha = (ri*qx) / (qi*rx)
    lngamac = np.log(phi)
    lngamac += 1 - phi
    lngamac -= 5*qi*(np.log(phi_tetha)+ 1 - phi_tetha)
    
    amn = a0 + a1 * T + a2 * T**2
    
    #residual part
    psi = np.exp(-amn/T)
    Xm = x@Vk
    Xm = Xm/Xm.sum()
    tetha =  Xm*Qk
    tetha /= tetha.sum() 
    Gm = lnG_cy(tetha, psi, Qk) 
    
    Gi = np.zeros((nc, ng))
    for i in range(nc): 
        Gi[i] = lnG_cy(tethai[i], psi, Qk)
    
    lngamar = (Vk*(Gm - Gi)).sum(axis=1)
    
    return lngamac + lngamar

