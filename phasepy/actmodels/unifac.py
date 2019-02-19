import numpy as np
from .actmodels_cy import lnG_cy

def unifac(x, T, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    
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

