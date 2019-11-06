import numpy as np
from .monomer_aux import  eta_eff, deta_eff,  d2eta_eff,  d3eta_eff
'''
#Derivadas respecto a eta
def a1s(eta, lam, eps):
    #a1s calculation Eq 39
    neff = eta_eff(eta, lam)
    
    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    return a1

def da1s(eta, lam, eps):
    
    neff, dneff = deta_eff(eta, lam)
    
    #a1s calculation Eq 39
    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    
    #da1s calculation 
    ter1a1 = -2 +3*neff - neff**2 + eta*(-5 + 2*neff)*dneff
    ter2a1 = 6*eps/((lam-3)*(1-neff)**4)
    da1 = ter1a1 * ter2a1

    return np.hstack([a1, da1])

def d2a1s(eta, lam, eps):
    neff, dneff, d2neff = d2eta_eff(eta, lam)
    
    #a1s calculation Eq 39
    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    
    #da1s calculation 
    ter1a1 = -2 +3*neff - neff**2 + eta*(-5 + 2*neff)*dneff
    ter2a1 = 6*eps/((lam-3)*(1-neff)**4)
    da1 = ter1a1 * ter2a1
    
    #d2a1s calculation 
    ter1a2 = 2. * dneff * (5. - 7*neff+ 2*neff**2 - 3* eta * (-3 + neff)*dneff)
    ter1a2 += eta*(-1 + neff)*(-5+2*neff)*d2neff
    ter2a2 = 6*eps/((lam-3)*(-1+neff)**5)
    d2a1 = ter1a2 * ter2a2
    return np.hstack([a1, da1, d2a1])

def d3a1s(eta, lam, eps):
    
    neff, dneff, d2neff, d3neff = d3eta_eff(eta, lam)
    
    #a1s calculation Eq 39

    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    
    #da1s calculation 
    ter1a1 = -2 +3*neff - neff**2 + eta*(-5 + 2*neff)*dneff
    ter2a1 = 6*eps/((lam-3)*(1-neff)**4)
    da1 = ter1a1 * ter2a1
    
    #d2a1s calculation 
    ter1a2 = 2. * dneff * (5. - 7*neff+ 2*neff**2 - 3* eta * (-3 + neff)*dneff)
    ter1a2 += eta*(-1 + neff)*(-5+2*neff)*d2neff
    ter2a2 = 6*eps/((lam-3)*(-1+neff)**5)
    d2a1 = ter1a2 * ter2a2
    
    #d3a1s calculation
    ter1a3 = -18. * (-3 + neff) * (-1 + neff) * dneff**2 
    ter1a3 +=  12. * eta * (-7 + 2* neff) *dneff**3
    ter1a3 += - 18 * eta *(-3 + neff) *(-1 + neff) * dneff * d2neff
    ter1a3 += (-1 + neff)**2 * (-5 +2 * neff) * (3*d2neff + eta * d3neff)
    ter2a3 = 6*eps/((lam-3)*(-1+neff)**6)
    d3a1 = ter1a3 * ter2a3
    return np.hstack([a1, da1, d2a1, d3a1])


'''
def a1s(eta, lam, cctes, eps):
    #a1s calculation Eq 39
    neff = eta_eff(eta, cctes)
    
    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    return a1

def da1s(eta, lam, cctes, eps):
    
    neff, dneff = deta_eff(eta, cctes)
    
    #a1s calculation Eq 39
    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    
    #da1s calculation 
    ter1a1 = -2 +3*neff - neff**2 + eta*(-5 + 2*neff)*dneff
    ter2a1 = 6*eps/((lam-3)*(1-neff)**4)
    da1 = ter1a1 * ter2a1

    return np.hstack([a1, da1])

def d2a1s(eta, lam, cctes, eps):
    neff, dneff, d2neff = d2eta_eff(eta, cctes)
    
    #a1s calculation Eq 39
    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    
    #da1s calculation 
    ter1a1 = -2 +3*neff - neff**2 + eta*(-5 + 2*neff)*dneff
    ter2a1 = 6*eps/((lam-3)*(1-neff)**4)
    da1 = ter1a1 * ter2a1
    
    #d2a1s calculation 
    ter1a2 = 2. * dneff * (5. - 7*neff+ 2*neff**2 - 3* eta * (-3 + neff)*dneff)
    ter1a2 += eta*(-1 + neff)*(-5+2*neff)*d2neff
    ter2a2 = 6*eps/((lam-3)*(-1+neff)**5)
    d2a1 = ter1a2 * ter2a2
    return np.hstack([a1, da1, d2a1])

def d3a1s(eta, lam, cctes, eps):
    
    neff, dneff, d2neff, d3neff = d3eta_eff(eta, cctes)
    
    #a1s calculation Eq 39

    ghs = (1 - neff/2) / (1 - neff)**3 
    a1vdw = -12 * eps * eta / (lam -3)
    a1 = a1vdw * ghs
    
    #da1s calculation 
    ter1a1 = -2 +3*neff - neff**2 + eta*(-5 + 2*neff)*dneff
    ter2a1 = 6*eps/((lam-3)*(1-neff)**4)
    da1 = ter1a1 * ter2a1
    
    #d2a1s calculation 
    ter1a2 = 2. * dneff * (5. - 7*neff+ 2*neff**2 - 3* eta * (-3 + neff)*dneff)
    ter1a2 += eta*(-1 + neff)*(-5+2*neff)*d2neff
    ter2a2 = 6*eps/((lam-3)*(-1+neff)**5)
    d2a1 = ter1a2 * ter2a2
    
    #d3a1s calculation
    ter1a3 = -18. * (-3 + neff) * (-1 + neff) * dneff**2 
    ter1a3 +=  12. * eta * (-7 + 2* neff) *dneff**3
    ter1a3 += - 18 * eta *(-3 + neff) *(-1 + neff) * dneff * d2neff
    ter1a3 += (-1 + neff)**2 * (-5 +2 * neff) * (3*d2neff + eta * d3neff)
    ter2a3 = 6*eps/((lam-3)*(-1+neff)**6)
    d3a1 = ter1a3 * ter2a3
    return np.hstack([a1, da1, d2a1, d3a1])

