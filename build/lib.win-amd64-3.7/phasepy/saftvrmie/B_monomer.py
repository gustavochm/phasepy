import numpy as np
from .monomer_aux import I_lam, J_lam


#Derivadas respecto a eta
def B(x0, eta, lam, eps):
    #B calculation Eq 33
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    eta13 = (1-eta)**3
    ter1 = (1-eta/2)*I/eta13 - 9.*eta*(1+eta)*J/(2.*eta13)
    b = 12.*eta*eps*ter1
    return b

def dB(x0, eta, lam, eps):
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    eta13 = (1-eta)**3.
    ter1 = (1.-eta/2.)*I/eta13 - 9.*eta*(1+eta)*J/(2.*eta13)
    b = 12.*eta*eps*ter1
    
    #first derivative 
    ter1b1 = I*(-2. + (eta-2.)*eta) + J*18.*eta*(1.+2.*eta)
    ter2b1 = -6.*eps/(eta-1.)**4.
    db = ter1b1*ter2b1
    return  np.hstack([b, db])

def d2B(x0, eta, lam, eps):
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    eta13 = (1.-eta)**3.
    ter1 = (1.-eta/2.)*I/eta13 - 9.*eta*(1.+eta)*J/(2.*eta13)
    b = 12.*eta*eps*ter1
    
    #first derivative 
    db = I*(-2. + (eta-2)*eta) + J*18.*eta*(1.+2.*eta)
    db *= (-6.*eps/(eta-1.)**4)
    
    #second derivative
    d2b = I*(-5. + (eta-2)*eta) + J*9.*(1 + eta*(7. +4.*eta))
    d2b *= 12.*eps/(eta-1)**5
    return np.hstack([b, db, d2b])

def d3B(x0, eta, lam, eps):
    I = I_lam(x0, lam)
    J = J_lam(x0, lam)
    
    eta13 = (1.-eta)**3.
    ter1 = (1.-eta/2.)*I/eta13 - 9.*eta*(1.+eta)*J/(2.*eta13)
    b = 12.*eta*eps*ter1
    
    #first derivative 
    db = I*(-2. + (eta-2.)*eta) + J*18.*eta*(1.+2.*eta)
    db *= (-6.*eps/(eta-1.)**4.)
    
    #second derivative
    d2b = I*(-5. + (eta-2.)*eta) + J*9.*(1. + eta*(7. +4.*eta))
    d2b *= 12.*eps/(eta-1.)**5.
    
    d3b = I*(9. - (eta-2.)*eta) - J*36.*(1. + eta*(3. + eta))
    d3b *= (36.*eps/(1.- eta)**6.)
    return  np.hstack([b, db, d2b, d3b])