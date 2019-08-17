from __future__ import division, print_function, absolute_import
import numpy as np
from ..math import gauss
from .tensionresult import TensionResult

def ten_fit(T, model, roots, weigths, P0 = None):
    
    #equilibrio LV
    Psat, vl, vv = model.psat(T, P0)
    rol = 1./vl
    rov = 1./vv
    
    #Dimensionless variables
    Tfactor, Pfactor, rofactor, tenfactor = model.sgt_adim_fit(T)
    
    rola = rol * rofactor
    rova = rov * rofactor 
    Tad = T * Tfactor
    Pad = Psat * Pfactor
    
    #potenciales quimicos
    mu0 = model.muad(rova,Tad)
    
    #Discretizacion de 
    roi = (rola-rova) * roots + rova
    wreal = (rola-rova) * weigths
    
    dOm = model.dOm(roi,Tad,mu0,Pad)
    tenint = np.nan_to_num(np.sqrt(2*dOm))
    ten = np.dot(wreal, tenint)
    ten *= tenfactor
    
    return ten 

def ten_pure(rov, rol, Tsat, Psat, model, n = 100, full_output = False):
    
    #roots and weights of Gauss quadrature
    roots, w = gauss(n)
    
    #variables adimensionales
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)

    rola = rol * rofactor
    rova = rov * rofactor 
    Tad = Tsat * Tfactor
    Pad = Psat * Pfactor   
    
    #Equilibrium chemical potential

    mu0 = model.muad(rova,Tad)
    
    roi = (rola-rova) * roots + rova
    wreal = (rola-rova)*w

    dOm = np.zeros(n)
    for i in range(n):
        dOm[i] = model.dOm(roi[i],Tad, mu0, Pad)
        
    tenint = np.nan_to_num(np.sqrt(2*dOm))
    ten = np.dot(wreal, tenint)
    ten *= tenfactor
    
    if full_output:
        zint = np.sqrt(1/(2*dOm))
        z = np.cumsum(wreal*zint)
        z *= zfactor
        roi /= rofactor
        dictresult = {'tension' : ten, 'ro': roi, 'z' : z,
        'GPT' : dOm}
        out = TensionResult(dictresult)
        return out
    
    return ten

'''
def ten_pure(T, model, P0 = None, full_output = False, n = 100):
    
    #roots and weights of Gauss quadrature
    roots, w = gauss(n)
    
    #LVE
    Psat = model.psat(T, P0)
    rol = model.density(T, Psat, 'L')
    rov = model.density(T, Psat, 'V')
    
    #variables adimensionales
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(T)
    
    rola = rol * rofactor
    rova = rov * rofactor 
    Tad = T * Tfactor
    Pad = Psat * Pfactor
    
    #Equilibrium chemical potential
    mu0 = model.muad(rova,Tad)
    
   
    roi = (rola-rova) * roots + rova
    wreal = (rola-rova)*w
    
    dOm = model.dOm(roi,Tad,mu0,Pad)
    tenint = np.nan_to_num(np.sqrt(2*dOm))
    ten = np.dot(wreal, tenint)
    ten *= tenfactor
    
    if full_output:
        zint = np.sqrt(1/(2*dOm))
        z = np.cumsum(wreal*zint)
        z *= zfactor
        roi /= rofactor
        dictresult = {'tension' : ten, 'ro': roi, 'z' : z,
        'GPT' : dOm}
        out = TensionResult(dictresult)
        return out
    
    return ten
'''