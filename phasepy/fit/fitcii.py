import numpy as np 
from ..sgt import ten_fit
from ..math import gauss

def fit_cii(tenexp, Texp, model, orden= 2, n = 100 ):
    
    """
    
    fit_cii fit influence parameters of SGT
    
    Inputs
    ----------
    tenexp: experimental surface tension in mN/m
    Texp: experimental temperature in K.
    model: object created from eos and component
    order: int, order of cii polynomial 
    n : int, number of integration points in SGT

    
    """
    roots, weigths = gauss(n)
    tena = np.zeros_like(tenexp)
    
    for i in range(len(Texp)):
        tena[i]=ten_fit(Texp[i], model, roots, weigths)
        
    cii=(tenexp/tena)**2
    
    return np.polyfit(Texp,cii,orden)