import numpy as np 
from ..sgt import ten_fit
from ..math import gauss

def fit_cii(tenexp, Texp, model, orden= 2, n = 100 ):
    
    #puntos y pesos de cuadratura Gauss
    roots, weigths = gauss(n)
    tena = np.zeros_like(tenexp)
    
    for i in range(len(Texp)):
        tena[i]=ten_fit(Texp[i], model, roots, weigths)
        
    cii=(tenexp/tena)**2
    
    return np.polyfit(Texp,cii,orden)