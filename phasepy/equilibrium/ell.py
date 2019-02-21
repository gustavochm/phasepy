import numpy as np
from .stability import tpd_minimas
from .flash import flash
from .multiflash import multiflash
        

def ell(x0, w0, Z, T, P, model, v0 = [None, None], full_output = False):
    
    """
    Liquid liquid equilibrium (z,T,P) -> (x,w,beta)
    
    Solves liquid liquid equilibrium from multicomponent mixtures at given
    pressure, temperature and overall composition.
    
    Parameters
    ----------
    
    x0 : array_like
        initial guess for liquid phase 1
    w0 : array_like
        initial guess for liquid phase 2
    z : array_like
        overal composition of mix
    T : float
        absolute temperature in K.
    P : float 
        pressure in en bar
    model : object
        created from mixture, eos and mixrule 
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    full_output: bool, optional
        wheter to outputs all calculation info
    
    Returns
    -------
    X : array_like
        liquid 1 mole fraction vector
    W : array_like
        liquid 2 mole fraction vector
    beta : float
        phase fraction of liquid 2
    
    """

    
    
    equilibrio = ['L', 'L']
    out = flash(x0, w0, equilibrio, Z, T, P, model, v0, True)
    X, W, beta = out.X, out.Y, out.beta
    v1, v2 = out.v1, out.v2
    X0 = np.array([X, W])
    
    beta0 = np.array([1-beta, beta, 0.])
    
    out = multiflash(X0, beta0, equilibrio, Z, T, P, model, [v1, v2], True)
    Xm, beta, tetha, v = out.X, out.beta, out.tetha, out.v
    
    if tetha > 0:
        xes, tpd_min2 = tpd_minimas(2,Xm[0],T,P,model, 'L', 'L', v)
        X0 = np.asarray(xes)
        beta0 = np.hstack([beta, 0.])
        out = multiflash(X0, beta0, equilibrio, Z, T, P, model, v, True) 
        Xm, beta, tetha, v = out.X, out.beta, out.tetha, out.v
        
    X, W = Xm
    if tetha > 0:
        W = X.copy()
        
    if full_output:
        return out 
        
    return X, W, beta[1]

__all__ = ['ell']
        


