from scipy.optimize import minimize 
import numpy as np
from ..cubic import prsveos

def psat_obj(C, Pexp, Texp):
    return np.sum((Pexp-np.exp(C[0]-C[1]/(Texp+C[2])))**2)

#Fit Antoine parameters
def fit_ant(Texp, Pexp, x0 = [0,0,0]):
    """
    fit_ant fit Antoine parameters, base exp
    
    Parameters
    ----------
    Texp: experimental temperature in K.
    Pexp : experimental pressure in bar.
    x0 : array_like, optional
        initial values.
    
    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize
    
    """
    fit = minimize(psat_obj, x0, args = (Pexp, Texp))
    return fit

def fobj_alpha(k,Texp,Pexp,cubic):
    cubic.k = k
    P = np.zeros_like(Pexp)
    for i in range(len(Texp)):
        P[i] = cubic.psat(Texp[i])
    return np.sum((P-Pexp)**2)

#fit  SV alpha
def fit_ksv(component ,Texp, Pexp, ksv0 = [1,0]):
    """
    fit_ksv fit PRSV alpha 
    
    Parameters
    ----------
    component : object
        created with component class
    Texp : array_like
        experimental temperature in K.
    Pexp : array_like
        experimental pressure in bar.
    ks0 : array_like, optional
        initial values.
    
    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize
    
    """
    cubic = prsveos(component)
    fit = minimize(fobj_alpha, ksv0, args = (Texp, Pexp, cubic ))
    return fit