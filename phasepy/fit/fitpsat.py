from scipy.optimize import minimize 
import numpy as np
from ..cubic import prsv

def psat_obj(C, Pexp, Texp):
    return np.sum((Pexp-np.exp(C[0]-C[1]/(Texp+C[2])))**2)

#Fit Antoine parameters
def fit_ant(Texp, Pexp, x0 = [0,0,0]):
    """
    fit_ant fit Antoine parameters, base exp
    
    Inputs
    ----------
    Texp: experimental temperature in K.
    Pexp : experimental pressure in bar.
    x0 : initial values.
    """
    ajuste = minimize(psat_obj, x0, args = (Pexp, Texp))
    return ajuste

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
    
    Inputs
    ----------
    component
    Texp: experimental temperature in K.
    Pexp : experimental pressure in bar.
    ks0 : initial values.
    """
    cubic = prsv(component)
    ajuste = minimize(fobj_alpha, ksv0, args = (Texp, Pexp, cubic ))
    return ajuste