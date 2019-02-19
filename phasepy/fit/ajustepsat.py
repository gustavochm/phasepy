from scipy.optimize import minimize 
import numpy as np
from ..cubic import prsv

def psat_obj(C, Pexp, Texp):
    return np.sum((Pexp-np.exp(C[0]-C[1]/(Texp+C[2])))**2)

#ajuste ecuacion de antoine 
def fit_ant(Texp, Pexp, x0 = [0,0,0]):
    ajuste = minimize(psat_obj, x0, args = (Pexp, Texp))
    return ajuste

#ajute alpha ecuacion cubica sv
def fobj_alpha(k,Texp,Pexp,cubic):
    cubic.k = k
    P = np.zeros_like(Pexp)
    for i in range(len(Texp)):
        P[i] = cubic.psat(Texp[i])
    return np.sum((P-Pexp)**2)

#ajuste ecuacion de alpha SV
def fit_ksv(component ,Texp, Pexp, ksv0 = [1,0]):
    cubic = prsv(component)
    ajuste = minimize(fobj_alpha, ksv0, args = (Texp, Pexp, cubic ))
    return ajuste