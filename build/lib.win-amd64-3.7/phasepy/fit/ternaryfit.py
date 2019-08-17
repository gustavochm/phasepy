from __future__ import division, print_function, absolute_import
import numpy as np
from ..equilibrium import haz
from ..actmodels import virialgama, nrtlter
from .fitmulticomponent import fobj_elv, fobj_ell, fobj_hazt

def fobj_nrtlrkt(D,Xexp,Wexp,Yexp,Texp,Pexp,mezcla, good_initial = True):
    
    n = len(Pexp)
    mezcla.rkt(D)
    
    vg = virialgama(mezcla, actmodel = nrtlter)
    x = np.zeros_like(Xexp)
    w = np.zeros_like(Wexp)
    y = np.zeros_like(Yexp)
    
    for i in range(n):
        x[:,i], w[:,i], y[:,i] = haz(Xexp[:,i],Wexp[:,i],Yexp[:,i],
                             Texp[i],Pexp[i],
                             vg, good_initial)
    
        
    error = ((np.nan_to_num(x)-Xexp)**2).sum()
    error += ((np.nan_to_num(w)-Wexp)**2).sum()
    error += ((np.nan_to_num(y)-Yexp)**2).sum()
    
    return error/n

def fobj_nrtlt(inc, mezcla, dataelv = None, dataell = None, dataellv = None,
               alpha_fixed = False, Tdep = False):
    
    if alpha_fixed:
        a12 = a13 = a23 = 0.2
        if Tdep:
            g12, g21, g13, g31, g23, g32, g12T, g21T, g13T, g31T, g23T, g32T = inc
            gT = np.array([[0, g12T, g13T],
                  [g21T, 0, g23T],
                  [g31T, g32T, 0]])
        else:
            g12, g21, g13, g31, g23, g32 = inc
            gT = None
    else:   
        if Tdep:
            g12, g21, g13, g31, g23, g32, g12T, g21T, g13T, g31T, g23T, g32T, a12, a13, a23 = inc
            gT = np.array([[0, g12T, g13T],
                  [g21T, 0, g23T],
                  [g31T, g32T, 0]])
        else:
            g12, g21, g13, g31, g23, g32, a12, a13, a23 = inc
            gT = None
    
    
    g = np.array([[0, g12, g13],
                  [g21, 0, g23],
                  [g31, g32, 0]])
    
    alpha = np.array([[0, a12, a13],
                      [a12, 0, a23],
                      [a13, a23, 0]])

    mezcla.NRTL(alpha, g, gT) 
    modelo = virialgama(mezcla)
    
    error = 0
    
    if dataelv is not None:
        error += fobj_elv(modelo, *dataelv)
    if dataell is not None:
        error += fobj_ell(modelo, *dataell)
    if dataellv is not None:
        error += fobj_hazt(modelo, *dataellv)
    return error
    

def fobj_kijt(inc, eos, mezcla, dataelv = None, dataell = None, dataellv = None):
    
    k12, k13, k23 = inc
    Kij=np.array([[0, k12, k13],
                  [k12, 0, k23],
                  [k13, k23, 0 ]])
    mezcla.kij_cubica(Kij)
    modelo = eos(mezcla)
    
    error = 0
    
    if dataelv is not None:
        error += fobj_elv(modelo, *dataelv)
    if dataell is not None:
        error += fobj_ell(modelo, *dataell)
    if dataellv is not None:
        error += fobj_hazt(modelo, *dataellv)
    return error