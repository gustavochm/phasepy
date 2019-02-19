import numpy as np
from ..actmodels import virialgama, wilson, rkb
from .ajustemulticomponente import fobj_elv, fobj_ell, fobj_hazb

#ajuste ELV con wilson
def fobj_wilson(inc, mezcla, dataelv):
    """ 
    Funcion objetivo para optimizar parametros de modelo de Wilson para mezcla binaria 
    en ELV
    
    Parametros
    ----------
    inc : array_like
          a12, a21 para evaluar modelo de wilson
    Xexp : array_like
            matriz  de datos experimentales de la fase liquida
    Yexp : array_like
            matrix de datos experimentales de fase gaseosa
    Texp : array_like
         temperatura de equilibrio experimental
    Pexp : array_like
        Presion de equilibrio experimental
    mezcla : object
          objeto creado a partir de mezcla 
    """
    
    a12, a21 = inc
    a = np.array([[0,a12],[a21,0]])
    
    mezcla.wilson(a)
    modelo = virialgama(mezcla, actmodel = wilson)
    
    elv = fobj_elv(modelo, *dataelv)
    
    return elv

def fobj_nrtl(inc, mezcla, dataelv = None, dataell = None, dataellv = None,
              alpha_fixed = False, Tdep = False):
    
    if alpha_fixed:
        alpha = 0.2
        if Tdep:
            g12,g21,g12T, g21T=inc
            gT=np.array([[0,g12T],[g21T,0]])
        else:
            g12,g21=inc
            gT = None
    else:   
        if Tdep:
            g12,g21,g12T, g21T,alpha=inc
            gT=np.array([[0,g12T],[g21T,0]])
        else:
            g12,g21,alpha=inc
            gT = None
        
    g=np.array([[0,g12],[g21,0]])  
    mezcla.NRTL(alpha, g, gT) 
    modelo = virialgama(mezcla)
    
    error = 0.
    
    if dataelv is not None:
        error += fobj_elv(modelo, *dataelv)
    if dataell is not None:
        error += fobj_ell(modelo, *dataell)
    if dataellv is not None:
        error += fobj_hazb(modelo, *dataellv)
    return error

def fobj_kij(kij, eos, mezcla, dataelv = None, dataell = None, dataellv = None):
    
    Kij=np.array([[0, kij],[kij,0]])
    mezcla.kij_cubic(Kij)
    cubica=eos(mezcla)
    
    error = 0.
    if dataelv is not None:
        error += fobj_elv(cubica, *dataelv)
    if dataell is not None:
        error += fobj_ell(cubica, *dataell)
    if dataellv is not None:
        error += fobj_hazb(cubica, *dataellv)
    return error

def fobj_rkb(inc, mezcla, dataelv = None, dataell = None, dataellv = None, Tdep = False):
    
    if Tdep:
        c, c1 = np.split(inc,2)
    else:
        c = inc
        c1 = np.zeros_like(c)
    mezcla.rkb(c, c1)
    modelo = virialgama(mezcla, actmodel = rkb)
    
    error = 0.
    
    if dataelv is not None:
        error += fobj_elv(modelo, *dataelv)
    if dataell is not None:
        error += fobj_ell(modelo, *dataell)
    if dataellv is not None:
        error += fobj_hazb(modelo, *dataellv)
    return error




