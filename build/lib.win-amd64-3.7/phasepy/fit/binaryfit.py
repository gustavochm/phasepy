from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import virialgama, wilson, rk
from .fitmulticomponent import fobj_elv, fobj_ell, fobj_hazb
from scipy.optimize import minimize, minimize_scalar


def fobj_wilson(inc, mix, dataelv):
    
    a12, a21 = inc
    a = np.array([[0,a12],[a21,0]])
    
    mix.wilson(a)
    model = virialgama(mix, actmodel = wilson)
    
    elv = fobj_elv(model, *dataelv)
    
    return elv

def fit_wilson(x0, mix, dataelv, minimize_options = {}):
    """ 
    fit_wilson: attemps to fit wilson parameters to LVE 
    
    Parameters
    ----------
    x0 : array_like
        initial values a12, a21 in K
    mix: object
        binary mixture
    dataelv: tuple
        (Xexp, Yelv, Texp, Pexp)
    
    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize
    
    """
    fit = minimize(fobj_wilson, x0, args = (mix, dataelv), **minimize_options)
    return fit


def fobj_nrtl(inc, mix, dataelv = None, dataell = None, dataellv = None,
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
    mix.NRTL(alpha, g, gT) 
    model = virialgama(mix)
    
    error = 0.
    
    if dataelv is not None:
        error += fobj_elv(model, *dataelv)
    if dataell is not None:
        error += fobj_ell(model, *dataell)
    if dataellv is not None:
        error += fobj_hazb(model, *dataellv)
    return error

def fit_nrtl(x0, mix, dataelv = None, dataell = None, dataellv = None,
              alpha_fixed = False, Tdep = False, minimize_options = {}):
    """ 
    fit_nrtl: attemps to fit nrtl parameters to LVE, LLE, LLVE 
    
    Parameters
    ----------
    x0 : array_like
        initial values interaction parameters (and aleatory factor) 
    mix: object
        binary mixture
    dataelv: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    dataell: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    dataellv: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)
    alpha_fit: bool, optional
        if True fix aleatory factor to 0.2
    Tdep: bool, optional
        Wheter the energy parameters have a temperature dependence
    
    Notes
    -----
    
    if Tdep True parameters are treated as:
            a12 = a12_1 + a12T * T
            a21 = a21_1 + a21T * T
            
    if alpha_fixed true and Tdep True:
        x0 = [a12, a21, a12T, a21T, alpha]
    if alpha_fixed false and Tdep False:
        x0 = [a12, a21, alpha]
    if alpha_fixed True and Tdep False:
        x0 = [a12, a21]
        
    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize
    """
    fit = minimize(fobj_nrtl, x0, args = (mix, dataelv, dataell, dataellv,
              alpha_fixed, Tdep), **minimize_options)
    return fit

def fobj_kij(kij, eos, mix, dataelv = None, dataell = None, dataellv = None):
    

    Kij=np.array([[0, kij],[kij,0]])
    mix.kij_cubic(Kij)
    cubica=eos(mix)
    
    error = 0.
    if dataelv is not None:
        error += fobj_elv(cubica, *dataelv)
    if dataell is not None:
        error += fobj_ell(cubica, *dataell)
    if dataellv is not None:
        error += fobj_hazb(cubica, *dataellv)
    return error

def fit_kij(kij_bounds, eos, mix, dataelv = None, dataell = None, dataellv = None):
    
    """ 
    fit_kij: attemps to fit kij to LVE, LLE, LLVE 
    
    Parameters
    ----------
    kij0 : array_like
        initial value for kij
    eos : function
        cubic eos to fit kij for qmr mixrule
    mix: object
        binary mixture
    dataelv: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    dataell: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    dataellv: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp) 

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize          
            
    """
    fit = minimize_scalar(fobj_kij, kij_bounds, 
                          args = (eos, mix, dataelv, dataell, dataellv))
    return fit

def fobj_rk(inc, mix, dataelv = None, dataell = None, dataellv = None,
            Tdep = False):
    
    if Tdep:
        c, c1 = np.split(inc,2)
    else:
        c = inc
        c1 = np.zeros_like(c)
    mix.rk(c, c1)
    modelo = virialgama(mix, actmodel = rk)
    
    error = 0.
    
    if dataelv is not None:
        error += fobj_elv(modelo, *dataelv)
    if dataell is not None:
        error += fobj_ell(modelo, *dataell)
    if dataellv is not None:
        error += fobj_hazb(modelo, *dataellv)
    return error

def fit_rk(inc0, mix, dataelv = None, dataell = None,
           dataellv = None, Tdep = False, minimize_options = {}):
    """ 
    fit_rk: attemps to fit RK parameters to LVE, LLE, LLVE 
    
    Parameters
    ----------
    inc0 : array_like
        initial values to RK parameters
    mix: object
        binary mixture
    dataelv: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    dataell: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    dataellv: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)   
    Tdep : bool,
        wheter the parameter will have a temperature dependence
        
    Notes
    -----
    
    if Tdep true:
                    C = C' + C'T
    if Tdep true:
        inc0 = [C'0, C'1, C'2, ..., C'0T, C'1T, C'2T... ]      
    if Tdep flase: 
            inc0 = [C0, C1, C2...]
            
    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize
    """
    fit = minimize(fobj_rk, inc0 ,args = (mix, dataelv, dataell, dataellv,
                                          Tdep ), **minimize_options)
    return fit




