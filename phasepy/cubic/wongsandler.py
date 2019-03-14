from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import nrtl, wilson, nrtlter, rk, unifac
from ..constants import R


def ws(X, T, ai, bi, C, Kij, ActModel, parameter):
    
    #Acivity coefficient
    lngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama,X)

    RT = R*T
    #Mixrule parameters
    ei = ai/(bi*RT)    
    biaiRT =  bi - ai/RT
    abij = np.add.outer(biaiRT, biaiRT)/2
    abij *= (1. - Kij)
    xbi_ai = X*abij
    Q  = np.sum(xbi_ai.T*X)
    dQ = 2*np.sum(xbi_ai , axis=1)
    D = Gex/C + np.dot(X, ei)
    D1 = 1. - D
    dD = ei + lngama/C
    
    # Mixture parameters
    bm = Q/D1
    am = bm*D*RT
    em = am/(bm*RT)
    #Partial Molar properties
    bp = dQ/D1 - Q/D1**2 * (1 - dD)
    ap =  RT*(D*bp + bm*dD) - am
    ep = em*(1 + ap/am - bp/bm)
    return am, bm, ep, ap, bp

def ws_nrtl(X, T, ai, bi, C, Kij, alpha, g, g1):
    parameter = (alpha, g, g1)
    am, bm, ep, ap, bp = ws(X, T, ai, bi, C, Kij, nrtl, parameter)
    return am, bm, ep, ap, bp

def ws_nrtlt(X, T, ai, bi, C, Kij, alpha, g, g1, D):
    parameter = (alpha, g, g1, D)
    am, bm, ep, ap, bp = ws(X, T, ai, bi, C, Kij, nrtlter, parameter)
    return am, bm, ep, ap, bp

def ws_wilson(X, T, ai, bi, C, Kij, Aij, vl):
    parameter=(Aij,vl)
    am, bm, ep, ap, bp = ws(X, T, ai, bi, C, Kij, wilson, parameter)
    return am, bm, ep, ap, bp

def ws_rk(X, T, ai, bi, C, Kij, Crk, Crk1, combinatory):
    parameter=(Crk, Crk1, combinatory)
    am, bm, ep, ap, bp = ws(X, T, ai, bi, C, Kij, rk, parameter)
    return am, bm, ep, ap, bp
    

def ws_unifac(X,T,ai,bi, C, Kij, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    parameter = (qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2)
    am, bm, ep, ap, bp = ws(X,T,ai,bi, C, Kij, unifac, parameter)
    return am,bm,ep,ap, bp
    
    
