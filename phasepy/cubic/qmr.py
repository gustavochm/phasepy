from __future__ import division, print_function, absolute_import
import numpy as np
from ..constants import R


#Quadratic mixrule      
def qmr(X, T, ai, bi, Kij):
    '''
    Quadratic mixrule QMR
    
    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    Kij : matrix of interaction parameters

    
    Out :
    am (mixture a term)
    bm (mixture b term)
    ep (e partial, e = a/(bRT) )
    ap (a partial molar)
    bp (b partial molar)
    '''

    aij=np.sqrt(np.outer(ai,ai))*(1-Kij)
    
    ax = aij*X
    #atractive term of mixture
    am = np.sum(ax.T*X)
    #atrative partial term 
    ap = 2*np.sum(ax, axis=1) - am
    
    bm = np.dot(bi,X)
    em = am/(bm*R*T)
    ep = em*(1+ap/am-bi/bm)
    
    return am, bm, ep, ap, bi