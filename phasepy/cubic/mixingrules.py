import numpy as np
from ..actmodels import nrtl, wilson, nrtlter, rk, unifac
from ..constants import R


#Modified Huron Vidal Mixrule

#adimentional volume of mixture
def U_mhv(em,c1,c2):
    ter1 = em-c1-c2
    ter2 = c1*c2+em
    
    Umhv = ter1 - np.sqrt(ter1**2 - 4*ter2)
    Umhv /= 2
    
    dUmhv = 1 - 0.5*(ter1**2 - 4*ter2)**-0.5*(-2*ter1-4)
    dUmhv /= 2
    
    return Umhv, dUmhv

#objetive function MHV
def f0_mhv(em,zm,c1,c2):
    
    Umhv, dUmhv = U_mhv(em,c1,c2)

    f0 = (-1-np.log(Umhv-1)-(em/(c1-c2)) * np.log((Umhv + c1)/(Umhv + c2)))
    f0 -= zm

    df0 = -dUmhv/(Umhv-1)-(1/(c1-c2))*np.log((Umhv+c1)/(Umhv+c2))
    df0 += dUmhv*em/((Umhv+c1)*(Umhv+c2))    
    return f0, df0

#adimentional paramter solver with newton method
def em_solver(X, e, zm,c1,c2):
    em = np.dot(X, e)
    it = 0.
    f0, df0 = f0_mhv(em,zm,c1,c2)
    error = 1.
    while error > 1e-6 and it < 30: 
        it += 1
        de = f0 / df0
        em -= de
        error = np.abs(de)
        f0, df0 = f0_mhv(em,zm,c1,c2)
    return em, df0
        
        
def mhv(X, T, ai, bi, c1, c2, ActModel, parameter):
    '''
    Modified Huron vidal mixrule
    
    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    ActModel: function, activity coefficient model.
    parameter : tuple of parameters to evaluate ActModel.

    
    Out :
    am (mixture a term)
    bm (mixture b term)
    ep (e partial, e = a/(bRT) )
    ap (a partial molar)
    '''
    e = ai/(bi*R*T)
    #Pure component reduced volume
    U=(e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2
    #Pure component fugacity at zero pressure
    z=-1-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))
    # Mixture Fugacity
    bm = np.dot(bi, X)
    #Acivity coefficient
    lngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama,X)
    
    bibm = bi/bm
    logbibm = np.log(bibm)
    
    zm = Gex + np.dot(z, X) - np.dot(logbibm,X)
    em, der = em_solver(X, e, zm, c1, c2)
    am=em*bm*R*T
    
    #partial fugacity
    zp = lngama + z - logbibm + bibm - 1.
    dedn = (zp-zm)/der
    #partial attractive term
    ap = am + em*(bi-bm)*R*T + dedn*bm*R*T
    #partial adimnetional term
    ep = em*(1 + ap/am - bi/bm)
    return am, bm, ep, ap

def mhv_nrtl(X, T, ai, bi, c1, c2, alpha, g, g1):
    '''
    Modified Huron vidal mixrule with nrtl model
    
    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    alpha, g, g1 : array_like, parameters to evaluate nrtl model

    
    Out :
    am (mixture a term)
    bm (mixture b term)
    ep (e partial, e = a/(bRT) )
    ap (a partial molar)
    '''
    parameter = (alpha, g, g1)
    am,bm,ep,ap = mhv(X,T,ai,bi,c1,c2,nrtl,parameter)
    return am,bm,ep,ap

def mhv_wilson(X, T, ai, bi, c1, c2, Aij, vl):
    '''
    Modified Huron vidal mixrule with wilson model
    
    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    Aij : array_like, parameters to evaluate wilson model
    vl : function to evaluate pure liquid volumes

    
    Out :
    am (mixture a term)
    bm (mixture b term)
    ep (e partial, e = a/(bRT) )
    ap (a partial molar)
    '''
    parameter=(Aij,vl)
    am,bm,ep,ap = mhv(X,T,ai,bi,c1,c2,wilson,parameter)
    return am,bm,ep,ap 

def mhv_nrtlt(X,T, ai, bi, c1, c2, alpha, g, g1, D):
    '''
    Modified Huron vidal mixrule with modified ternary nrtl model
    
    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    alpha, g, g1 : array_like, parameters to evaluate nrtl model
    D : array_like, parameter to evaluate ternary term.

    
    Out :
    am (mixture a term)
    bm (mixture b term)
    ep (e partial, e = a/(bRT) )
    ap (a partial molar)
    '''
    parameter=(alpha, g, g1, D)
    am,bm,ep,ap = mhv(X, T, ai, bi, c1, c2, nrtlter, parameter)
    return am,bm,ep,ap 



def mhv_rk(X, T, ai, bi, c1, c2, C, C1, combinatory):
    '''
    Modified Huron vidal mixrule with Redlich Kister model
    
    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    C, C1 : array_like, parameters to evaluate Redlich Kister polynomial
    combinatory: array_like, array_like, contains info of the order of polynomial
            coefficients by pairs.

    
    Out :
    am (mixture a term)
    bm (mixture b term)
    ep (e partial, e = a/(bRT) )
    ap (a partial molar)
    '''
    parameter=(C, C1, combinatory)
    am,bm,ep,ap = mhv(X, T, ai, bi, c1, c2, rk, parameter)
    return am,bm,ep,ap

def mhv_unifac(X,T,ai,bi,c1,c2, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    '''
    Modified Huron vidal mixrule with UNIFAC model
    
    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    T: Absolute temperature in K
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    c1, c2: cubic eos constants
    qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2: parameters to evaluae modified
        Dortmund UNIFAC.

    
    Out :
    am (mixture a term)
    bm (mixture b term)
    ep (e partial, e = a/(bRT) )
    ap (a partial molar)
    '''
    parameter = (qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2)
    am,bm,ep,ap = mhv(X,T,ai,bi,c1,c2, unifac, parameter)
    return am,bm,ep,ap

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
    
    return am,bm,ep,ap