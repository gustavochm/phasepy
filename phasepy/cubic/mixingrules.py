import numpy as np
from ..actmodels import nrtl, wilson, nrtlter, rk, unifac
from ..constants import R


#Regla de mezclado MHV-NRTL
def U_mhv(em,c1,c2):
    ter1 = em-c1-c2
    ter2 = c1*c2+em
    
    Umhv = ter1 - np.sqrt(ter1**2 - 4*ter2)
    Umhv /= 2
    
    dUmhv = 1 - 0.5*(ter1**2 - 4*ter2)**-0.5*(-2*ter1-4)
    dUmhv /= 2
    
    return Umhv, dUmhv

def f0_mhv(em,zm,c1,c2):
    
    Umhv, dUmhv = U_mhv(em,c1,c2)

    f0 = (-1-np.log(Umhv-1)-(em/(c1-c2)) * np.log((Umhv + c1)/(Umhv + c2)))
    f0 -= zm

    df0 = -dUmhv/(Umhv-1)-(1/(c1-c2))*np.log((Umhv+c1)/(Umhv+c2))
    df0 += dUmhv*em/((Umhv+c1)*(Umhv+c2))    
    return f0, df0

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
    Regla de mezcla avanzada a limite presion cero mhv
    
    Parametros
    ----------
    X :[x1, x2, ..., xc]
    T: temperatura en K
    a_puros :  en bar cm6/mol2
    b_puros :  en cm3/mol,
    g y alpha : parametros parametros de NRTL
    
    Salida : am (a de mezcla)
             bm (b de mezcla)
             ep (e parcial, e = a/(bRT) )
             ap (a parcial)
    '''
    e = ai/(bi*R*T)
    #volumen reducido puros
    U=(e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2
    #fugacidad adimensional puros a presion cero
    z=-1-np.log(U-1)-(e/(c1-c2))*np.log((U+c1)/(U+c2))
    # fugacidad mezcla
    bm = np.dot(bi, X)
    lngama = ActModel(X, T, *parameter)
    Gex = np.dot(lngama,X)
    
    bibm = bi/bm
    logbibm = np.log(bibm)
    
    zm = Gex + np.dot(z, X) - np.dot(logbibm,X)
    em, der = em_solver(X, e, zm, c1, c2)
    am=em*bm*R*T
    
    #fugacidades parciales
    zp = lngama + z - logbibm + bibm - 1.
    dedn = (zp-zm)/der
    #termino coheviso a parciales
    ap = am + em*(bi-bm)*R*T + dedn*bm*R*T
    ep = em*(1 + ap/am - bi/bm)
    return am, bm, ep, ap

def mhv_nrtl(X, T, ai, bi, c1, c2, g, alpha, g1):
    '''
    Regla de mezcla avanzada a limite presion cero mhv con modelo nrtl
    
    Parametros
    ----------
    X :[x1, x2, ..., xc]
    T: temperatura en K
    a_puros :  en bar cm6/mol2
    b_puros :  en cm3/mol,
    g y alpha : parametros parametros de NRTL
    
    Salida : am (a de mezcla)
             bm (b de mezcla)
             ep (e parcial, e = a/(bRT) )
             ap (a parcial)
    '''
    parameter=(g,alpha, g1)
    am,bm,ep,ap = mhv(X,T,ai,bi,c1,c2,nrtl,parameter)
    return am,bm,ep,ap

def mhv_wilson(X, T, ai, bi, c1, c2, Aij, vl):
    '''
    Regla de mezcla avanzada a limite presion cero mhv con modelo de wilson
    
    Parametros
    ----------
    X :[x1, x2, ..., xc]
    T: temperatura en K
    a_puros :  en bar cm6/mol2
    b_puros :  en cm3/mol,
    g y alpha : parametros parametros de NRTL
    
    Salida : am (a de mezcla)
             bm (b de mezcla)
             ep (e parcial, e = a/(bRT) )
             ap (a parcial)
    '''
    parameter=(Aij,vl)
    am,bm,ep,ap = mhv(X,T,ai,bi,c1,c2,wilson,parameter)
    return am,bm,ep,ap 

def mhv_nrtlt(X,T, ai, bi, c1, c2, g, alpha, g1, D):
    '''
    Regla de mezcla avanzada a limite presion cero mhv con modelo de nrtl
    modficado
    
    Parametros
    ----------
    X :[x1, x2, ..., xc]
    T: temperatura en K
    a_puros :  en bar cm6/mol2
    b_puros :  en cm3/mol,
    g y alpha : parametros parametros de NRTL
    
    Salida : am (a de mezcla)
             bm (b de mezcla)
             ep (e parcial, e = a/(bRT) )
             ap (a parcial)
    '''
    parameter=(g, alpha, g1, D)
    am,bm,ep,ap = mhv(X, T, ai, bi, c1, c2, nrtlter, parameter)
    return am,bm,ep,ap 



def mhv_rk(X, T, ai, bi, c1, c2, C, C1, combinatoria):
    '''
    Regla de mezcla avanzada a limite presion cero mhv con modelo nrtl
    
    Parametros
    ----------
    X :[x1, x2, ..., xc]
    T: temperatura en K
    a_puros :  en bar cm6/mol2
    b_puros :  en cm3/mol,
    g y alpha : parametros parametros de NRTL
    
    Salida : am (a de mezcla)
             bm (b de mezcla)
             ep (e parcial, e = a/(bRT) )
             ap (a parcial)
    '''
    parameter=(C, C1, combinatoria)
    am,bm,ep,ap = mhv(X, T, ai, bi, c1, c2, rk, parameter)
    return am,bm,ep,ap

def mhv_unifac(X,T,ai,bi,c1,c2, qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2):
    '''
    Regla de mezcla avanzada a limite presion cero mhv con modelo nrtl
    
    Parametros
    ----------
    X :[x1, x2, ..., xc]
    T: temperatura en K
    a_puros :  en bar cm6/mol2
    b_puros :  en cm3/mol,
    g y alpha : parametros parametros de NRTL
    
    Salida : am (a de mezcla)
             bm (b de mezcla)
             ep (e parcial, e = a/(bRT) )
             ap (a parcial)
    '''
    parameter = (qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2)
    am,bm,ep,ap = mhv(X,T,ai,bi,c1,c2, unifac, parameter)
    return am,bm,ep,ap

#Regla de mezclado QMR      
def qmr(X, T, ai, bi, Kij):
    '''
    Regla de mezcla avanzada qmr
    
    Parametros
    ----------
    X :[x1, x2, ..., xc]
    T: temperatura en K
    a_puros :  en bar cm6/mol2
    b_puros :  en cm3/mol,
    Kij : matriz de parametros de correccion de regla mezclado cuadratica
    
    Salida : am (a de mezcla)
             bm (b de mezcla)
             ep (e parcial, e = a/(bRT) )
             ap (a parcial)
    '''
    

    aij=np.sqrt(np.outer(ai,ai))*(1-Kij)
    
    ax = aij*X
    #virial de mezcla
    am = np.sum(ax.T*X)
    #virial parcial
    ap = 2*np.sum(ax, axis=1) - am
    
    bm = np.dot(bi,X)
    em = am/(bm*R*T)
    ep = em*(1+ap/am-bi/bm)
    
    return am,bm,ep,ap