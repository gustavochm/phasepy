import numpy as np
from scipy.optimize import fsolve
from ..math import lobatto, colocAB

def fobj_beta0(ro, ro_s, s, T, mu0, sqrtci, model):     
    nc = model.nc
    # se ingresa el vector densidades adimensionales sin la variable independiente
    # esta se tiene que agregar
    ro = np.insert(ro, s, ro_s)
    dmu = model.muad(ro, T) - mu0
    
    f1 = sqrtci[s]*dmu
    f2 = sqrtci*dmu[s]
    
    return (f1-f2)[np.arange(nc) != s ]

def ten_beta0_reference(ro1, ro2, Tsat, Psat, model, s = 0, n = 100, full_output = False ):

    nc = model.nc
    
    #adimensionalizar variables 
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = ro1*rofactor
    ro2a = ro2*rofactor
    
    cij = model.ci(Tsat)
    cij /= cij[0,0]
    ci = np.diag(cij)
    sqrtci = np.sqrt(ci)
    
    mu0 = model.muad(ro1a, Tsat)
    
    #pesos y raices de integracion
    roots, weights  = lobatto(n)

    
    ro_s = (ro2a[s]-ro1a[s])*roots + ro1a[s] #puntos de integracion 
    wreal = weights*(ro2a[s]-ro1a[s])#pesos de integracion
    
        
    #definicion matrices colocaciones A para primera derivada, B para segunda
    A,B = colocAB(ro_s)
    
    rodep = np.zeros([nc-1, n])
    rodep[:,0] = ro1a[np.arange(nc) !=s]
    for i in range(1,n):
        rodep[:,i]=fsolve(fobj_beta0,rodep[:,i-1],args=(ro_s[i],s, Tsat, mu0, sqrtci, model))
    ro = np.insert(rodep, s, ro_s, axis = 0)
    dro = rodep@A.T
    dro = np.insert(dro, s, np.ones(n), axis = 0)
    suma = np.zeros(n)
    dom = np.zeros(n)
    for k in range(n):
        for i in range(nc):
            for j in range(nc):
                suma[k] += cij[i,j]*dro[i,k]*dro[j,k]
        dom[k] = model.dOm(ro[:,k], Tsat, mu0, Pad)
    dom[0], dom[-1] = 0., 0.
    
    intten=np.nan_to_num(np.sqrt(suma*(2*dom)))
    ten = np.dot (intten, wreal)
    
    ten*= tenfactor
    
    if full_output:
        #Resolucion del perfil
        with np.errstate(divide='ignore'):
            intz = (np.sqrt(suma/(2*dom)))
        intz[np.isinf(intz)] = 0
        z=np.cumsum(intz*wreal)
        z /= zfactor
        ro /= rofactor
        return ten, ro, z
    
    return ten