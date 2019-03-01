import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import cumtrapz
from .cijmix_cy import cmix_cy
from .tensionresult import TensionResult


def fobj_sk(inc, spath, T, mu0, ci, sqrtci, model):   
    ro = inc[:-1]
    alpha = inc[-1]
    mu = model.muad(ro, T)
    obj = np.zeros_like(inc)
    obj[:-1] = mu - (mu0 + alpha*sqrtci)
    obj[-1] = spath - sqrtci.dot(ro)
    return obj

def ten_beta0_sk(ro1, ro2, Tsat, Psat, model, n = 200, full_output = False ):
    
    nc = model.nc
    
    #Dimensionless variables
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = ro1*rofactor
    ro2a = ro2*rofactor
    
    cij = model.ci(Tsat)
    cij /= cij[0,0]
    ci = np.diag(cij)
    sqrtci = np.sqrt(ci)
    
    mu0 = model.muad(ro1a, Tsat)
    
    #Path function
    s0 = sqrtci.dot(ro1a)
    s1 = sqrtci.dot(ro2a)
    spath = np.linspace(s0, s1, n)
    ro = np.zeros([nc,n])
    alphas = np.zeros(n)
    ro[:,0] = ro1a
    ro[:,-1] = ro2a
    
    deltaro = ro2a - ro1a
    i = 1
    r0 = ro1a + deltaro * (spath[i]-s0)
    r0 = np.hstack([r0, 0])
    ro0 = fsolve(fobj_sk,r0,args=(spath[i], Tsat, mu0, ci, sqrtci, model))
    ro[:,i] = ro0[:-1]
    
    for i in range(1,n):
        ro0 = fsolve(fobj_sk, ro0, args=(spath[i], Tsat, mu0, ci, sqrtci, model))
        alphas[i] = ro0[-1]
        ro[:,i] = ro0[:-1]
        
    #Derivatives respect to path function
    drods = np.gradient(ro, spath, edge_order = 2, axis = 1)
    
    suma = cmix_cy(drods, cij)
    dom = np.zeros(n)
    for k in range(1, n - 1):
        dom[k] = model.dOm(ro[:,k], Tsat, mu0, Pad)

    
    integral = np.nan_to_num(np.sqrt(2*dom*suma))
    tension = np.abs(np.trapz(integral, spath))
    tension *= tenfactor
    
    if full_output:
        #Zprofile
        with np.errstate(divide='ignore'):
            intz = (np.sqrt(suma/(2*dom)))
        intz[np.isinf(intz)] = 0
        z = np.abs(cumtrapz(intz,spath, initial = 0))
        z /= zfactor
        ro /= rofactor
        dictresult = {'tension' : tension, 'ro': ro, 'z' : z,
        'GPT' : dom, 'path': spath, 'alphas' : alphas}
        out = TensionResult(dictresult)
        return out
    
    return tension

