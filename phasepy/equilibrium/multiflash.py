

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import minimize
from ..math import gdem
from .equilibriumresult import EquilibriumResult

def multiflash_obj(inc, Z, K):
    
    inc[inc <= 1e-10] = 1e-10
    beta , tetha = np.array_split(inc,2)
    ninc = len(inc)
    nbeta = len(beta)
    
    betar = beta[1:]
    Kexp = K.T*np.exp(tetha)
    K1 = Kexp - 1. #nc x P - 1 
    sum1 = 1. + (K1*betar).T.sum(axis=0) # P - 1
    Xref = Z/sum1
    
    f1 = beta.sum() - 1.
    f2 = (K1.T*Xref).sum(axis=1)
    f3 = (betar*tetha) / (betar + tetha)
    f = np.hstack([f1,f2,f3])
    
    
    jac = np.zeros([ninc, ninc])
    Id = np.eye(nbeta-1)
    
    #f1 derivative
    f1db = np.ones_like(beta)
    jac[0,:nbeta] = f1db
    
    #f2 derivative
    df20 = Z*K1.T/sum1**2
    jac[1:nbeta,1:nbeta ] = - df20@K1
    aux = df20*(Kexp*betar).T
    jac[1:nbeta, nbeta:] = Id*((Kexp.T*Z/sum1).T.sum(axis=0))-aux@aux.T

    #f3 derivative
    f3db =  (tetha / (betar + tetha))**2
    f3dt = (betar/(betar + tetha))**2
    jac[nbeta:, 1:nbeta] = Id*f3db
    jac[nbeta:, nbeta:] = Id*f3dt
    
    return f, jac, Kexp, Xref

def gibbs_obj(ind, phases, Z, T, P, modelo, v0):
    
    nfase = len(phases)
    nc = modelo.nc
    ind = ind.reshape(nfase - 1 , nc)
    dep = Z - ind.sum(axis= 0)
    
    X = np.zeros((nfase, nc))
    X[1:] = ind
    X[0] = dep
    X[X < 1e-8] = 1e-8
    n = X.copy()
    X = (X.T/X.sum(axis=1)).T
    
    lnphi = np.zeros_like(X)
    global vg
    vg = v0.copy()
    for i, state in enumerate(phases):
        lnphi[i], vg[i]  = modelo.logfugef(X[i], T, P, state, v0[i])
    fug = np.nan_to_num(np.log(X) + lnphi)
    G = np.dot(n,fug)
    dG = (fug[1:] - fug[0]).flatten()
    return G, dG


def multiflash(X0, betatetha, equilibrium, z, T, P, model, v0 = [None], full_output = False):

    
    """
    multiflash (z,T,P) -> (x,w,y,beta)
    
    Parameters
    ----------
    X0 : array_like
        (n_phases, nc) initial guess
    betatheta : array_like
        phase fractions and stability array 
    equilibrium : list
        'L' for liquid, 'V' for vaopur phase. ['L', 'L', 'V'] for LLVE
    z : array_like
        overall system composition
    T : float
        absolute temperature in K.
    P : float
        pressure in bar
    model : object
        created from mixture, eos and mixrule 
    v0 : list, optional
        if supplied volume used as initial value to compute fugacities
    full_output: bool, optional
        wheter to outputs all calculation info
    
    Returns
    -------
    
    """
    

    

    nfase = len(equilibrium)  
    
    nc = model.nc
    if np.all(X0.shape == (nfase, nc)) or len(z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')
        
    tol = 1e-10
    error = 1
    it = 0
    itacc = 0
    n = 5
    
    X = X0.copy()
    lnphi = np.zeros_like(X) #crea matriz donde se almacenan los ln coef de fug
    
    if len(v0) != 1 and len(v0) != nfase:
        v0 *= nfase
    v = v0.copy()
        
    
    for i, estado in enumerate(equilibrium):
        lnphi[i], v[i] = model.logfugef(X[i], T, P, estado, v0[i])
        
    lnK = lnphi[0] - lnphi[1:]
    K = np.exp(lnK)
    
    x = betatetha
    
    while error > tol and itacc < 4:
        it += 1
        lnK_old = lnK.copy()
        
        itin = 0
        ef = 1
        ex = 1
        
        while ef > 1e-6 and ex > 1e-6 and itin < 20:
            itin += 1
            f, jac, Kexp, Xref = multiflash_obj(x, z, K)
            dx = np.linalg.solve(jac,-f)
            x += dx
            ef = np.linalg.norm(f)
            ex = np.linalg.norm(dx)
            
    
        x[x <= 1e-10] = 0
        beta , tetha = np.array_split(x,2)
        beta /= beta.sum()
    
        #acualizacion y normalizacion de las composicion
        X[0] = Xref
        X[1:] = Xref*Kexp.T
        X = np.abs(X)
        X = (X.T/X.sum(axis=1)).T
        
        #Recalculo de las coef de fugacidad 
        for i, estado in enumerate(equilibrium):
            lnphi[i], v[i] = model.logfugef(X[i], T, P, estado, v[i])
        lnK = lnphi[0] - lnphi[1:]
        
        #sustitucion sucesiva acelarada
        if it == (n-3):
            lnK3 = lnK.flatten()
        elif it == (n-2):
            lnK2 = lnK.flatten()
        elif it == (n-1):
            lnK1 = lnK.flatten()
        elif it == n:
            it = 0 
            itacc += 1
            lnKf = lnK.flatten()
            dacc = gdem(lnKf, lnK1, lnK2, lnK3).reshape(lnK.shape)
            lnK += dacc
          
        K = np.exp(lnK)
        error = ((lnK-lnK_old)**2).sum()

    if error > tol and itacc == 4 and ef < 1e-6 and np.all(tetha >0):
        global vg
        ind0 = (X.T*beta).T[1:].flatten()
        ind1 = minimize(gibbs_obj, ind0, 
                       args = (equilibrium, z, T, P, model, v), jac = True,
                      method = 'BFGS' )
        v = vg.copy()
        it += ind1.nit
        error = ind1.fun
        nc = model.nc
        ind = ind1.x.reshape(nfase - 1 , nc)
        dep = z - ind.sum(axis= 0)
        X[1:] = ind
        X[0] = dep
        beta = X.sum(axis=1)
        X[beta > 0] = (X[beta > 0].T/beta[beta > 0]).T
        X = (X.T/X.sum(axis=1)).T
        
    if full_output:
        sol = {'T' : T, 'P': P, 'error_outer':error, 'error_inner': ef, 'iter':it,
               'beta': beta, 'tetha': tetha,                
               'X' : X, 'v':v, 'states' : equilibrium}
        out = EquilibriumResult(sol)
        return out 
        
    return X, beta, tetha


