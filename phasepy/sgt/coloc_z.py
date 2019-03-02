from __future__ import division, print_function, absolute_import
import numpy as np
from .tensionresult import TensionResult
from ..math import gauss, colocAB
from scipy.optimize import root
from .cijmix_cy import cmix_cy

def fobj_z_newton(rointer, Binter, dro20, dro21, mu0, T, cij, n, nc, model):
    rointer = rointer.reshape([nc, n])
    dmu = np.zeros([n, nc])

    for i in range(n):
        dmu[i] = model.muad(rointer[:,i], T)
    dmu -= mu0
    dmu = dmu.T
    
    dro2 = np.matmul(rointer,Binter.T)
    dro2 += dro20 
    dro2 += dro21
    
    ter1 = np.matmul(cij,dro2)
    fo = ter1 - dmu
    return fo.flatten()


def ten_sgt(ro1, ro2, Tsat, Psat, model, ro0 = 'linear',
            z0 = 5, dz = 1.5, itmax = 10, n = 20, full_output = False ):
    
    z = z0
    nc = model.nc
    
    #Dimensionless Variables
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = ro1*rofactor
    ro2a = ro2*rofactor

    
    cij = model.ci(Tsat)
    cij /= cij[0,0]
    dcij = np.linalg.det(cij)
    if np.isclose(dcij, 0):
        raise Exception('Determinant of influence parameters matrix is: {}'.format(dcij))
    
    #Chemical potential
    mu0 = model.muad(ro1a, Tsat)
    
    #Nodes and weights of integration
    roots, weights = gauss(n)
    rootsf = np.hstack([0. ,roots, 1.])
    #Coefficent matrix for derivatives
    A, B = colocAB(rootsf)
       
    #Initial profiles
    if ro0 == 'linear':
        #Linear Profile
        pend = (ro2a - ro1a)
        b = ro1a
        pfl = (np.outer(roots, pend) + b)
        rointer = (pfl.T).copy()
    elif ro0 == 'hyperbolic':
        #Hyperbolic profile
        inter = 8*roots - 4
        thb = np.tanh(2*inter)
        pft = np.outer(thb,(ro2a - ro1a))/2 + (ro1a + ro2a)/2
        rointer = pft.T
    elif isinstance(ro0,  np.ndarray):
        #Check dimensiones
        if ro0.shape[0] == nc and ro0.shape[1] == n:
            rointer = ro0.copy()
            rointer *= rofactor
        else:
            raise Exception('Shape of initial value must be nc x n')

    error = 1.
    it = 0
    ten_old = 0.
    tol = 1e-3
    while error > tol and it < itmax:
        it += 1
        zad = z*zfactor
        Ar = A/zad
        Br = B/zad**2

        Binter = Br[1:-1, 1:-1]
        B0 = Br[1:-1, 0]
        B1 = Br[1:-1, -1]
        dro20 = np.outer(ro1a, B0) #cte
        dro21 = np.outer(ro2a, B1) #cte

        Ainter = Ar[1:-1, 1:-1]
        A0 = Ar[1:-1, 0]
        A1 = Ar[1:-1, -1]
        dro10 = np.outer(ro1a, A0) #cte
        dro11 = np.outer(ro2a, A1) #cte

        sol = root(fobj_z_newton, rointer.flatten(), method = 'lm',
                   args = (Binter, dro20, dro21, mu0, Tsat, cij, n, nc, model))

        rointer = sol.x
        rointer = rointer.reshape([nc, n])

        dro = np.matmul(rointer,Ainter.T)
        dro += dro10 
        dro += dro11
        
        suma = cmix_cy(dro, cij)
        dom = np.zeros(n)
        for k in range(n):
            dom[k] = model.dOm(rointer[:,k], Tsat, mu0, Pad)
        dom[dom < 0] = 0.
        intten=np.nan_to_num(np.sqrt(2*suma*dom))
        ten = np.dot(intten, weights)
        ten *= zad
        ten *= tenfactor
        error = np.abs(ten - ten_old)
        ten_old = ten
        z += dz
    success = sol.success and (error < tol)
    
    if full_output: 

        znodes = (z-dz) * rootsf
        ro = np.insert(rointer, 0, ro1a, axis = 1)
        ro = np.insert(ro, n+1, ro2a, axis = 1)
        ro /= rofactor
        dictresult = {'tension' : ten, 'ro': ro, 'z' : znodes,
        'GPT' : np.hstack([0, dom, 0]),
        'success' : success, 
        'message' : sol.message,
        'fun_norm' : np.linalg.norm(sol.fun),
        'error':error, 'iter':it}
        out = TensionResult(dictresult)
        return out
    
    return ten