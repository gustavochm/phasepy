import numpy as np
from scipy.optimize import fsolve
from .multiflash import multiflash
from .equilibriumresult import EquilibriumResult


def haz_objb(inc,T_P,tipo,modelo, index, equilibrio, v0):
    X0 = inc[:-1].reshape(3,2)
    P_T = inc[-1]
    
    if tipo == 'T':
        P=P_T
        T=T_P
    elif tipo == 'P':
        T=P_T
        P=T_P
    

    nc = modelo.nc
    X = np.zeros((3, nc))
    X[:,index] = X0
    lnphi = np.zeros_like(X)
    
    for i, estado in enumerate(equilibrio):
        lnphi[i], _ = modelo.logfugef(X[i], T, P, estado, v0[i])
        
    lnK = lnphi[0] - lnphi[1:]
    K = np.exp(lnK)
    return np.hstack([(K[:,index]*X0[0] - X0[1:]).flatten(), X0.sum(axis=1) - 1.])

def haz_pb(X0,P_T,T_P,tipo,modelo,index, equilibrio, v0 = [None, None, None]):
    sol = fsolve(haz_objb, np.hstack([X0.flatten(),P_T]),args = (T_P,tipo,modelo, index, equilibrio, v0))
    
    var = sol[-1]
    X = sol[:-1].reshape(3,2)
    
    return X, var

def haz_objt(inc, T, P, model, v0 = [None, None, None]):
    
    X, W, Y = np.split(inc,3)
    
    global vx, vw, vy
    vx, vw, vy = v0
    
    fugX, vx = model.logfugef(X, T, P, 'L', vx)
    fugW, vw = model.logfugef(W, T, P, 'L', vw)
    fugY, vy = model.logfugef(Y, T, P, 'V', vy)
    
    K1 = np.exp(fugX-fugY)
    K2 = np.exp(fugX-fugW)
    
    return np.hstack([K1*X-Y,K2*X-W, X.sum()-1, Y.sum()-1,W.sum()-1])
    

def haz(X0, W0, Y0, T, P, model, good_initial = False,
         v0 = [None, None, None], full_output = False):
    
    """
    haz (T,P) -> (x, w, y)
    
    Inputs
    ----------
    
    X0 : array_like, guess composition of liquid 1
    W0 : array_like, guess composition of liquid 2
    Y0 : array_like, guess composition of vapour 1
    T : absolute temperature in K.
    P : pressure in bar
    
    model : object created from mixture, eos and mixrule 
    
    good_initial: bool, if True skip Gupta's methodand solves full system of equations.
    v0 : list, if supplied volume used as initial value to compute fugacities
    full_output: bool, wheter to outputs all calculation info

    
    """
    Z0 = (X0+Y0+W0)/3
    nonzero = np.count_nonzero(Z0)
    x0 = np.vstack([X0,W0,Y0])
    b0 = np.array([0.33,0.33,0.33,0,0])
    
    #check for binary mixture
    if nonzero == 2:
        index = np.nonzero(Z0)[0]
        sol = np.zeros_like(x0)
        sol[:, index], T = haz_pb(x0[:,index],T,P,'P',model,index, 'LLV', v0)
        X, W, Y = sol
        return X, W, Y, T
    
    if not good_initial:
        out = multiflash(x0, b0, ['L','L','V'], Z0, T, P, model, v0, True)
    else:  
        global vx, vw, vy
        sol = fsolve(haz_objt, x0.flatten() ,args = (T, P, model, v0))
        x0 = sol.reshape([model.nc, 3])
        Z0 = x0.mean(axis=0)
        out = multiflash(x0, b0, ['L','L','V'], Z0, T, P, model, [vx,vw,vy], True)
    
    Xm, beta, tetha, equilibrio = out.X, out.beta, out.tetha, out.states
    error_inner =  out.error_inner
    v = out.v
    
    if error_inner > 1e-6:
        order = [2, 0, 1]  #Y, X, W
        Xm = Xm[order]
        betatetha = np.hstack([beta[order], tetha])
        equilibrio = np.asarray(equilibrio)[order]
        v0 = np.asarray(v)[order]
        out = multiflash(Xm, betatetha, equilibrio, Z0, T, P, model, v0 , full_output = True)
        order = [1, 2, 0]
        Xm, beta, tetha, equilibrio = out.X, out.beta, out.tetha, out.states
        error_inner =  out.error_inner
        if error_inner > 1e-6:
            order = [2, 1, 0]  #W, X, Y
            Xm = Xm[order]
            betatetha = np.hstack([beta[order], tetha])
            equilibrio = np.asarray(equilibrio)[order]
            v0 = np.asarray(out.v)[order]
            out = multiflash(Xm, betatetha, equilibrio, Z0, T, P, model, v0 , full_output = True)
            order = [1, 0, 2]
            Xm, beta, tetha, equilibrio = out.X, out.beta, out.tetha, out.states
            error_inner =  out.error_inner
        Xm = Xm[order]
        beta = beta[order]
        tetha = np.hstack([0., tetha])
        tetha = tetha[order]
        v = (out.v)[order]
    else:
        tetha = np.hstack([0., tetha])


    if full_output:
        info = {'T' : T, 'P': P, 'error_outer':out.error_outer, 'error_inner': error_inner, 
                'iter': out.iter, 'beta': beta, 'tetha': tetha,                
               'X' : Xm, 'v':v, 'states' : ['L','L','V']}
        out = EquilibriumResult(info)
        return out
        
    tethainestable = tetha > 0.
    Xm[tethainestable] = None
    X, W, Y = Xm

    return  X, W, Y

__all__ = ['haz']