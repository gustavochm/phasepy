import numpy as np
from ..equilibrium import bubblePy, ell, tpd_min, hazb, hazt

def fobj_elv(model, Xexp, Yexp, Texp, Pexp):
    """ 
    Objective function to fit parameters for ELV in multicomponent mixtures
    """
    P = np.zeros_like(Pexp) #n,
    Y = np.zeros_like(Yexp) #nc,n
        
    n = len(Pexp)
    for i in range(n):
        Y[:,i], P[i] = bubblePy(Yexp[:,i], Pexp[i], Xexp[:,i], Texp[i], model)
    
    error = ((Y-Yexp)**2).sum()
    error += ((P/Pexp-1)**2).sum()
    error /= n
    return error


def fobj_ell(model,Xexp,Wexp,Texp,Pexp):
    """ 
    Objective function to fit parameters for ELL in multicomponent mixtures
    """
    X=np.zeros_like(Xexp)
    W=np.zeros_like(Wexp)

    n = len(Texp)
    Z = (Xexp+Wexp)/2
    for i in range(n):
        X0, tpd = tpd_min(Xexp[:,i],Z[:,i],Texp[i],Pexp[i],model, 'L', 'L')
        W0, tpd = tpd_min(Wexp[:,i],Z[:,i],Texp[i],Pexp[i],model, 'L', 'L')
        X[:,i], W[:,i], beta = ell(X0 ,W0 , Z[:,i], Texp[i], Pexp[i], model)
    
    error = ((X-Xexp)**2).sum()
    error += ((W-Wexp)**2).sum()
    error /= n
    return error

def fobj_hazb(model, Xellv, Wellv, Yellv, Tellv, Pellv,  info = [1,1,1]):
    """ 
    Objective function to fit parameters for ELLV in binary mixtures
    """
    n = len(Tellv)
    X = np.zeros_like(Xellv)
    W = np.zeros_like(Wellv)
    Y = np.zeros_like(Yellv)
    P = np.zeros_like(Pellv)
    Zll = (Xellv + Wellv) / 2
    
    for i in range(n):
        try:
            X0, tpd = tpd_min(Xellv[:,i],Zll[:,i],Tellv[i],Pellv[i],model, 'L', 'L')
            W0, tpd = tpd_min(Wellv[:,i],Zll[:,i],Tellv[i],Pellv[i],model, 'L', 'L')
            X[:,i], W[:,i], Y[:,i] ,P[i] = hazb(X0, W0, Yellv[:,i], 
             Pellv[i], Tellv[i], 'T', model)
        except: 
            pass
            

    error = info[0]*((X-Xellv)**2).sum()
    error += info[1]*((W-Wellv)**2).sum()
    error += info[2]*((Y - Yellv)**2).sum()
    error += ((P/Pellv- 1)**2).sum()
    error /= n
    
    return  error

def fobj_hazt(model, Xellv, Wellv, Yellv, Tellv, Pellv):
    """ 
    Objective function to fit parameters for ELLV in multicomponent mixtures
    """

    n = len(Tellv)
    X = np.zeros_like(Xellv)
    W = np.zeros_like(Wellv)
    Y = np.zeros_like(Yellv)
    
    error = 0
    for i in range(n):
        try:
            X[:,i], W[:,i], Y[:,i] = hazt(Xellv[:,i], Wellv[:,i], Yellv[:,i],
                                          Tellv[i], Pellv[i], model, True)
        except ValueError: 
            X[:,i], W[:,i], Y[:,i], T = hazt(Xellv[:,i], Wellv[:,i], Yellv[:,i],
                                    Tellv[i], Pellv[i], model, True)
            error += (T/Tellv[i]-1)**2
        except:
            pass

    
    error += ((np.nan_to_num(X) - Xellv)**2).sum()
    error += ((np.nan_to_num(Y) - Yellv)**2).sum()
    error += ((np.nan_to_num(W) - Wellv)**2).sum()
    error /= n
    return error
