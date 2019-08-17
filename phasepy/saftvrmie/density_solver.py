import numpy as np
from scipy.optimize import minimize_scalar, brentq
kb = 1.3806488e-23 # K/J
Na = 6.02214e23 

def dPsaft_fun(rho, T, saft):
    _, dafcn, d2afcn = saft.d2afcn_drho(rho, T) 
    dPsaft = 2 * rho * dafcn + rho**2 * d2afcn 
    dPsaft /= Na
    return dPsaft

def Psaft_obj(rho, T, saft, Pspec):
    _, dafcn,  = saft.dafcn_drho(rho, T) 
    Psaft = rho**2 * dafcn / Na
    return Psaft - Pspec

def density_topliss(state, T, P, saft):
    
    #lower boundary a zero density
    rho_lb = 1e-5
    P_lb = 0.
    dP_lb = Na * kb * T
    
    #upper boundary limit at infinity pressure
    etamax = 0.7405
    rho_lim = (6 * etamax) / (saft.ms * np.pi * saft.sigma**3)
    #Mejorar esta parte con el maximo valor posible de eta de la densidad
    ub_sucess = False
    rho_ub = 0.4 *  rho_lim
    it = 0
    while not ub_sucess and it < 5:
        it += 1
        P_ub, dP_ub = saft.dP_drho(rho_ub, T)
        rho_ub += 0.1 *  rho_lim
        ub_sucess = P_ub > P and dP_ub > 0
        
    #Calculo de derivada numerica en densidad nula
    rho_lb1 = 1e-23 * rho_lim
    P_lb1, dP_lb1 = saft.dP_drho(rho_lb1, T)
    d2P_lb1 = (dP_lb1 - dP_lb) / rho_lb1
    if d2P_lb1 > 0:
        flag = 3
    else: 
        flag = 1
        
    #Comienza Etapa 1    
    bracket = [rho_lb, rho_ub]
    if flag == 1:
        #Se debe encontrar el punto de inflexion
        sol_inf = minimize_scalar(dPsaft_fun, args = (T, saft),
                                  bounds = bracket, method = 'Bounded' )
        rho_inf = sol_inf.x
        dP_inf = sol_inf.fun
        if dP_inf > 0:
            flag = 3
        else: 
            flag = 2
            
    #Etapa 2
    if flag == 2:
        if state == 'L':
            bracket[0] = rho_inf
        elif state == 'V':
            bracket[1] = rho_inf
        rho_ext = brentq(dPsaft_fun, bracket[0], bracket[1], args =(T, saft))
        P_ext, dP_ext = saft.dP_drho(rho_ext, T)
        if P_ext > P and state == 'V':
            bracket[1] = rho_ext
        elif P_ext < P and state == 'L':
            bracket[0] = rho_ext
        else:
            flag = -1

    if flag == -1:
        rho = np.nan
    else:
        rho = brentq(Psaft_obj, bracket[0], bracket[1], args = (T, saft, P))

    return rho

def density_newton(rho0, T, P, saft):
    
    rho = 1.*rho0
    Psaft, dPsaft = saft.dP_drho(rho, T)
    FO = Psaft - P
    dFO = dPsaft
    for i in range(10):
        rho -= FO/dFO
        Psaft, dPsaft = saft.dP_drho(rho, T)
        FO = Psaft - P
        dFO = dPsaft
        if np.abs(FO) < 1e-6:
            break
    return rho
        