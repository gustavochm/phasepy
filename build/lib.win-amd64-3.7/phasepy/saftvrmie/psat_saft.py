import numpy as np
from scipy.optimize import root

kb = 1.3806488e-23 # K/J
Na = 6.02214e23 
R = Na * kb


def mu_obj(rho, T, saft):
    rhol, rhov = Na * rho
    
    afcnl, dafcnl, d2afcnl = saft.d2afcn_drho(rhol, T)
    Pl =  rhol**2 * dafcnl / Na
    dPl = (2 * rhol * dafcnl + rhol**2 * d2afcnl)
    
    mul = afcnl + rhol*dafcnl
    dmul = Na * (rhol*d2afcnl + 2*dafcnl)
    
    afcnv, dafcnv, d2afcnv = saft.d2afcn_drho(rhov, T)
    Pv =  rhov**2 * dafcnv / Na
    dPv = (2 * rhov * dafcnv + rhov**2 * d2afcnv)
    
    muv = afcnv + rhov * dafcnv
    dmuv = Na * (rhol*d2afcnv + 2*dafcnv)

    
    FO = np.array([mul-muv, Pl - Pv])
    dFO = np.array([[dmul, -dmuv],
                   [dPl, - dPv]])
    return FO, dFO


def psat(saft, T, P0 = None, v0 = [None, None]):
    
    P0input = P0 is None
    v0input = v0 == [None, None]
    
    if P0input and v0input:
        print('You need to provide either initial pressure or volumes')
    elif not P0input:
        good_initial = False
        P = P0
    elif not v0input:
        good_initial = True

    vl, vv = v0
    if not good_initial:
        lnphiv, vv = saft.logfug(T, P, 'V', vv)
        lnphil, vl = saft.logfug(T, P, 'L', vl)
        FO = lnphiv - lnphil
        dFO = (vv - vl)/ R / T
        P -= FO/dFO
        for i in range(10):
            lnphiv, vv = saft.logfug(T, P, 'V', vv)
            lnphil, vl = saft.logfug(T, P, 'L', vl)
            FO = lnphiv - lnphil
            dFO = (vv - vl)/ R / T
            P -= FO/dFO
            sucess = abs(FO) <= 1e-6
            if sucess: break
        if not sucess:
            rho0 = 1. / np.array([vl, vv])
            sol = root(mu_obj, rho0, args = (T, saft), jac = True)
            rhol, rhov = sol.x
            vl, vv = 1./sol.x
            rhomolecular = rhol * Na
            afcn, dafcn = saft.dafcn_drho(rhomolecular, T)
            P =  rhomolecular**2 * dafcn/ Na
    else: 
        rho0 = 1. / np.asarray([v0])
        sol = root(mu_obj, rho0, args = (T, saft), jac = True)
        if sol.success:
            rhol, rhov = sol.x
            vl, vv = 1./sol.x
            rhomolecular = rhol * Na
            afcn, dafcn = saft.dafcn_drho(rhomolecular, T)
            P =  rhomolecular**2 * dafcn/ Na
        else:
            P = None
    return P, vl, vv