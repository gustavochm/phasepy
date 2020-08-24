from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from scipy.integrate import cumtrapz
from .cijmix_cy import cmix_cy
from .tensionresult import TensionResult


def fobj_sk(inc, spath, temp_aux, mu0, ci, sqrtci, model):
    ro = np.abs(inc[:-1])
    alpha = inc[-1]
    mu = model.muad_aux(ro, temp_aux)
    obj = np.zeros_like(inc)
    obj[:-1] = mu - (mu0 + alpha*sqrtci)
    obj[-1] = spath - sqrtci.dot(ro)
    return obj


def ten_beta0_sk(rho1, rho2, Tsat, Psat, model, n=200, full_output=False,
                 alpha0=None):

    nc = model.nc

    # Dimensionless variables
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = rho1*rofactor
    ro2a = rho2*rofactor

    cij = model.ci(Tsat)
    cij /= cij[0, 0]
    ci = np.diag(cij)
    sqrtci = np.sqrt(ci)

    temp_aux = model.temperature_aux(Tsat)

    mu0 = model.muad_aux(ro1a, temp_aux)
    mu02 = model.muad_aux(ro2a, temp_aux)
    if not np.allclose(mu0, mu02):
        raise Exception('Not equilibria compositions, mu1 != mu2')

    # Path function
    s0 = sqrtci.dot(ro1a)
    s1 = sqrtci.dot(ro2a)
    spath = np.linspace(s0, s1, n)
    ro = np.zeros([nc, n])
    alphas = np.zeros(n)
    ro[:, 0] = ro1a
    ro[:, -1] = ro2a

    deltaro = ro2a - ro1a
    i = 1
    r0 = ro1a + deltaro * (spath[i]-s0) / n
    if alpha0 is None:
        alpha0 = np.mean((model.muad_aux(r0, temp_aux) - mu0)/sqrtci)
    r0 = np.hstack([r0, alpha0])
    ro0 = root(fobj_sk, r0, args=(spath[i], temp_aux, mu0, ci, sqrtci, model),
               method='lm')
    ro0 = ro0.x
    ro[:, i] = np.abs(ro0[:-1])

    for i in range(1, n):
        sol = root(fobj_sk, ro0, args=(spath[i], temp_aux, mu0, ci,
                   sqrtci, model), method='lm')
        ro0 = sol.x
        alphas[i] = ro0[-1]
        ro[:, i] = ro0[:-1]

    # Derivatives respect to path function
    drods = np.gradient(ro, spath, edge_order=2, axis=1)

    suma = cmix_cy(drods, cij)
    dom = np.zeros(n)
    for k in range(1, n - 1):
        dom[k] = model.dOm_aux(ro[:, k], temp_aux, mu0, Pad)

    integral = np.nan_to_num(np.sqrt(2*dom*suma))
    tension = np.abs(np.trapz(integral, spath))
    tension *= tenfactor

    if full_output:
        # Zprofile
        with np.errstate(divide='ignore'):
            intz = (np.sqrt(suma/(2*dom)))
        intz[np.isinf(intz)] = 0
        z = np.abs(cumtrapz(intz, spath, initial=0))
        z /= zfactor
        ro /= rofactor
        dictresult = {'tension': tension, 'rho': ro, 'z': z,
                      'GPT': dom, 'path': spath, 'alphas': alphas}
        out = TensionResult(dictresult)
        return out

    return tension
