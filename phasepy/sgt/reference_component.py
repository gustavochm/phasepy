from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import fsolve
from ..math import lobatto, colocA
from .cijmix_cy import cmix_cy
from .tensionresult import TensionResult


def fobj_beta0(ro, ro_s, s, temp_aux, mu0, sqrtci, model):
    nc = model.nc
    ro = np.insert(ro, s, ro_s)
    dmu = model.muad_aux(ro, temp_aux) - mu0

    f1 = sqrtci[s]*dmu
    f2 = sqrtci*dmu[s]

    return (f1-f2)[np.arange(nc) != s]


def ten_beta0_reference(rho1, rho2, Tsat, Psat, model, s=0,
                        n=100, full_output=False):

    nc = model.nc

    # Dimensionless profile
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
    # roots and weights for Lobatto quadrature
    roots, weights = lobatto(n)

    ro_s = (ro2a[s]-ro1a[s])*roots+ro1a[s]  # Integration nodes
    wreal = np.abs(weights*(ro2a[s]-ro1a[s]))  # Integration weights

    # A matrix for derivatives with orthogonal collocation
    A = colocA(roots) / (ro2a[s]-ro1a[s])

    rodep = np.zeros([nc-1, n])
    rodep[:, 0] = ro1a[np.arange(nc) != s]
    for i in range(1, n):
        rodep[:, i] = fsolve(fobj_beta0, rodep[:, i-1],
                             args=(ro_s[i], s, temp_aux, mu0, sqrtci, model))
    ro = np.insert(rodep, s, ro_s, axis=0)
    dro = rodep@A.T
    dro = np.insert(dro, s, np.ones(n), axis=0)

    suma = cmix_cy(dro, cij)

    dom = np.zeros(n)
    for k in range(1, n-1):
        dom[k] = model.dOm_aux(ro[:, k], temp_aux, mu0, Pad)

    intten = np.nan_to_num(np.sqrt(suma*(2*dom)))
    ten = np.dot(intten, wreal)
    ten *= tenfactor

    if full_output:
        # Z profile
        with np.errstate(divide='ignore'):
            intz = (np.sqrt(suma/(2*dom)))
        intz[np.isinf(intz)] = 0
        z = np.cumsum(intz*wreal)
        z /= zfactor
        ro /= rofactor
        dictresult = {'tension': ten, 'rho': ro, 'z': z,
                      'GPT': dom}
        out = TensionResult(dictresult)
        return out

    return ten
