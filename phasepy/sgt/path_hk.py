from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import cumtrapz
from .cijmix_cy import cmix_cy
from .tensionresult import TensionResult


def fobj_beta0(dro, ro1, dh2, s, temp_aux, mu0, ci, sqrtci, model):
    ro = ro1 + dro
    dmu = model.muad_aux(ro, temp_aux) - mu0

    f1 = sqrtci[s]*dmu
    f2 = sqrtci*dmu[s]
    obj = f1 - f2
    obj[s] = dh2 - ci.dot(dro**2)
    return obj


def ten_beta0_hk(rho1, rho2, Tsat, Psat, model, n=1000, full_output=False):

    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = rho1*rofactor
    ro2a = rho2*rofactor

    nc = model.nc

    temp_aux = model.temperature_aux(Tsat)

    mu0 = model.muad_aux(ro1a, temp_aux)
    mu02 = model.muad_aux(ro2a, temp_aux)
    if not np.allclose(mu0, mu02):
        raise Exception('Not equilibria compositions, mu1 != mu2')

    cij = model.ci(Tsat)
    c1 = cij[0, 0]
    cij /= c1
    ci = np.diag(cij)
    sqrtci = np.sqrt(ci)

    Nsteps = n
    deltaro = ro2a-ro1a
    Hl0 = ci.dot(deltaro)
    dH = Hl0/(Nsteps + 1)
    dH2 = dH**2

    dro = [np.zeros(nc)]
    ro = [ro1a]

    i = 1
    dr0 = deltaro*dH2
    dro0 = fsolve(fobj_beta0, dr0, args=(ro[i-1], dH2, 0, temp_aux, mu0, ci,
                  sqrtci, model))
    ro0 = np.add(ro[i-1], dro0)
    ro.append(ro0)
    dro.append(dro0)
    end = False

    while not end and i < 2*Nsteps:
        i += 1
        dro0 = fsolve(fobj_beta0, dro[i-1], args=(ro[i-1], dH2, 0,
                      temp_aux, mu0, ci, sqrtci, model))
        ro0 = ro[i-1] + dro0
        ro.append(ro0)
        dro.append(dro0)
        end = np.allclose(ro0, ro2a, rtol=1e-2)

    ro.append(ro2a)
    dro.append(ro2a-ro[-2])
    dro2 = np.asarray(dro).T

    # Path Error
    Hl = np.sqrt(ci.dot(dro2**2)).sum()
    dH = Hl/(Nsteps + 1)
    dH2 = dH**2
    error = np.abs(Hl - Hl0)
    it = 0

    while error > 1e-3 and it < 5:
        it += 1
        Hl0 = Hl
        dro = [np.zeros(nc)]
        ro = [ro1a]
        i = 1
        dr0 = deltaro*dH2
        dro0 = fsolve(fobj_beta0, dr0, args=(ro[i-1], dH2, 0, temp_aux,
                      mu0, ci, sqrtci, model))
        ro0 = np.add(ro[i-1], dro0)
        ro.append(ro0)
        dro.append(dro0)
        end = np.allclose(ro[i], ro2a, rtol=1e-2)

        while i < Nsteps:
            i += 1
            dro0 = fsolve(fobj_beta0, dro[i-1], args=(ro[i-1], dH2, 0,
                          temp_aux, mu0, ci, sqrtci, model))
            ro0 = np.add(ro[i-1], dro0)
            ro.append(ro0)
            dro.append(dro0)

        ro.append(ro2a)
        dro.append(ro2a-ro[-2])
        dro2 = np.asarray(dro).T

        Hl = np.sqrt(ci.dot(dro2**2)).sum()
        dH = Hl/(Nsteps + 1)
        dH2 = dH**2
        error = np.abs(Hl - Hl0)

    ro2 = np.asarray(ro).T
    Hi = dH * np.arange(0, Nsteps + 2)
    drodh = np.gradient(ro2, Hi, edge_order=2, axis=1)

    suma = cmix_cy(drodh, cij)
    dom = np.zeros(Nsteps + 2)
    for k in range(1, Nsteps + 1):
        dom[k] = model.dOm_aux(ro2[:, k], temp_aux, mu0, Pad)

    # Tension computation
    integral = np.nan_to_num(np.sqrt(2*dom*suma))
    ten = np.abs(np.trapz(integral, Hi))
    ten *= tenfactor

    if full_output:
        # Z profile
        with np.errstate(divide='ignore'):
            intz = (np.sqrt(suma/(2*dom)))
        intz[np.isinf(intz)] = 0
        z = np.abs(cumtrapz(intz, Hi, initial=0))
        z /= zfactor
        ro2 /= rofactor
        dictresult = {'tension': ten, 'rho': ro2, 'z': z,
                      'GPT': dom, 'path': Hi}
        out = TensionResult(dictresult)
        return out
    return ten
