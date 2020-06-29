from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import root
from ..math import lobatto
from .cijmix_cy import cmix_cy
from .tensionresult import TensionResult


def fobj_saddle(ros, mu0, T, eos):
    mu = eos.muad(ros, T)
    return mu - mu0


def sgt_linear(rho1, rho2, Tsat, Psat, model, n=100, full_output=False):
    """
    SGT linear for mixtures (rho1, rho2, T, P) -> interfacial tension

    Parameters
    ----------
    rho1 : float
        phase 1 density vector
    rho2 : float
        phase 2 density vector
    Tsat : float
        saturation temperature
    Psat : float
        saturation pressure
    model : object
        created with an EoS
    n : int, optional
        number points to solve density profiles
    full_output : bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    ten : float
        interfacial tension between the phases
    """

    # Dimensionless variables
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = rho1*rofactor
    ro2a = rho2*rofactor

    cij = model.ci(Tsat)
    cij /= cij[0, 0]

    mu0 = model.muad(ro1a, Tsat)
    mu02 = model.muad(ro2a, Tsat)
    if not np.allclose(mu0, mu02):
        raise Exception('Not equilibria compositions, mu1 != mu2')
    roots, weights = lobatto(n)
    s = 0
    ro_s = (ro2a[s]-ro1a[s])*roots + ro1a[s]  # integrations nodes
    wreal = np.abs(weights*(ro2a[s]-ro1a[s]))  # integration weights

    # Linear profile
    pend = (ro2a - ro1a)
    b = ro1a
    ro = (np.outer(roots, pend) + b).T

    # Derivatives respect to component 1
    dro = np.gradient(ro, ro_s, edge_order=2, axis=1)

    suma = cmix_cy(dro, cij)
    dom = np.zeros(n)
    for k in range(1, n - 1):
        dom[k] = model.dOm(ro[:, k], Tsat, mu0, Pad)

    integral = np.nan_to_num(np.sqrt(2*dom*suma))
    tension = np.dot(integral, wreal)
    tension *= tenfactor

    if full_output:
        # Z profile
        with np.errstate(divide='ignore'):
            intz = (np.sqrt(suma/(2*dom)))
        intz[np.isinf(intz)] = 0
        z = np.cumsum(intz*wreal)
        z /= zfactor
        ro /= rofactor
        dictresult = {'tension': tension, 'rho': ro, 'z': z,
                      'GPT': np.hstack([0, dom, 0])}
        out = TensionResult(dictresult)
        return out

    return tension


def sgt_spot(rho1, rho2, Tsat, Psat, model, n=50, full_output=False):
    """
    SGT spot for mixtures (rho1, rho2, T, P) -> interfacial tension

    Parameters
    ----------
    rho1 : float
        phase 1 density vector
    rho2 : float
        phase 2 density vector
    Tsat : float
        saturation temperature
    Psat : float
        saturation pressure
    model : object
        created with an EoS
    n : int, optional
        number points to solve density profiles
    full_output : bool, optional
        wheter to outputs all calculation info

    Returns
    -------
    ten : float
        interfacial tension between the phases
    """

    # adimensionalizar variables
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = rho1*rofactor
    ro2a = rho2*rofactor

    cij = model.ci(Tsat)
    cij /= cij[0, 0]

    mu0 = model.muad(ro1a, Tsat)
    mu02 = model.muad(ro2a, Tsat)
    if not np.allclose(mu0, mu02):
        raise Exception('Not equilibria compositions, mu1 != mu2')

    roots, weights = lobatto(n)
    s = 0
    try:
        ros = (ro1a + ro2a)/2
        ros = root(fobj_saddle, ros, args=(mu0, Tsat, model), method='lm')
        if ros.success:
            ros = ros.x
            # segment1
            # Linear profile
            pend = (ros - ro1a)
            b = ro1a
            ro1 = (np.outer(roots, pend) + b).T

            ro_s1 = (ros[s]-ro1a[s])*roots + ro1a[s]  # integration nodes
            wreal1 = np.abs(weights*(ros[s]-ro1a[s]))  # integration weights

            dro1 = np.gradient(ro1, ro_s1, edge_order=2, axis=1)

            # segment2
            # Linear profile
            pend = (ro2a - ros)
            b = ros
            ro2 = (np.outer(roots, pend) + b).T

            ro_s2 = (ro2a[s]-ro1a[s])*roots + ro1a[s]  # integration nodes
            wreal2 = np.abs(weights*(ro2a[s]-ro1a[s]))  # integration weights

            dro2 = np.gradient(ro2, ro_s2, edge_order=2, axis=1)

            dom1 = np.zeros(n)
            dom2 = np.zeros(n)
            for i in range(n):
                dom1[i] = model.dOm(ro1[:, i], Tsat, mu0, Pad)
                dom2[i] = model.dOm(ro2[:, i], Tsat, mu0, Pad)
            dom1[0] = 0.
            dom2[-1] = 0.

            suma1 = cmix_cy(dro1, cij)
            suma2 = cmix_cy(dro2, cij)

            integral1 = np.nan_to_num(np.sqrt(2*dom1*suma1))
            integral2 = np.nan_to_num(np.sqrt(2*dom2*suma2))
            tension = np.dot(integral1, wreal1)
            tension += np.dot(integral2, wreal2)
            tension *= tenfactor
            out = tension
            if full_output:
                with np.errstate(divide='ignore'):
                    intz1 = np.sqrt(suma1/(2*dom1))
                    intz2 = np.sqrt(suma2/(2*dom2))
                intz1[np.isinf(intz1)] = 0
                intz2[np.isinf(intz2)] = 0
                z1 = np.cumsum(intz1*wreal1)
                z2 = z1[-1] + np.cumsum(intz2*wreal2)
                z = np.hstack([z1, z2])
                ro = np.hstack([ro1, ro2])
                z /= zfactor
                ro /= rofactor
                dictresult = {'tension': tension, 'rho': ro, 'z': z,
                              'GPT': np.hstack([dom1, dom2])}
                out = TensionResult(dictresult)
        else:
            out = sgt_linear(ro1, ro2, Tsat, Psat, model, n, full_output)
    except:
        out = sgt_linear(ro1, ro2, Tsat, Psat, model, n, full_output)

    return out
