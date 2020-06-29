from __future__ import division, print_function, absolute_import
from scipy.optimize import minimize
import numpy as np


def fobj_c(c, eos, Texp, Pexp, rhoexp):
    eos.c = c
    n = len(Texp)
    rho = np.zeros_like(rhoexp)
    for i in range(n):
        rho[i] = eos.density(Texp[i], Pexp[i], 'L')
    return np.sum((rho/rhoexp - 1)**2)


def fit_vt(component, eos, Texp, Pexp, rhoexp, c0=0.):
    """
    fit Volume Translation for cubic EoS

    Parameters
    ----------
    component : object
        created with component class
    eos:  function
        cubic eos function
    Texp : array_like
        experimental temperature in K.
    Pexp : array_like
        experimental pressure in bar.
    rhoexp : array_like
        experimental liquid density at given temperature and
        pressure in mol/cm3.
    c0 : float, optional
        initial values.

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    cubic = eos(component, volume_translation=True)
    fit = minimize(fobj_c, c0, args=(cubic, Texp, Pexp, rhoexp))
    return fit
