from __future__ import division, print_function, absolute_import
import numpy as np
from ..sgt import ten_fit
from ..math import gauss


def fit_cii(tenexp, Texp, model, order=2, n=100, P0=None):
    """
    fit influence parameters of SGT

    Parameters
    ----------
    tenexp: array_like
        experimental surface tension in mN/m
    Texp: array_like
        experimental temperature in K.
    model: object
        created from eos and component
    order: int, optional
        order of cii polynomial
    n : int, optional
        number of integration points in SGT
    P0 : array_like , optional
        initial guess for saturation pressure

    Returns
    -------
    cii : array_like
        polynomial coefficients of influence parameters of SGT
    """
    roots, weigths = gauss(n)
    tena = np.zeros_like(tenexp)
    neq = len(Texp)
    if P0 is None:
        psat0 = neq * [None]
    else:
        if len(P0) != neq:
            raise Exception('P0 lenght must be the same as Texp')
        else:
            psat0 = P0

    for i in range(len(Texp)):
        tena[i] = ten_fit(Texp[i], model, roots, weigths, psat0[i])

    cii = (tenexp/tena)**2

    return np.polyfit(Texp, cii, order)
