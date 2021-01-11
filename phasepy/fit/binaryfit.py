from __future__ import division, print_function, absolute_import
import numpy as np
from ..actmodels import virialgamma
from .fitmulticomponent import fobj_elv, fobj_ell, fobj_hazb
from scipy.optimize import minimize, minimize_scalar


def fobj_wilson(inc, mix, datavle, virialmodel='Tsonopoulos'):

    a12, a21 = inc
    a = np.array([[0, a12], [a21, 0]])

    mix.wilson(a)
    model = virialgamma(mix, virialmodel=virialmodel, actmodel='wilson')

    elv = fobj_elv(model, *datavle)

    return elv


def fit_wilson(x0, mix, datavle, virialmodel='Tsonopoulos',
               minimize_options={}):
    """
    fit_wilson: attemps to fit wilson parameters to LVE

    Parameters
    ----------
    x0 : array_like
        initial values a12, a21 in K
    mix: object
        binary mixture
    datavle: tuple
        (Xexp, Yelv, Texp, Pexp)
    virialmodel : function
        function to compute virial coefficients, available options are
        'Tsonopoulos', 'Abbott' or 'ideal_gas'
    minimize_options: dict
        Dictionary of any additional spefication for scipy minimize

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    fit = minimize(fobj_wilson, x0, args=(mix, datavle, virialmodel),
                   **minimize_options)
    return fit


def fobj_nrtl(inc, mix, datavle=None, datalle=None, datavlle=None,
              alpha_fixed=False, alpha0=0.2, Tdep=False,
              virialmodel='Tsonopoulos'):

    if alpha_fixed:
        alpha = alpha0
        if Tdep:
            g12, g21, g12T, g21T = inc
            gT = np.array([[0, g12T], [g21T, 0]])
        else:
            g12, g21 = inc
            gT = None
    else:
        if Tdep:
            g12, g21, g12T, g21T, alpha = inc
            gT = np.array([[0, g12T], [g21T, 0]])
        else:
            g12, g21, alpha = inc
            gT = None
    Alpha = np.array([[0., alpha], [alpha, 0.]])
    g = np.array([[0, g12], [g21, 0]])
    mix.NRTL(Alpha, g, gT)
    model = virialgamma(mix, virialmodel=virialmodel, actmodel='nrtl')

    error = 0.

    if datavle is not None:
        error += fobj_elv(model, *datavle)
    if datalle is not None:
        error += fobj_ell(model, *datalle)
    if datavlle is not None:
        error += fobj_hazb(model, *datavlle)
    return error


def fit_nrtl(x0, mix, datavle=None, datalle=None, datavlle=None,
             alpha_fixed=False, alpha0=0.2, Tdep=False,
             virialmodel='Tsonopoulos', minimize_options={}):
    """
    fit_nrtl: attemps to fit nrtl parameters to LVE, LLE, LLVE

    Parameters
    ----------
    x0 : array_like
        initial values interaction parameters (and aleatory factor)
    mix: object
        binary mixture
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)
    alpha_fit: bool, optional
        if True fix aleatory factor to the value of alpha0
    alpha0: float
        value of aleatory factor if fixed
    Tdep: bool, optional
        Wheter the energy parameters have a temperature dependence
    virialmodel : function
        function to compute virial coefficients, available options are
        'Tsonopoulos', 'Abbott' or 'ideal_gas'
    minimize_options: dict
        Dictionary of any additional spefication for scipy minimize

    Notes
    -----

    if Tdep True parameters are treated as:
            a12 = a12_1 + a12T * T
            a21 = a21_1 + a21T * T

    if alpha_fixed False and Tdep True:
        x0 = [a12, a21, a12T, a21T, alpha]
    if alpha_fixed False and Tdep False:
        x0 = [a12, a21, alpha]
    if alpha_fixed True and Tdep False:
        x0 = [a12, a21]

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize
    """
    fit = minimize(fobj_nrtl, x0, args=(mix, datavle, datalle, datavlle,
                   alpha_fixed, alpha0, Tdep, virialmodel), **minimize_options)
    return fit


def fobj_kij(kij, eos, mix, datavle=None, datalle=None, datavlle=None):

    Kij = np.array([[0, kij], [kij, 0]])
    mix.kij_cubic(Kij)
    cubica = eos(mix)

    error = 0.
    if datavle is not None:
        error += fobj_elv(cubica, *datavle)
    if datalle is not None:
        error += fobj_ell(cubica, *datalle)
    if datavlle is not None:
        error += fobj_hazb(cubica, *datavlle)
    return error


def fit_kij(kij_bounds, eos, mix, datavle=None, datalle=None, datavlle=None):
    """
    fit_kij: attemps to fit kij to LVE, LLE, LLVE

    Parameters
    ----------
    kij_bounds : tuple
        bounds for kij correction
    eos : function
        cubic eos to fit kij for qmr mixrule
    mix: object
        binary mixture
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize

    """
    fit = minimize_scalar(fobj_kij, kij_bounds,
                          args=(eos, mix, datavle, datalle, datavlle))
    return fit


def fobj_rk(inc, mix, datavle=None, datalle=None, datavlle=None,
            Tdep=False, virialmodel='Tsonopoulos'):

    if Tdep:
        c, c1 = np.split(inc, 2)
    else:
        c = inc
        c1 = np.zeros_like(c)
    mix.rk(c, c1)
    modelo = virialgamma(mix, virialmodel=virialmodel, actmodel='rk')

    error = 0.

    if datavle is not None:
        error += fobj_elv(modelo, *datavle)
    if datalle is not None:
        error += fobj_ell(modelo, *datalle)
    if datavlle is not None:
        error += fobj_hazb(modelo, *datavlle)
    return error


def fit_rk(inc0, mix, datavle=None, datalle=None,
           datavlle=None, Tdep=False,
           virialmodel='Tsonopoulos', minimize_options={}):
    """
    fit_rk: attemps to fit RK parameters to LVE, LLE, LLVE

    Parameters
    ----------
    inc0 : array_like
        initial values to RK parameters
    mix: object
        binary mixture
    datavle: tuple, optional
        (Xexp, Yexp, Texp, Pexp)
    datalle: tuple, optional
        (Xexp, Wexp, Texp, Pexp)
    datavlle: tuple, optional
        (Xexp, Wexp, Yexp, Texp, Pexp)
    Tdep : bool,
        wheter the parameter will have a temperature dependence
    virialmodel : function
        function to compute virial coefficients, available options are
        'Tsonopoulos', 'Abbott' or 'ideal_gas'
    minimize_options: dict
        Dictionary of any additional spefication for scipy minimize

    Notes
    -----

    if Tdep true:
        C = C' + C'T
    if Tdep true:
        inc0 = [C'0, C'1, C'2, ..., C'0T, C'1T, C'2T... ]
    if Tdep flase:
        inc0 = [C0, C1, C2...]

    Returns
    -------
    fit : OptimizeResult
        Result of SciPy minimize
    """
    fit = minimize(fobj_rk, inc0, args=(mix, datavle, datalle, datavlle,
                   Tdep, virialmodel), **minimize_options)
    return fit
