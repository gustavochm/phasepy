from __future__ import division, print_function, absolute_import
from scipy.optimize import brentq, newton
from ..constants import R


def fobj_tsat(T, P, cubic):

    a = cubic.a_eos(T)
    b = cubic.b

    A = a*P/(R*T)**2
    B = b*P/(R*T)
    Z = cubic._Zroot(A, B)
    Zl = min(Z)
    Zv = max(Z)
    fugL = cubic._logfug_aux(Zl, A, B)
    fugV = cubic._logfug_aux(Zv, A, B)
    FO = fugV-fugL

    return FO


def tsat(cubic, P, T0=None, Tbounds=None):
    """
    Computes saturation temperature with cubic eos

    Parameters
    ----------
    cubic: object
        cubic eos object
    P: float
        saturation pressure [bar]
    T0 : float, optional
         Temperature to start iterations [K]
    Tbounds : tuple, optional
            (Tmin, Tmax) Temperature interval to start iterations [K]

    Returns
    -------
    T : float
        saturation temperature [K]
    vl: float
        saturation liquid volume [cm3/mol]
    vv: float
        saturation vapor volume [cm3/mol]

    """
    bool1 = T0 is None
    bool2 = Tbounds is None

    if bool1 and bool2:
        raise Exception('You must provide either Tbounds or T0')

    if not bool1:
        sol = newton(fobj_tsat, x0=T0, args=(P, cubic),
                     full_output=False)
        Tsat = sol[0]
    elif not bool2:
        sol = brentq(fobj_tsat, Tbounds[0], Tbounds[1], args=(P, cubic),
                     full_output=False)
        Tsat = sol

    vl = 1./cubic.density(Tsat, P, 'L')
    vv = 1./cubic.density(Tsat, P, 'V')
    out = (Tsat, vl, vv)
    return out
