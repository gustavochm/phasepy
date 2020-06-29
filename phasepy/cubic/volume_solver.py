from __future__ import division, print_function, absolute_import
import numpy as np


# Auxiliar function that computes objective function and its derivatives
def _volume_newton_aux(V, B, D_RT, P_RT, c1, c2):

    Vc1B = V + c1*B
    Vc2B = V + c2*B
    V_B = V - B

    f0 = P_RT - (1./V_B - D_RT / (Vc1B * Vc2B))

    df0 = 1/V_B**2 - D_RT * (1./(Vc1B * Vc2B**2) + 1./(Vc1B**2 * Vc2B))

    return f0, df0


# Functions that solves volume using Newtons's Method
def volume_newton(v0, P_RT, D_RT, B, c1, c2):
    V = 1. * v0
    f0, df0 = _volume_newton_aux(V, B, D_RT, P_RT, c1, c2)
    error = np.abs(f0)
    it = 0
    while error > 1e-8 and it < 20:
        it += 1
        dV = - f0 / df0
        V += dV
        error = np.abs(f0)
        f0, df0 = _volume_newton_aux(V, B, D_RT, P_RT, c1, c2)
    return V
