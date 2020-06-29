from __future__ import division, print_function, absolute_import
import numpy as np


def alpha_vdw():
    return 1.


# Redlich Kwong's alphas function
def alpha_rk(T, Tc):
    return np.sqrt(T/Tc)**-0.5


# Soaves's alpha function
def alpha_soave(T, k, Tc):
    return (1+k*(1-np.sqrt(T/Tc)))**2


# SV's alphas function
def alpha_sv(T, ksv, Tc):
    ksv = ksv.T
    k0 = ksv[0]
    k1 = ksv[1]
    Tr = T/Tc
    sTr = np.sqrt(Tr)
    return (1+(k0+k1*(0.7-Tr)*(1+sTr))*(1-sTr))**2
