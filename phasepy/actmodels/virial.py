from __future__ import division, print_function, absolute_import
import numpy as np
from ..constants import R


def Virialmix(mix):
    '''
    VirialMix creates the needed arrays to work with multicomponent
    virial eos

    Parameters
    ----------
    mix: object
        Created from two or more components

    Returns
    -------
    Tij: array
        Square matrix of critical temperatures
    Pij: array
        Square matrix of critical pressures
    Zij: array
        Square matrix of critical compressibility factor
    wij: array
        Square matrix of acentric factor

    '''
    Tc = np.asarray(mix.Tc)
    Pc = np.asarray(mix.Pc)
    Zc = np.asarray(mix.Zc)
    w = np.asarray(mix.w)
    Vc = np.asarray(mix.Vc)

    Vc3 = Vc**(1/3)
    vij = (np.add.outer(Vc3, Vc3)/2)**3
    kij = 1 - np.sqrt(np.outer(Vc, Vc))/vij
    Tij = np.sqrt(np.outer(Tc, Tc))*(1-kij)
    wij = np.add.outer(w, w)/2
    Zij = np.add.outer(Zc, Zc)/2
    Pij = Zij*R*Tij/vij
    np.fill_diagonal(Pij, Pc)

    return Tij, Pij, Zij, wij


def Tsonopoulos(T, Tij, Pij, wij):
    r'''
    Returns array of virial coefficient for a mixture at given
    temperature with Tsonopoulos correlation for the first virial
    coefficient, `B`:

    .. math::
	\frac{BP_c}{RT_c} = B^{(0)} + \omega B^{(1)}

    Where :math:`B^{(0)}` and :math:`B^{(1)}` are obtained from:

    .. math::
	B^{(0)} &= 0.1445 - \frac{0.33}{T_r} - \frac{0.1385}{T_r^2} - \frac{0.0121}{T_r^3} - \frac{0.000607}{T_r^8} \\
	B^{(1)} &= 0.0637 + \frac{0.331}{T_r^2} - \frac{0.423}{T_r^3} - \frac{0.008}{T_r^8}
    '''
    Tr = T/Tij
    B0 = 0.1145-0.330/Tr-0.1385/Tr**2-0.0121/Tr**3-0.000607/Tr**8
    B1 = 0.0637+0.331/Tr**2-0.423/Tr**3-0.008/Tr**8
    Bij = (B0+wij*B1)*R*Tij/Pij
    return Bij


def Abbott(T, Tij, Pij, wij):
    r'''
    Returns array of virial coefficients for a mixture at given
    temperature with Abbott-Van Ness correlation for the first virial
    coefficient, `B`:

    .. math::
        \frac{BP_c}{RT_c} = B^{(0)} + \omega B^{(1)}

    Where :math:`B^{(0)}` and :math:`B^{(1)}` are obtained from:

    .. math::
	B^{(0)} &= 0.083 - \frac{0.422}{T_r^{1.6}}\\
	B^{(1)} &= 0.139 + \frac{0.179}{T_r^{4.2}}
    '''
    Tr = T/Tij
    B0 = 0.083-0.422/Tr**1.6
    B1 = 0.139-0.172/Tr**4.2
    Bij = (B0+wij*B1)*R*Tij/Pij
    return Bij


def ideal_gas(T, Tij, Pij, wij):
    r'''
    Returns array of ideal virial coefficients (zeros). The model equation is

    .. math::
        Z = \frac{Pv}{RT} = 1

    Note: Ideal gas model uses only the shape of Tij to produce zeros.
    '''
    Bij = np.zeros_like(Tij)
    return Bij


def virial(x, T, Tij, Pij, wij, virialmodel):
    '''
    Computes the virial coefficient and partial virial coefficient for a
    mixture at given temperature and composition.

    Parameters
    ----------
    x: array
        fraction mole array
    T : float
        absolute temperature in K
    Tij: array
        Square matrix of critical temperatures
    Pij: array
        Square matrix of critical pressures
    wij: array
        Square matrix of acentric
    virialmodel : function
        Function that computes the virial coefficient.

    Returns
    -------
    Bi: array
        Array of virial coefficient of pure component
    Bp : array
        Array of partial virial coefficients
    '''

    Bij = virialmodel(T, Tij, Pij, wij)

    Bx = Bij*x
    # Mixture Virial
    Bm = np.sum(Bx.T*x)
    # Molar partial virial
    Bp = 2*np.sum(Bx, axis=1) - Bm

    return Bij, Bp
