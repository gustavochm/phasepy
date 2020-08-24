from __future__ import division, print_function, absolute_import
import numpy as np
from .path_sk import ten_beta0_sk
from .path_hk import ten_beta0_hk
from .reference_component import ten_beta0_reference


def sgt_mix_beta0(rho1, rho2, Tsat, Psat, model, n=100, method='reference',
                  full_output=False, s=0, alpha0=None):
    """
    SGT for mixtures and beta = 0 (rho1, rho2, T, P) -> interfacial tension

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
    method : string
        method used to solve density profiles, available options are
        'reference' for using reference component method, 'cornelisse' for
        Cornelisse path function and 'liang' for Liang path function
    full_output : bool, optional
        wheter to outputs all calculation info
    s : int
        index of reference component used in refernce component method
    alpha0 : float
        initial guess for solving Liang path function


    Returns
    -------
    ten : float
        interfacial tension between the phases
    """

    cij = model.ci(Tsat)
    cij /= cij[0, 0]
    dcij = np.linalg.det(cij)
    if not np.isclose(dcij, 0):
        warning = 'Influece parameter matrix is not singular probably a'
        warning += ' beta has been set up.'
        raise Exception(warning)

    if method == 'reference':
        sol = ten_beta0_reference(rho1, rho2, Tsat, Psat, model, s, n,
                                  full_output)
    elif method == 'cornelisse':
        sol = ten_beta0_hk(rho1, rho2, Tsat, Psat, model, n, full_output)
    elif method == 'liang':
        sol = ten_beta0_sk(rho1, rho2, Tsat, Psat, model, n, full_output,
                           alpha0)
    else:
        raise Exception('Method not implemented')

    return sol
