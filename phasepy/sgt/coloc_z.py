from __future__ import division, print_function, absolute_import
import numpy as np
from .tensionresult import TensionResult
from ..math import gauss, colocAB
from scipy.optimize import root
from scipy.interpolate import interp1d
from .cijmix_cy import cmix_cy


def fobj_z_newton(rointer, Binter, dro20, dro21, mu0, temp_aux, cij, n,
                  nc, model):
    rointer = np.abs(rointer.reshape([nc, n]))
    dmu = np.zeros([n, nc])

    for i in range(n):
        dmu[i] = model.muad_aux(rointer[:, i], temp_aux)
    dmu -= mu0
    dmu = dmu.T

    dro2 = np.matmul(rointer, Binter.T)
    dro2 += dro20
    dro2 += dro21

    ter1 = np.matmul(cij, dro2)
    fo = ter1 - dmu
    return fo.flatten()


def dfobj_z_newton(rointer, Binter, dro20, dro21, mu0, temp_aux, cij, n,
                   nc, model):

    index0 = rointer < 0
    rointer = np.abs(rointer.reshape([nc, n]))
    dmu = np.zeros([n, nc])
    d2mu = np.zeros([n, nc, nc])

    for i in range(n):
        dmu[i], d2mu[i] = model.dmuad_aux(rointer[:, i], temp_aux)
    dmu -= mu0
    dmu = dmu.T
    d2mu = d2mu.T

    dro2 = np.matmul(rointer, Binter.T)
    dro2 += dro20
    dro2 += dro21

    ter1 = np.matmul(cij, dro2)
    fo = ter1 - dmu

    eye = np.eye(n, n)
    dml = []
    for i in range(nc):
        auxlist = []
        for j in range(nc):
            auxlist.append(eye * d2mu[i, j])
        dml.append(auxlist)
    jac_dmu = np.block(dml)

    jacter1 = np.multiply.outer(cij, Binter.T).reshape(nc, nc*n, n)
    fo_jac = np.hstack(jacter1).T - jac_dmu
    fo_jac[:, index0] *= -1

    return fo.flatten(), fo_jac


def sgt_mix(rho1, rho2, Tsat, Psat, model, rho0='linear',
            z0=10., dz=1.5, itmax=10, n=20, full_output=False,
            ten_tol=1e-2, root_method='lm', solver_opt=None):
    """
    SGT for mixtures and beta != 0 (rho1, rho2, T, P) -> interfacial tension

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
    rho0 : string, array_like or TensionResult
        inital values to solve the BVP, avaialable options are 'linear' for
        linear density profiles, 'hyperbolic' for hyperbolic
        like density profiles. An array can also be supplied or a
        TensionResult of a previous calculation.
    z0 :  float
        initial interfacial lenght
    dz : float
        increase in interfacial lenght per interation
    itmax : int
        maximun number of iterations
    n : int, optional
        number points to solve density profiles
    full_output : bool, optional
        wheter to outputs all calculation info
    root_method: string, optional
        Method used un SciPy's root function
        default 'lm',  other options: 'krylov', 'hybr'. See SciPy documentation
        for more info
    solver_opt : dict, optional
        aditional solver options passed to SciPy solver

    Returns
    -------
    ten : float
        interfacial tension between the phases
    """

    z = z0
    nc = model.nc

    # Dimensionless Variables
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    rho1a = rho1*rofactor
    rho2a = rho2*rofactor

    cij = model.ci(Tsat)
    cij /= cij[0, 0]
    dcij = np.linalg.det(cij)
    if np.isclose(dcij, 0):
        warning = 'Determinant of influence parameters matrix is: {}'
        raise Exception(warning.format(dcij))

    temp_aux = model.temperature_aux(Tsat)

    # Chemical potential
    mu0 = model.muad_aux(rho1a, temp_aux)
    mu02 = model.muad_aux(rho2a, temp_aux)
    if not np.allclose(mu0, mu02):
        raise Exception('Not equilibria compositions, mu1 != mu2')

    # Nodes and weights of integration
    roots, weights = gauss(n)
    rootsf = np.hstack([0., roots, 1.])
    # Coefficent matrix for derivatives
    A, B = colocAB(rootsf)

    # Initial profiles
    if isinstance(rho0, str):
        if rho0 == 'linear':
            # Linear Profile
            pend = (rho2a - rho1a)
            b = rho1a
            pfl = (np.outer(roots, pend) + b)
            rointer = (pfl.T).copy()
        elif rho0 == 'hyperbolic':
            # Hyperbolic profile
            inter = 8*roots - 4
            thb = np.tanh(2*inter)
            pft = np.outer(thb, (rho2a-rho1a))/2+(rho1a+rho2a)/2
            rointer = pft.T
        else:
            raise Exception('Initial density profile not known')
    elif isinstance(rho0,  TensionResult):
        _z0 = rho0.z
        _ro0 = rho0.rho
        z = _z0[-1]
        rointer = interp1d(_z0, _ro0)(roots * z)
        rointer *= rofactor
    elif isinstance(rho0,  np.ndarray):
        # Check dimensiones
        if rho0.shape[0] == nc and rho0.shape[1] == n:
            rointer = rho0.copy()
            rointer *= rofactor
        else:
            raise Exception('Shape of initial value must be nc x n')
    else:
        raise Exception('Initial density profile not known')

    if root_method == 'lm' or root_method == 'hybr':
        if model.secondordersgt:
            fobj = dfobj_z_newton
            jac = True
        else:
            fobj = fobj_z_newton
            jac = None
    else:
        fobj = fobj_z_newton
        jac = None

    error = 1.
    it = 0
    ten_old = 0.
    tol = ten_tol
    while error > tol and it < itmax:
        it += 1
        zad = z*zfactor
        Ar = A/zad
        Br = B/zad**2

        Binter = Br[1:-1, 1:-1]
        B0 = Br[1:-1, 0]
        B1 = Br[1:-1, -1]
        dro20 = np.outer(rho1a, B0)  # cte
        dro21 = np.outer(rho2a, B1)  # cte

        Ainter = Ar[1:-1, 1:-1]
        A0 = Ar[1:-1, 0]
        A1 = Ar[1:-1, -1]
        dro10 = np.outer(rho1a, A0)  # cte
        dro11 = np.outer(rho2a, A1)  # cte

        sol = root(fobj, rointer.flatten(), method=root_method, jac=jac,
                   args=(Binter, dro20, dro21, mu0, temp_aux, cij, n, nc,
                   model), options=solver_opt)

        rointer = sol.x
        rointer = np.abs(rointer.reshape([nc, n]))

        dro = np.matmul(rointer, Ainter.T)
        dro += dro10
        dro += dro11

        suma = cmix_cy(dro, cij)
        dom = np.zeros(n)
        for k in range(n):
            dom[k] = model.dOm_aux(rointer[:, k], temp_aux, mu0, Pad)
        dom[dom < 0] = 0.
        intten = np.nan_to_num(np.sqrt(2*suma*dom))
        ten = np.dot(intten, weights)
        ten *= zad
        ten *= tenfactor
        error = np.abs(ten - ten_old)
        ten_old = ten
        z += dz

    fun_error = np.linalg.norm(sol.fun)/n/nc
    success = sol.success and (error < tol)

    if full_output:

        znodes = (z-dz) * rootsf
        ro = np.insert(rointer, 0, rho1a, axis=1)
        ro = np.insert(ro, n+1, rho2a, axis=1)
        ro /= rofactor
        dictresult = {'tension': ten, 'rho': ro, 'z': znodes,
                      'GPT': np.hstack([0, dom, 0]),
                      'success': success,
                      'message': sol.message,
                      'fun_norm': fun_error,
                      'error': error, 'iter': it}
        out = TensionResult(dictresult)
        return out

    return ten


# Function for solving sgt for a fixed interfacial lenght
def sgt_zfixed(rho1, rho2, Tsat, Psat, model, rho0='linear', z=10, n=20,
               full_output=False, root_method='lm', solver_opt=None):
    """
    SGT for mixtures and beta != 0 (rho1, rho2, T, P) -> interfacial tension

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
    rho0 : string, array_like or TensionResult
        inital values to solve the BVP, avaialable options are 'linear' for
        linear density profiles, 'hyperbolic' for hyperbolic
        like density profiles. An array can also be supplied or a
        TensionResult of a previous calculation.
    z :  float
        initial interfacial lenght
    n : int, optional
        number points to solve density profiles
    full_output : bool, optional
        wheter to outputs all calculation info
    root_method: string, optional
        Method used un SciPy's root function
        default 'lm',  other options: 'krylov', 'hybr'. See SciPy documentation
        for more info
    solver_opt : dict, optional
        aditional solver options passed to SciPy solver

    Returns
    -------
    ten : float
        interfacial tension between the phases
    """

    nc = model.nc

    # Dimensionless Variables
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    rho1a = rho1*rofactor
    rho2a = rho2*rofactor

    cij = model.ci(Tsat)
    cij /= cij[0, 0]
    dcij = np.linalg.det(cij)
    if np.isclose(dcij, 0):
        warning = 'Determinant of influence parameters matrix is: {}'
        raise Exception(warning.format(dcij))

    temp_aux = model.temperature_aux(Tsat)

    # Chemical potential
    mu0 = model.muad_aux(rho1a, temp_aux)
    mu02 = model.muad_aux(rho2a, temp_aux)
    if not np.allclose(mu0, mu02):
        raise Exception('Not equilibria compositions, mu1 != mu2')

    # Nodes and weights of integration
    roots, weights = gauss(n)
    rootsf = np.hstack([0., roots, 1.])
    # Coefficent matrix for derivatives
    A, B = colocAB(rootsf)

    # Initial profiles
    if rho0 == 'linear':
        # Linear Profile
        pend = (rho2a - rho1a)
        b = rho1a
        pfl = (np.outer(roots, pend) + b)
        rointer = (pfl.T).copy()
    elif rho0 == 'hyperbolic':
        # Hyperbolic profile
        inter = 8*roots - 4
        thb = np.tanh(2*inter)
        pft = np.outer(thb, (rho2a-rho1a))/2+(rho1a+rho2a)/2
        rointer = pft.T
    elif isinstance(rho0,  TensionResult):
        _z0 = rho0.z
        _ro0 = rho0.rho
        z = _z0[-1]
        rointer = interp1d(_z0, _ro0)(roots * z)
        rointer *= rofactor
    elif isinstance(rho0,  np.ndarray):
        # Check dimensiones
        if rho0.shape[0] == nc and rho0.shape[1] == n:
            rointer = rho0.copy()
            rointer *= rofactor
        else:
            raise Exception('Shape of initial value must be nc x n')

    zad = z*zfactor
    Ar = A/zad
    Br = B/zad**2

    Binter = Br[1:-1, 1:-1]
    B0 = Br[1:-1, 0]
    B1 = Br[1:-1, -1]
    dro20 = np.outer(rho1a, B0)  # cte
    dro21 = np.outer(rho2a, B1)  # cte

    Ainter = Ar[1:-1, 1:-1]
    A0 = Ar[1:-1, 0]
    A1 = Ar[1:-1, -1]
    dro10 = np.outer(rho1a, A0)  # cte
    dro11 = np.outer(rho2a, A1)  # cte

    if model.secondordersgt:
        fobj = dfobj_z_newton
        jac = True
    else:
        fobj = fobj_z_newton
        jac = None

    sol = root(fobj, rointer.flatten(), method=root_method, jac=jac,
               args=(Binter, dro20, dro21, mu0, temp_aux, cij, n, nc, model),
               options=solver_opt)

    rointer = sol.x
    rointer = np.abs(rointer.reshape([nc, n]))

    dro = np.matmul(rointer, Ainter.T)
    dro += dro10
    dro += dro11

    suma = cmix_cy(dro, cij)
    dom = np.zeros(n)
    for k in range(n):
        dom[k] = model.dOm_aux(rointer[:, k], temp_aux, mu0, Pad)
    dom[dom < 0] = 0.
    intten = np.nan_to_num(np.sqrt(2*suma*dom))
    ten = np.dot(intten, weights)
    ten *= zad
    ten *= tenfactor
    fun_error = np.linalg.norm(sol.fun)/n/nc
    success = sol.success

    if full_output:

        znodes = z * rootsf
        ro = np.insert(rointer, 0, rho1a, axis=1)
        ro = np.insert(ro, n+1, rho2a, axis=1)
        ro /= rofactor
        dictresult = {'tension': ten, 'rho': ro, 'z': znodes,
                      'GPT': np.hstack([0, dom, 0]),
                      'success': success,
                      'message': sol.message,
                      'fun_norm': fun_error}
        out = TensionResult(dictresult)
        return out

    return ten
