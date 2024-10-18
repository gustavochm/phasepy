from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import minimize
from ..math import gdem, dem
from .equilibriumresult import EquilibriumResult
from .multiflash import multiflash_obj
from .stability import lle_init


def gibbs_obj(ind0, n_fluid, n_solid, solid_phases_index,
              lnphi_sol, Z, temp_aux, P, model, equilibrium_fluid):

    nc = model.nc
    n_phases = n_fluid + n_solid

    n_fluid_ind = ind0[:nc*(n_fluid-1)].reshape(n_fluid-1, nc)
    n_solid_ind = np.zeros([n_solid, nc])
    for i in range(n_solid):
        n_solid_ind[i, solid_phases_index[i]] = ind0[nc*(n_fluid-1) + i]
    n_ind = np.vstack([n_fluid_ind, n_solid_ind])
    n_dep = Z - np.sum(n_ind, axis=0)

    X = np.zeros([n_phases, nc])
    X[0] = n_dep
    X[1:] = n_ind
    mole_number = X.copy()
    X[X < 1e-8] = 1e-8
    X = (X.T/X.sum(axis=1)).T

    lnphi = np.zeros_like(X)
    lnphi[n_fluid:] = lnphi_sol

    global vg
    for i, state in enumerate(equilibrium_fluid):
        lnphi[i], vg[i] = model.logfugef_aux(X[i], temp_aux, P, state, vg[i])

    with np.errstate(all='ignore'):
        fug = np.nan_to_num(np.log(X) + lnphi)
        dfug = fug[1:] - fug[0]

    solid_list = []
    for i in range(n_solid):
        solid_list.append(dfug[n_fluid-1+i, solid_phases_index[i]])

    G = np.sum(fug * mole_number)
    return G


def dgibbs_obj(ind0, n_fluid, n_solid, solid_phases_index,
               lnphi_sol, Z, temp_aux, P, model, equilibrium_fluid):

    nc = model.nc
    n_phases = n_fluid + n_solid

    n_fluid_ind = ind0[:nc*(n_fluid-1)].reshape(n_fluid-1, nc)
    n_solid_ind = np.zeros([n_solid, nc])
    for i in range(n_solid):
        n_solid_ind[i, solid_phases_index[i]] = ind0[nc*(n_fluid-1) + i]
    n_ind = np.vstack([n_fluid_ind, n_solid_ind])
    n_dep = Z - np.sum(n_ind, axis=0)

    X = np.zeros([n_phases, nc])
    X[0] = n_dep
    X[1:] = n_ind
    mole_number = X.copy()
    X[X < 1e-8] = 1e-8
    X = (X.T/X.sum(axis=1)).T

    lnphi = np.zeros_like(X)
    lnphi[n_fluid:] = lnphi_sol

    global vg
    for i, state in enumerate(equilibrium_fluid):
        lnphi[i], vg[i] = model.logfugef_aux(X[i], temp_aux, P, state, vg[i])

    with np.errstate(all='ignore'):
        fug = np.nan_to_num(np.log(X) + lnphi)
        dfug = fug[1:] - fug[0]

    solid_list = []
    for i in range(n_solid):
        solid_list.append(dfug[n_fluid-1+i, solid_phases_index[i]])

    G = np.sum(fug * mole_number)
    dG = np.hstack([dfug[:(n_fluid-1)].flatten(), solid_list])
    return G, dG


def multiflash_solid(Z, T, P, model,
                     X_fluid, n_fluid, equilibrium_fluid,
                     X_solid, n_solid, solid_phases_index,
                     beta0=None, v0=[None],
                     K_tol=1e-10, nacc=5, accelerate_every=5, full_output=False,
                     tetha_max=10., beta_min=1e-6):
    """
    Multiflash algorithm for equilibrium considering solid phases

    The solid phases are considered as pure phases, so the composition
    of the solid phases is not updated in the equilibrium calculation.

    Parameters
    ----------
    Z : array_like
        Overall composition
    T : float
        Temperature [K]
    P : float
        Pressure [bar]
    model : object
        created with eos and mixing rules
    X_fluid : array_like
        Composition of the fluid phases
    n_fluid : int
        Number of fluid phases
    equilibrium_fluid : list
        List of strings with the state of each fluid phase. 'L' for liquid
        'V' for vapor.
    X_solid : array_like
        Composition of the solid phases
    n_solid : int
        Number of solid phases
    solid_phases_index : list   
        List of indices of species allowed to solify.
    beta0 : array_like, optional
        Initial guess for the phase fraction of the phases.
        If None, the initial guess is 1/n_phases for each phase.
    v0 : array_like, optional
        Initial guess for the molar volume of the fluid phases.
    K_tol : float, optional
        Tolerance for the K factor.
    nacc : int, optional   
        Number of accelerated sustitution steps.
    accelerate_every : int, optional
        Number of iterations before to accelerate the sustitution.
        Mus be greater than or equal to 4. Set to zero to deactivate the
        acceleration.
    full_output : bool, optional
        If True, the output is a dictionary with all the information of the
        equilibrium. If False, the output is a tuple with the fluid and solid
        phases compositions, the phase fractions and the stability variables.
    tetha_max : float, optional
        Maximum value for the stability variable. Needed to avoid overflow.
        Default is 10.
    beta_min : float, optional
        Minimum value for the phase fraction. Default is 1e-6.
        Phases with fraction below this value are considered not present
        in the equilibrium.
    """

    # to make sure the compositions are arrays
    X_solid = np.asarray(X_solid)
    X_fluid = np.asarray(X_fluid)

    nc = model.nc
    temp_aux = model.temperature_aux(T)

    # the fugacity coefficients are directly computed for the pure solid
    lnphi_sol = np.zeros([n_solid, nc])
    for i in range(n_solid):
        with np.errstate(all='ignore'):
            lnphi_sol[i] = model.logfugef_aux(X_solid[i], temp_aux, P, 'L')[0]
            lnphi_sol[i] -= (model.dHf_r) * (1. / T - 1. / model.Tf)
    # This is needed just in case any dHf_r or Tf is zero
    lnphi_sol = np.nan_to_num(lnphi_sol)

    if len(v0) == 1 and len(v0) != n_fluid:
        v0 *= n_fluid
    v = v0.copy()

    X = np.vstack([X_fluid, X_solid])
    lnphi = np.zeros_like(X)
    lnphi[n_fluid:] = lnphi_sol
    for i, state in enumerate(equilibrium_fluid):
        lnphi[i], v[i] = model.logfugef_aux(X[i], temp_aux, P, state, v0[i])

    lnK = lnphi[0] - lnphi[1:]
    K = np.exp(lnK)
    # Allowing the K factor ONLY for the specie present in the solid phase
    K[(n_fluid-1):][X[n_fluid:] == 0.] = 0.

    n_phases = n_fluid + n_solid
    if beta0 is None:
        beta = np.ones(n_phases)/n_phases
        tetha = np.zeros(n_phases-1)
        betatetha = np.hstack([beta, tetha])
    else:
        betatetha = np.hstack([beta0, np.zeros(n_phases-1)])

    # x is the iteration variable; x = [beta, theta]
    x = betatetha
    # start iterations
    error_outer = 1.
    it = 0
    itacc = 0
    ittotal = 0
    n = accelerate_every  # number of iterations to accumulate before ASS
    method = "ASS"
    while error_outer > K_tol and itacc < nacc:
        ittotal += 1
        it += 1
        lnK_old = lnK.copy()

        ef = 1.
        ex = 1.
        itin = 0
        while ef > 1e-8 and ex > 1e-8 and itin < 30:
            itin += 1
            f, jac, Kexp, Xref = multiflash_obj(x, Z, K)
            dx = np.linalg.solve(jac, -f)
            x += dx
            ef = np.linalg.norm(f)
            ex = np.linalg.norm(dx)
            x[x < beta_min] = 0.
            x[x > tetha_max] = tetha_max
        error_inner = ef

        beta, tetha = np.array_split(x, 2)
        beta /= beta.sum()

        # Update compositions
        X[0] = Xref
        X[1:(n_fluid)] = (Xref*Kexp.T)[:(n_fluid-1)]
        X = np.abs(X)
        X = (X.T/X.sum(axis=1)).T

        for i, state in enumerate(equilibrium_fluid):
            lnphi[i], v[i] = model.logfugef_aux(X[i], temp_aux, P, state, v[i])

        lnK = lnphi[0] - lnphi[1:]
        error_outer = np.sum((lnK - lnK_old)**2)

        # Accelerate succesive sustitution
        if it == (n-3):
            lnK3 = lnK.flatten()
        elif it == (n-2):
            lnK2 = lnK.flatten()
        elif it == (n-1):
            lnK1 = lnK.flatten()
        elif it == n:
            it = 0
            itacc += 1
            lnKf = lnK.flatten()
            dacc = gdem(lnKf, lnK1, lnK2, lnK3).reshape(lnK.shape)
            if np.all(np.logical_not(np.isnan(dacc))):
                lnK += dacc

        # Updating K values
        K = np.exp(lnK)
        # Allowing the K factor ONLY for the specie present in the solid phase
        K[(n_fluid-1):][X[n_fluid:] == 0.] = 0.

    # if error > K_tol and itacc == nacc and np.all(tetha == 0):
    # if True:
    if error_outer > K_tol and itacc == nacc:
        method = "Gibbs minimization"
        global vg
        vg = v.copy()

        mole_number = (X.T*beta).T
        solid_list = []
        for i in range(n_solid):
            solid_list.append(mole_number[n_fluid+i, solid_phases_index[i]])
        ind0 = np.hstack([mole_number[1:n_fluid].flatten(), solid_list])
        args = (n_fluid, n_solid, solid_phases_index,
                lnphi_sol, Z, temp_aux, P, model, equilibrium_fluid)
        bounds = len(ind0) * [(0, None)]

        # If the initial guess is "poor" the jacobian might suggest a wrong direction
        # This is just to improve the initial guess without any derivative information
        Gsol_init = minimize(gibbs_obj, ind0, args=args, method='SLSQP', bounds=bounds,
                             tol=K_tol, options={'maxiter': 15})
        ittotal += Gsol_init.nit
        Gsol = Gsol_init
        jac = Gsol_init.jac

        # Now that we have a better initial guess, we can use the jacobian
        # L-BFGS-B method seems to work well after this step but the final solution might not be super accurate
        Gsol_LBFGSB = minimize(dgibbs_obj, Gsol_init.x, jac=True, args=args,
                               method='L-BFGS-B', bounds=bounds, tol=K_tol,
                               options={'gtol': K_tol, 'ftol': K_tol})

        if Gsol_LBFGSB.fun < Gsol.fun:
            ittotal += Gsol_LBFGSB.nit
            Gsol = Gsol_LBFGSB
            jac = Gsol_LBFGSB.jac

        # Refinining the solution. SLSQP works well when there is a very good initial guess
        if np.mean(jac**2) > K_tol:
            Gsol_slsqp = minimize(dgibbs_obj, Gsol.x, jac=True, args=args,
                                  method='SLSQP', bounds=bounds, tol=K_tol)
            if Gsol_slsqp.fun < Gsol.fun:
                ittotal += Gsol_slsqp.nit
                Gsol = Gsol_slsqp
                jac = Gsol_slsqp.jac

        # reconstructing the compositions vectors
        ind_sol = Gsol.x
        n_fluid_ind = ind_sol[:nc*(n_fluid-1)].reshape(n_fluid-1, nc)
        n_solid_ind = np.zeros([n_solid, nc])
        for i in range(n_solid):
            n_solid_ind[i, solid_phases_index[i]] = ind_sol[nc*(n_fluid-1) + i]
        n_ind = np.vstack([n_fluid_ind, n_solid_ind])
        n_dep = Z - np.sum(n_ind, axis=0)

        X = np.zeros([n_phases, nc])
        X[0] = n_dep
        X[1:n_fluid] = n_fluid_ind
        X[n_fluid:] = n_solid_ind
        beta = np.sum(X, axis=1)

        with np.errstate(all='ignore'):
            X = (X.T/beta).T

        # deleting phases that are below the threshold
        beta[beta <= beta_min] = 0.
        beta /= np.sum(beta)

        # if any fluid phase has been deleted the it is set to the reference fluid phase
        beta_fluid_zero = beta == 0
        beta_fluid_zero[n_fluid:] = False  # do not change the composition of solid phases
        X[beta_fluid_zero] = X[0]

        # minimization should have eliminated unstable phases, setting
        # the stability variables to zero
        tetha = np.zeros(n_phases-1)

        # Inner error is related to mass balance
        error_inner = np.linalg.norm(Z - np.nansum(X.T*beta, axis=1))

        # The jacobian might not be zero for phases not present in the
        # equilibrium result
        where_beta_bools = beta > 0
        where_beta_bools_fluid = where_beta_bools[:n_fluid]
        where_beta_bools_solid = where_beta_bools[(n_fluid):]

        where_jac_fluid = [nc * [x] for x in where_beta_bools_fluid][1:]
        where_jac_fluid = [x for xs in where_jac_fluid for x in xs]
        where_jac = np.hstack([where_jac_fluid, where_beta_bools_solid]).astype(bool)

        error_outer = np.linalg.norm(jac[where_jac])

    X_fluid = X[:n_fluid]
    # X_solid = X[n_fluid:]

    if full_output:
        sol = {'T': T, 'P': P, 'error_outer': error_outer,
               'error_inner': error_inner,
               'iter': ittotal, 'beta': beta, 'tetha': tetha,
               'X_fluid': X_fluid, 'v_fluid': v,
               'states_fluid': equilibrium_fluid,
               'X_solid': X_solid, 'method': method}
        out = EquilibriumResult(sol)
    else: 
        out = X_fluid, X_solid, beta, tetha
    return out


def slle(Z, T, P, model,
         X_fluid0=None,
         solid_phases_index=[],
         beta0=None, v0=[None],
         K_tol=1e-10, nacc=5, accelerate_every=5, full_output=False,
         tetha_max=10., beta_min=1e-6):

    """
    Solid-liquid-liquid equilibrium (SLLE) calculation for a mixture.
    Function to compute the solid-liquid-liquid equilibrium of a mixture
    at given temperature, pressure and global composition.

    Parameters
    ----------
    Z : array_like
        Global composition of the mixture.
    T : float
        Temperature of the mixture [K].
    P : float
        Pressure of the mixture [bar].
    model : object
        created from mixture, eos and mixrule
    solid_phases_index : array_like
        Indexes of the solid phases to be considered in the equilibrium.
    X_fluid0 : array_like, optional
        Initial guess for the fluid phases compositions.
        If not given, the initial guess is same as the global composition.
    beta0 : array_like, optional
        Initial guess for the phase fractions.
        If not given, all phases are assumed to have the same fraction.
    v0 : array_like, optional
        Initial guess for the molar volumes of the fluid phases.
    K_tol : float, optional
        Tolerance for the phase equilibrium.
    nacc : int, optional
        Number of accelerated successive substitution cycles.
    accelerate_every : int, optional
        Number of iterations before to accelerate the sustitution.
        Mus be greater than or equal to 4. Set to zero to deactivate the
        acceleration.
    full_output : bool, optional
        If True, the output is a dictionary with all the information of the
        equilibrium. If False, the output is a tuple with the fluid and solid
        phases compositions, the phase fractions and the stability variables.
    tetha_max : float, optional
        Maximum value for the stability variable. Needed to avoid overflow.
        Default is 10.
    beta_min : float, optional
        Minimum value for the phase fraction. Default is 1e-6.
        Phases with fraction below this value are considered not present
        in the equilibrium.

    Returns
    -------
    X_fluid : ndarray
        Fluid phases compositions.
    X_solid : ndarray
        Solid phases compositions.
    beta : ndarray
        Phase fractions [fluid phases, solid phases].
    tetha : ndarray
        Stability variables [fluid phases[1:], solid phases].
    """
    if len(solid_phases_index) < 1:
        raise Exception('At least one solid phase must be given')

    # fluid phase set up
    if X_fluid0 is None:
        xll, wll = lle_init(Z, T, P, model)
        X_fluid = np.stack([xll, wll])
    else:
        X_fluid = np.asarray(X_fluid0)
    n_fluid = len(X_fluid)
    equilibrium_fluid = n_fluid * ['L']

    # solid phase set up
    nc = model.nc
    eye = np.eye(nc)
    solid_phases_index = np.array(solid_phases_index, dtype=int)
    solid_phases_index = solid_phases_index[solid_phases_index >= 0] # to avoid negative indexes
    solid_phases_index = solid_phases_index[solid_phases_index <= (nc - 1)] # to avoid index bigger than nc
    solid_phases_index = np.unique(solid_phases_index) # to avoid duplicated same phases

    if len(solid_phases_index) < 1:
        raise Exception('Valid solid phase indexes must be given (0 to nc-1)')

    if np.any(model.Tf[solid_phases_index] == 0.) or np.any(model.dHf[solid_phases_index] == 0.):
        raise Exception('The solid phase(s) must have a valid melting temperature and heat of fusion')

    n_solid = len(solid_phases_index)  # number of solid phases
    X_solid = [eye[index] for index in solid_phases_index]

    out = multiflash_solid(Z, T, P, model,
                           X_fluid, n_fluid, equilibrium_fluid,
                           X_solid, n_solid, solid_phases_index,
                           beta0=beta0, v0=v0,
                           K_tol=K_tol, nacc=nacc,
                           accelerate_every=accelerate_every,
                           full_output=True, tetha_max=tetha_max,
                           beta_min=beta_min)

    # if the mass balanced failed it is likely the reference phase is not
    # the stable phase, so we try again with the second liquid as reference
    # phase
    if out.error_inner > 1e-6 or np.isnan(out.error_inner):
        X_fluid = X_fluid[::-1]
        v0 = v0[::-1]
        out = multiflash_solid(Z, T, P, model,
                               X_fluid, n_fluid, equilibrium_fluid,
                               X_solid, n_solid, solid_phases_index,
                               beta0=beta0, v0=v0,
                               K_tol=K_tol, nacc=nacc,
                               accelerate_every=accelerate_every,
                               full_output=True, tetha_max=tetha_max,
                               beta_min=beta_min)

    if not full_output:
        out = out.X_fluid, out.X_solid, out.beta, out.tetha

    return out


def sle(Z, T, P, model,
        X_fluid0=None,
        solid_phases_index=[],
        beta0=None, v0=[None],
        K_tol=1e-10, nacc=5,
        accelerate_every=5,
        full_output=False,
        tetha_max=10., beta_min=1e-6):
    """
    Solid-liquid equilibrium
    Function to compute the solid-liquid equilibrium of a mixture
    at given temperature, pressure and global composition.

    Parameters
    ----------
    Z : array_like
        Global composition of the mixture.
    T : float
        Temperature of the mixture [K].
    P : float
        Pressure of the mixture [bar].
    model : object
        created from mixture, eos and mixrule
    solid_phases_index : array_like
        Indexes of the solid phases to be considered in the equilibrium.
    X_fluid0 : array_like, optional
        Initial guess for the fluid phases compositions.
        If not given, the initial guess is same as the global composition.
    beta0 : array_like, optional
        Initial guess for the phase fractions.
        If not given, all phases are assumed to have the same fraction.
    v0 : array_like, optional
        Initial guess for the molar volumes of the fluid phases.
    K_tol : float, optional
        Tolerance for the phase equilibrium.
    nacc : int, optional
        Number of accelerated successive substitution cycles.
    accelerate_every : int, optional
        Number of iterations before to accelerate the sustitution.
        Mus be greater than or equal to 4. Set to zero to deactivate the
        acceleration.
    full_output : bool, optional
        If True, the output is a dictionary with all the information of the
        equilibrium. If False, the output is a tuple with the fluid and solid
        phases compositions, the phase fractions and the stability variables.
    tetha_max : float, optional
        Maximum value for the stability variable. Needed to avoid overflow.
        Default is 10.
    beta_min : float, optional
        Minimum value for the phase fraction. Default is 1e-6.
        Phases with fraction below this value are considered not present
        in the equilibrium.

    Returns
    -------
    X_fluid : ndarray
        Fluid phases compositions.
    X_solid : ndarray
        Solid phases compositions.
    beta : ndarray
        Phase fractions [fluid phases, solid phases].
    tetha : ndarray
        Stability variables [fluid phases[1:], solid phases].
    """

    if len(solid_phases_index) < 1:
        raise Exception('At least one solid phase must be given')

    # fluid phase set up
    if X_fluid0 is None:
        X_fluid = 1. * Z
    else:
        X_fluid = 1. * np.asarray(X_fluid0)
    n_fluid = 1
    equilibrium_fluid = ['L']

    # solid phase set up
    nc = model.nc
    eye = np.eye(nc)
    solid_phases_index = np.array(solid_phases_index, dtype=int)
    solid_phases_index = solid_phases_index[solid_phases_index >= 0] # to avoid negative indexes
    solid_phases_index = solid_phases_index[solid_phases_index <= (nc - 1)] # to avoid index bigger than nc
    solid_phases_index = np.unique(solid_phases_index) # to avoid duplicated same phases

    if len(solid_phases_index) < 1:
        raise Exception('Valid solid phase indexes must be given (0 to nc-1)')

    if np.any(model.Tf[solid_phases_index] == 0.) or np.any(model.dHf[solid_phases_index] == 0.):
        raise Exception('The solid phase(s) must have a valid melting temperature and heat of fusion')

    n_solid = len(solid_phases_index) # number of solid phases
    X_solid = [eye[index] for index in solid_phases_index]

    out = multiflash_solid(Z, T, P, model,
                           X_fluid, n_fluid, equilibrium_fluid,
                           X_solid, n_solid, solid_phases_index,
                           beta0=beta0, v0=v0,
                           K_tol=K_tol, nacc=nacc,
                           accelerate_every=accelerate_every,
                           full_output=full_output,
                           tetha_max=tetha_max, beta_min=beta_min)

    return out
