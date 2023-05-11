from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import minimize
from ..math import gdem
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
        # print(n_fluid+i-1, solid_phases_index[i])
        solid_list.append(dfug[n_fluid-1+i, solid_phases_index[i]])

    G = np.sum(fug * mole_number)
    dG = np.hstack([dfug[:(n_fluid-1)].flatten(), solid_list])
    return G, dG


def multiflash_solid(Z, T, P, model,
                     X_fluid, n_fluid, equilibrium_fluid,
                     X_solid, n_solid, lnphi_sol, solid_phases_index,
                     beta0=None, v0=[None],
                     K_tol=1e-10, nacc=5, full_output=False):
    nc = model.nc
    temp_aux = model.temperature_aux(T)

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
        betas = np.ones(n_phases)/n_phases
        tethas = np.zeros(n_phases-1)
        betatetha = np.hstack([betas, tethas])
    else:
        betatetha = np.hstack([beta0, np.zeros(n_phases-1)])

    # x is the iteration variable; x = [beta, theta]
    x = betatetha
    # start iterations
    error = 1
    it = 0
    itacc = 0
    ittotal = 0
    n = 5  # number of iterations to accumulate before ASS
    while error > K_tol and itacc < nacc:
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

        x[x <= 1e-10] = 0.
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
        error = np.sum((lnK - lnK_old)**2)

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
            lnK += dacc
        """
        if it == (n-2):
            lnK2 = lnK.flatten()
        elif it == (n-1):
            lnK1 = lnK.flatten()
        elif it == n:
            it = 0
            itacc += 1
            lnKf = lnK.flatten()
            dacc = dem(lnKf, lnK1, lnK2).reshape(lnK.shape)
            lnK += dacc
        # itacc += 1
        """

        # Updating K values
        K = np.exp(lnK)
        # Allowing the K factor ONLY for the specie present in the solid phase
        K[(n_fluid-1):][X[n_fluid:] == 0.] = 0.

    # if error > K_tol and itacc == nacc and np.all(tetha == 0):
    if error > K_tol and itacc == nacc and ef < 1e-8:
    # if True:
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
        min_sol = minimize(gibbs_obj, ind0, jac=True, args=args,
                           method='L-BFGS-B', bounds=bounds, tol=K_tol,
                           options={'gtol': K_tol, 'ftol': K_tol})

        # reconstructing the compositions vectors
        n_fluid_ind = ind0[:nc*(n_fluid-1)].reshape(n_fluid-1, nc)
        n_solid_ind = np.zeros([n_solid, nc])
        for i in range(n_solid):
            n_solid_ind[i, solid_phases_index[i]] = ind0[nc*(n_fluid-1) + i]
        n_ind = np.vstack([n_fluid_ind, n_solid_ind])
        n_dep = Z - np.sum(n_ind, axis=0)

        X = np.zeros([n_phases, nc])
        X[0] = n_dep
        X[1:n_fluid] = n_fluid_ind
        X[n_fluid:] = n_solid_ind
        beta = np.sum(X, axis=1)
        with np.errstate(all='ignore'):
            X = (X.T/beta).T

        # updating errors and iterations
        ittotal += min_sol.nit
        # betas_fluid = np.hstack([nc*[b] for b in beta[1:n_fluid]])
        # betas_opti = np.hstack([betas_fluid, beta[n_fluid:]])
        # error = np.linalg.norm(min_sol.jac * betas_opti)
        error = np.linalg.norm(min_sol.jac)
        # return min_sol.jac, beta
        # return min_sol

    X_fluid = X[:n_fluid]
    X_solid = X[n_fluid:]

    if full_output:
        sol = {'T': T, 'P': P, 'error_outer': error, 'error_inner': ef,
               'iter': ittotal, 'beta': beta, 'tetha': tetha,
               'X_fluid': X_fluid, 'v_fluid': v,
               'states_fluid': equilibrium_fluid,
               'X_solid': X_solid,
               }
        out = EquilibriumResult(sol)
    else: 
        out = X_fluid, X_solid, beta, tetha
    return out


def slle(Z, T, P, model,
         X_fluid0=None,  
         solid_phases_index=[], 
         beta0=None, v0=[None],
         K_tol=1e-10, nacc=5, full_output=False):

    if len(solid_phases_index) < 1:
        raise Exception('At least one solid phase must be given')

    # fluid phase set up
    if X_fluid0 is None:
        xll, wll = lle_init(Z, T, P, model)
        X_fluid = np.stack([wll, xll])
    n_fluid = len(X_fluid)
    equilibrium_fluid = n_fluid * ['L']
    
    # solid phase set up
    nc = model.nc
    eye = np.eye(nc)
    solid_phases_index = np.array(solid_phases_index, dtype=int)
    solid_phases_index = solid_phases_index[solid_phases_index >= 0] # to avoid negative indexes
    solid_phases_index = solid_phases_index[solid_phases_index <= (nc - 1)] # to avoid index bigger than nc
    solid_phases_index = np.unique(solid_phases_index) # to avoid duplicated same phases

    n_solid = len(solid_phases_index) # number of solid phases
    X_solid = [eye[index] for index in solid_phases_index]
    
    temp_aux = model.temperature_aux(T)
    # the fugacity coefficients are directly computed for the pure solid
    # so the given composition does not make any impact of the output logfug_sol
    # for simplicity I'm just using the global composition
    lnphi_sol = model.logfugef_aux(Z, temp_aux, P, state='S')[0]
    
    out = multiflash_solid(Z, T, P, model,
                           X_fluid, n_fluid, equilibrium_fluid,
                           X_solid, n_solid, lnphi_sol, solid_phases_index,
                           beta0=beta0, v0=v0,
                           K_tol=K_tol, nacc=nacc, full_output=full_output)
    return out


def sle(Z, T, P, model,
        X_fluid0=None,  
        solid_phases_index=[],
        beta0=None, v0=[None],
        K_tol=1e-10, nacc=5, full_output=False):

    if len(solid_phases_index) < 1:
        raise Exception('At least one solid phase must be given')

    # fluid phase set up
    if X_fluid0 is None:
        X_fluid = 1. * Z
    else:
        X_fluid = 1. * X_fluid0
    n_fluid = 1
    equilibrium_fluid = ['L']
    
    # solid phase set up
    nc = model.nc
    eye = np.eye(nc)
    solid_phases_index = np.array(solid_phases_index, dtype=int)
    solid_phases_index = solid_phases_index[solid_phases_index >= 0] # to avoid negative indexes
    solid_phases_index = solid_phases_index[solid_phases_index <= (nc - 1)] # to avoid index bigger than nc
    solid_phases_index = np.unique(solid_phases_index) # to avoid duplicated same phases

    n_solid = len(solid_phases_index) # number of solid phases
    X_solid = [eye[index] for index in solid_phases_index]
    
    temp_aux = model.temperature_aux(T)
    # the fugacity coefficients are directly computed for the pure solid
    # so the given composition does not make any impact of the output logfug_sol
    # for simplicity I'm just using the global composition
    lnphi_sol = model.logfugef_aux(Z, temp_aux, P, state='S')[0]
    
    out = multiflash_solid(Z, T, P, model,
                           X_fluid, n_fluid, equilibrium_fluid,
                           X_solid, n_solid, lnphi_sol, solid_phases_index,
                           beta0=beta0, v0=v0,
                           K_tol=K_tol, nacc=nacc, full_output=full_output)
    return out





