from __future__ import division, print_function, absolute_import
import numpy as np


# Quadratic mixrule
def qmr(X, RT, ai, bi, order, Kij):
    '''
    Quadratic mixrule QMR

    Inputs
    ----------
    X : molar fraction array [x1, x2, ..., xc]
    RT: Absolute temperature in K plus R
    ai :  pure component attrative term in bar cm6/mol2
    bi :  pure component cohesive term in cm3/mol
    Kij : matrix of interaction parameters


    Out :
    D (mixture a term)
    Di (mixture a term first derivative)
    Dij (mixture a term second derivative)
    B (mixture b term)
    Bi (mixture b term first derivative)
    Bij (mixture a term second derivative)
    '''

    aij = np.sqrt(np.outer(ai, ai))*(1-Kij)
    ax = aij * X
    # Atractive term of mixture
    D = np.sum(ax.T*X)
    # Mixture covolume
    B = np.dot(bi, X)

    if order == 0:
        mixparameters = D, B
    elif order == 1:
        Di = 2*np.sum(ax, axis=1)
        Bi = bi
        mixparameters = D, Di, B, Bi
    elif order == 2:
        Di = 2*np.sum(ax, axis=1)
        Dij = 2*aij
        Bi = bi
        Bij = 0.
        mixparameters = D, Di, Dij, B, Bi, Bij
    else:
        raise Exception('Derivative order not valid')

    return mixparameters
