from __future__ import division, print_function, absolute_import
import numpy as np
from .coloc_cy import jcobi_roots, colocAB, colocA, colocB


def gauss(n):
    roots, dif1 = jcobi_roots(n, N0=0, N1=0, Al=0, Be=0)
    ax = 1
    ax /= roots
    ax /= (1-roots)
    vect = ax/dif1**2
    vect /= np.sum(vect)
    return roots, vect


def lobatto(n):
    roots, dif1 = jcobi_roots(n - 2, N0=1, N1=1, Al=1, Be=1)
    s0 = 2/(n-1)/n
    vect = s0/dif1**2
    vect /= np.sum(vect)
    return roots, vect


def gdem(X, X1, X2, X3):
    dX2 = X - X3
    dX1 = X - X2
    dX = X - X1
    b01 = dX.dot(dX1)
    b02 = dX.dot(dX2)
    b12 = dX1.dot(dX2)
    b11 = dX1.dot(dX1)
    b22 = dX2.dot(dX2)
    den = b11*b22-b12**2
    with np.errstate(divide='ignore', invalid='ignore'):
        mu1 = (b02*b12 - b01*b22)/den
        mu2 = (b01*b12 - b02*b11)/den
        dacc = (dX - mu2*dX1)/(1+mu1+mu2)
    return np.nan_to_num(dacc)


__all__ = ['gauss', 'lobatto', 'colocAB', 'colocA', 'colocB', 'gdem']
