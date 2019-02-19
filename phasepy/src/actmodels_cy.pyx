from __future__ import division
import numpy as np
cimport cython
from libc.math cimport log

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef nrtl_cy(double [:] X, double [:,:] tau, double [:, :] G):
    
    cdef int i, j, k, nc
    nc = X.shape[0]
    cdef double A, SumA, SumB, SumC, SumD, SumE, aux, aux2
    cdef double [:] lngama = np.zeros(nc)
    
    for i in range(nc):
        SumC = SumD = SumE = 0.
        for j in range(nc):
            A = X[j]*G[i,j]
            SumA = SumB = 0.
            for k in range(nc):
                aux = X[k]*G[k,j]
                SumA += aux
                SumB += aux*tau[k,j]
            SumC += A/SumA*(tau[i,j]-SumB/SumA)
            aux2 = X[j]*G[j,i]
            SumD += aux2*tau[j,i]
            SumE += aux2
        lngama[i] = SumD/SumE+SumC
    return lngama.base

@cython.boundscheck(False)
@cython.wraparound(False)
def rkb_cy(double [:] x, double [:] G):
    
    cdef int m, i
    m = G.shape[0]
    cdef double SumA, SumB, dx, x1, x2
    cdef double[:] Mp = np.zeros(2)
    
    x1 = x[0]
    x2 = x[1]
    dx = x1 - x2
    SumA = G[0]
    SumB = 0.
    for i in range(1,m):
        SumA += G[i]*dx**i
        SumB += i*G[i]*dx**(i-1)
    Mp[0] = x2**2*(SumA+2*x1*SumB)
    Mp[1] = x1**2*(SumA-2*x2*SumB)
    return Mp.base


@cython.boundscheck(False)
@cython.wraparound(False)
def rk_cy(double [:] x, double [:, :] G, int [:,:] combinatoria):

    cdef int ncomb, m, nc, a, b, i, j
    ncomb = G.shape[0]
    m = G.shape[1]
    nc = x.shape[0]
    cdef double[:] Mp = np.zeros(nc)
    cdef double x1, x2, SumA, SumB, dx, aux
    
    
    for j in range(ncomb):
        a = combinatoria[j, 0]
        b = combinatoria[j, 1]
        x1 = x[a]
        x2 = x[b]
        SumA = SumB = G[j,0]
        dx = x1-x2
        for i in range(1,m):
            aux = (dx)**(i-1)*G[j,i]
            SumA += aux*(dx+2*x1*i)
            SumB += aux*(dx-2*x2*i)
        Mp[a] += SumA*x2**2
        Mp[b] += SumB*x1**2
        
    return Mp.base

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lnG_cy(double [:] tetha, double [:, :] psi, double [:] Qk):
    cdef int k, i, j, m
    k = tetha.shape[0]
    cdef double SumA, SumB, SumC
    cdef double [:] G = np.zeros(k)
    for i in range(k):
        SumA = 0.
        SumB = 0.
        for j in range(k):
            SumC = 0.
            for m in range(k):
                SumC += tetha[m]*psi[m, j]
            SumB += tetha[j]*psi[i,j]/SumC
            SumA += tetha[j]*psi[j,i]
        G[i] = Qk[i]*(1 - log(SumA) - SumB)
    return G.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def wilson_cy(double [:] X, double [:,:] M):
    
    cdef int nc, i, j, k
    nc = X.shape[0]
    cdef double SumA, SumB, SumC
    cdef double [:] lngama= np.zeros(nc)

    for i in range(nc):
        SumA = 0.
        SumB = 0.
        for j in range(nc):
            SumA += X[j]*M[i,j]
            SumC = 0.
            for k in range(nc):
                SumC += X[k]*M[j,k]
            SumB += X[j]*M[j,i]/SumC    
        lngama[i] = -log(SumA)+1-SumB
    return lngama.base