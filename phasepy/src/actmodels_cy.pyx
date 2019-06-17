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
cdef x_auxf(double [:] x, double [:] x_aux, int n):
    cdef int i
    for i in range(n):
        x_aux[i] = x[i]

@cython.boundscheck(False)
@cython.wraparound(False)
def rkter_nrtl_cy(double [:] x, double [:] xd):
    cdef int n, i, k
    n = 3
    cdef double [:] q = np.zeros(n)
    cdef double [:] x_aux = np.zeros(n)
    cdef double x1, x2, x3
    x_auxf(x, x_aux, n)
    for i in range(n):
        if x_aux[i] !=0:
            x_aux[i] = 1.
        x1, x2, x3 = x_aux[0], x_aux[1], x_aux[2]
        for k in range(n):
            if k != i:
                q[i] -= (-1+3*x[i])*xd[k]
            else:
                q[i] += (2-3*x[i])*xd[k]
        q[i] *= x1*x2*x3
        x_auxf(x, x_aux, n)
    return q.base

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

ctypedef fused int_or_long:
    cython.int
    cython.long

@cython.boundscheck(False)
@cython.wraparound(False)
def rk_cy(double [:] x, double [:, :] G, int_or_long [:,:] combinatory):

    cdef int ncomb, m, nc, i, j, k, l
    ncomb = combinatory.shape[0]
    m = G.shape[1]
    nc = x.shape[0]
    cdef double[:] dge = np.zeros(nc)
    cdef double xi, xj, SumA, SumC, dx, ge, xixj, aux
    
    ge = 0.
    for k in range(ncomb):
        i = combinatory[k, 0]
        j = combinatory[k, 1]
        xi = x[i]
        xj = x[j]
        dx = xi - xj
        SumA = G[k, 0]
        SumC = 0.
        for l in range(1, m):
            SumA += G[k,l]*dx**l
            SumC += G[k,l]*l*dx**(l-1.)
        xixj = xi*xj
        aux = xixj*SumC
        ge +=  xixj*SumA
        dge[i] += xj*SumA + aux
        dge[j] += xi*SumA - aux
        
    return ge, dge.base


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