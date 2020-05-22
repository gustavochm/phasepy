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
@cython.cdivision(True)
cpdef dnrtl_cy(double [:] X, double [:,:] tau, double [:, :] G):
    cdef int i, j, k, nc
    nc = X.shape[0]
    cdef double [:] lngama = np.zeros(nc)
    cdef double [:, :] dlngama = np.zeros([nc, nc])
    cdef double SumA, SumB, aux

    cdef double[:] Ci = np.zeros([nc])
    cdef double[:] Si = np.zeros([nc])
    cdef double[:, :] epsij = np.zeros([nc, nc])

    for i in range(nc):
        for j in range(nc):
            aux = X[j] * G[j, i]
            Si[i] += aux
            Ci[i] += aux * tau[j,i]

    for i in range(nc):
        for j in range(nc):
            epsij[i,j] = G[i,j] * (tau[i,j] - Ci[j]/Si[j])/Si[j]

    for i in range(nc):
        SumA = 0.
        for k in range(nc):
            SumA += X[k] * epsij[i, k]
        lngama[i] = Ci[i] / Si[i] + SumA

    for i in range(nc):
        for j in range(i, nc):
            SumB = 0.
            for k in range(nc):
                SumB += X[k] * (G[i, k]*epsij[j,k] + G[j, k] * epsij[i,k]) / Si[k]
            dlngama[i,j] = epsij[i,j] + epsij[j,i] - SumB
            dlngama[j,i] = dlngama[i,j]

    return lngama.base , dlngama.base

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
def drkter_nrtl_cy(double [:] x, double [:] xd, double[:] d):

    cdef int n, i, k
    n = 3
    cdef double [:] lngama = np.zeros(n)
    cdef double [:, :] dlngama = np.zeros([n, n])
    cdef double [:] x_aux = np.zeros(n)
    cdef double [:] x_aux2 = np.zeros(n)
    cdef double x1, x2, x3, x12, x22, x32, gex

    gex = 0.
    for i in range(3):
        gex += xd[i]

    for i in range(3):
        for j in range(3):
            x_auxf(x, x_aux, n)
            x_aux[i] = 1.
            x1, x2, x3 = x_aux[0], x_aux[1], x_aux[2]
            if i == j:
                dlngama[i,j] = -2*d[i] + 3*xd[i] + 3*gex
                dlngama[i,j] *= -x1*x2*x3
                lngama[i] += (2-3*x[i])*xd[j]
            else:
                x_auxf(x_aux, x_aux2, n)
                x_aux2[j] = 1.
                x12, x22, x32 = x_aux2[0], x_aux2[1], x_aux2[2]
                dlngama[i,j] =  xd[i] + (1 - 3*x[i])*(gex + xd[j])
                dlngama[i,j] *= x12*x22*x32
                lngama[i] -= (-1+3*x[i])*xd[j]
        lngama[i] *= x1*x2*x3
    return lngama.base, dlngama.base

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
def drk_cy(double [:] x, double [:, :] G, int_or_long [:,:] combinatory):

    cdef int ncomb, m, nc, i, j, k, l
    ncomb = combinatory.shape[0]
    m = G.shape[1]
    nc = x.shape[0]
    cdef double[:] dge = np.zeros(nc)
    cdef double[:, :] d2ge = np.zeros([nc,nc])
    cdef double SumA, SumB, SumC, SumD, SumE, SumF
    cdef double xi, xj, dx, ge, xixj, aux

    ge = 0.
    for k in range(ncomb):
        i = combinatory[k, 0]
        j = combinatory[k, 1]
        xi = x[i]
        xj = x[j]
        dx = xi - xj

        d2ge[i, j] += G[k, 0]

        SumE = 0.
        SumF = 0.

        SumA = 0.
        SumB = 0.
        if m > 1:
            for l in range(1, m):
                SumA += G[k,l]*dx**l
                SumB += G[k,l] *l*dx**(l-1.)
            SumE += G[k, 1]
            SumF += -G[k, 1]
            d2ge[i, j] += xi * G[k, 1]

        xixj = xi*xj
        aux = xixj*SumB
        ge +=  xixj*(SumA + G[k,0])
        dge[i] += xj*(SumA + G[k,0]) + aux
        dge[j] += xi*(SumA + G[k,0]) - aux

        SumE += SumB
        SumF += -SumB
        d2ge[i,j] += SumA  - xj * SumB
        SumC = 0.
        SumD = 0.
        if m > 2:
            for l in range(2, m):
                SumC += l * G[k,l] * (dx**(l-2.) *( 1-l) )
                SumD += G[k,l]*l*dx**(l-1.)
        SumE += -xi*SumC + SumD
        SumF += -xj*SumC - SumD
        d2ge[i,j] += xixj*SumC + xi * SumD

        d2ge[i, i] += xj * SumE
        d2ge[j, j] += xi * SumF
        d2ge[j, i] = d2ge[i, j]
    return ge, dge.base, d2ge.base


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
    cdef double SumA, SumB, SumC, xsum
    cdef double [:] lngama= np.zeros(nc)

    xsum = 0.
    for i in range(nc):
        xsum += X[i]

    for i in range(nc):
        SumA = 0.
        SumB = 0.
        for j in range(nc):
            SumA += X[j]*M[i,j]
            SumC = 0.
            for k in range(nc):
                SumC += X[k]*M[j,k]
            SumB += X[j]*M[j,i]/SumC
        lngama[i] = -log(SumA)+ xsum - SumB
    return lngama.base

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dwilson_cy(double [:] X, double [:,:] M):

    cdef int nc, i, j, k
    nc = X.shape[0]
    cdef double SumA, SumB, SumC, xsum
    cdef double [:] Si = np.zeros(nc)
    cdef double [:] lngama = np.zeros(nc)
    cdef double [:, :] dlngama = np.zeros([nc, nc])

    xsum = 0.
    for i in range(nc):
        xsum += X[i]
        for j in range(nc):
            Si[i] += X[j] * M[i, j]

    for i in range(nc):
        SumA = 0.
        for k in range(nc):
            SumA += X[k] * M[k, i] / Si[k]
        lngama[i] = xsum - log(Si[i]) - SumA

    for i in range(nc):
        for j in range(i, nc):
            SumB = 0.
            for k in range(nc):
                SumB += X[k] * M[k,i] * M[k, j] / Si[k]**2
            dlngama[i,j] = 1. -M[i,j] / Si[i] - M[j,i] / Si[j] + SumB
            dlngama[j,i] = dlngama[i,j]

    return lngama.base, dlngama.base
