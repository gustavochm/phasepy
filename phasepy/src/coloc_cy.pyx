from __future__ import division 
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def jcobi_roots(int n, int N0, int N1, int Al, int Be):
    
    cdef int i, j, k, Ab, Ad, Ap, nt
    cdef double z, zc, z1, y, x, xn, xn1, xd, xd1, xp, xp1
    cdef bint success
    #vectores derivadas y raices
    cdef double[:] dif1 = np.zeros(n)
    cdef double[:] dif2 = np.zeros(n)
    cdef double[:] root = np.zeros(n)
    
    #coeficientes polinomio Jacobi
    Ab = Al + Be
    Ad = Be - Al
    Ap = Be * Al
    
    dif1[0] =  (Ad/(Ab+2)+1)/2
    dif2[0] = 0.
    
    if not (n < 2):
        for i in range(1,n):
            z1 = i
            z = Ab + 2*z1
            dif1[i] = (Ab*Ad/z/(z+2)+1)/2
            if i == 1:
                dif2[i] = (Ab + Ap + z1)/z/z/(z+1)
            else:
                z **= 2
                y = z1 * (Ab + z1)
                y *= (Ap + y)
                dif2[i] = y / z / (z-1)
        
    #determinacion de las raices del polinomio de Jacobi
    
    x = 0.0
    for i in range(n):
        success = False
        while not success:
            xd = 0.0 
            xn = 1.0
            xd1 = 0.0
            xn1 = 0.0
            for j in range(n):
                xp = (dif1[j]-x)*xn - dif2[j]*xd
                xp1 = (dif1[j]-x)*xn1 - dif2[j]*xd1-xn
                xd = xn
                xd1 =  xn1
                xn = xp
                xn1 = xp1
            zc = 1.0
            z = xn/xn1
            if not (i == 0):
                for j in range(1,i+1):
                    zc -=  z / (x - root[j-1])
            z /= zc
            x -= z
            success = (abs(z) < 1e-9)
        root[i] = x
        x += 0.0001
        
    if N0 == 1 and N1 == 1:
        root = np.hstack([0,root,1])
    elif N0 == 1:
        root = np.hstack([0,root])
    elif N1 == 1:
        root = np.hstack([root,1])
    
    nt = n + N0 + N1
    dif1 = np.ones(nt)
    
    for k in range(nt):
        x=root[k]
        for j in range(nt):
            if j!=k:
                y = x-root[j]
                dif1[k] = y*dif1[k]
    
    return root.base, dif1.base


@cython.boundscheck(False)
@cython.wraparound(False)
def colocAB(double [:] roots):
    cdef int n, k, j, i 
    cdef double x, y, gen
    n = roots.shape[0]
    cdef double [:] dif1 = np.ones(n)
    cdef double [:] dif2 = np.zeros(n)
    cdef double [:] dif3 = np.zeros(n)
    cdef double [:,:] B = np.zeros([n,n])
    cdef double [:,:] A = np.zeros([n,n])
    for k in range(n):
        x=roots[k]
        for j in range(n):
            if j!=k:
                y=x-roots[j]
                dif3[k]=y*dif3[k]+3.*dif2[k]
                dif2[k]=y*dif2[k]+2.*dif1[k]
                dif1[k]=y*dif1[k]
    for i in range(n):
        for j in range(n):
            if j!=i:
                y=roots[i]-roots[j]
                gen=dif1[i]/dif1[j]/y
                A[i,j]=gen
                B[i,j]=gen*(dif2[i]/dif1[i]-2/y)
            else:
                A[i,j]=dif2[i]/dif1[i]/2
                B[i,j]=dif3[i]/dif1[i]/3
    return A.base, B.base

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def colocA(double [:] roots):
    cdef int n, k, j, i 
    cdef double x, y, gen
    n = roots.shape[0]
    cdef double [:] dif1 = np.ones(n)
    cdef double [:] dif2 = np.zeros(n)
    cdef double [:,:] A = np.zeros([n,n])
    for k in range(n):
        x=roots[k]
        for j in range(n):
            if j!=k:
                y=x-roots[j]
                dif2[k]=y*dif2[k]+2.*dif1[k]
                dif1[k]=y*dif1[k]
    for i in range(n):
        for j in range(n):
            if j!=i:
                y=roots[i]-roots[j]
                gen=dif1[i]/dif1[j]/y
                A[i,j]=gen
            else:
                A[i,j]=dif2[i]/dif1[i]/2
    return A.base

@cython.boundscheck(False)
@cython.wraparound(False)
def colocB(double [:] roots):
    cdef int n, k, j, i 
    cdef double x, y, gen
    n = roots.shape[0]
    cdef double [:] dif1 = np.ones(n)
    cdef double [:] dif2 = np.zeros(n)
    cdef double [:] dif3 = np.zeros(n)
    cdef double [:,:] B = np.zeros([n,n])
    for k in range(n):
        x=roots[k]
        for j in range(n):
            if j!=k:
                y=x-roots[j]
                dif3[k]=y*dif3[k]+3.*dif2[k]
                dif2[k]=y*dif2[k]+2.*dif1[k]
                dif1[k]=y*dif1[k]
    for i in range(n):
        for j in range(n):
            if j!=i:
                y=roots[i]-roots[j]
                gen=dif1[i]/dif1[j]/y
                B[i,j]=gen*(dif2[i]/dif1[i]-2/y)
            else:
                B[i,j]=dif3[i]/dif1[i]/3
    return B.base
