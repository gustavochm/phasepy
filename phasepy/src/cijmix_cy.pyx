import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cmix_cy(double [:,:] drodz, double [:, :] cij):
    
    cdef int n, nc, i, j, k
    nc = drodz.shape[0]
    n = drodz.shape[1]
    cdef double[:] suma = np.zeros(n)

    for k in range(n):
        for i in range(nc):
            for j in range(nc):
                suma[k] += cij[i,j]*drodz[i,k]*drodz[j,k]
    return suma.base