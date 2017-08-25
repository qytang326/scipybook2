import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def pairwise_dist_cython2(double[:, ::1] X):
    cdef int m, n, i, j, k
    cdef double tmp, d
    m, n = X.shape[0], X.shape[1]    
    cdef double[:, ::1] D = np.empty((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            d = 0.0
            for k in range(n):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = D[j, i] = sqrt(d)
    return np.asarray(D)