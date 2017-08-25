import numpy as np
cimport numpy as np

cdef extern from 'wrapped_code_1.h':
    void autofunc(double a, double b, double c, double *out_1451769269)

def autofunc_c(double a, double b, double c):

    cdef np.ndarray[np.double_t, ndim=2] out_1451769269 = np.empty((2,1))
    autofunc(a, b, c, <double*> out_1451769269.data)
    return out_1451769269