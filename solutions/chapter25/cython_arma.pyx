#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def arma(double[::1] parameters,
         double[::1] data,
         int p=0,
         int q=0):

    cdef size_t tau, t, i
    tau = data.shape[0]

    cdef double[::1] errors = np.zeros(tau)

    for t in range(p, tau):
        errors[t] = data[t] - parameters[0]
        for i in range(p):
            errors[t] -= parameters[i+1] * data[t-i-1]
        for i in range(q):
            if (t-i) >= 0:
                errors[t] -= parameters[i+p+1] * errors[t-i-1]

    return np.asarray(errors)
