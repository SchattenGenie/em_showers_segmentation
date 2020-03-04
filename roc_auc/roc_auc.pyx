cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fabs, log, abs
np.import_array()

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def fast_auc(np.ndarray[double] y_true, np.ndarray[double] y_prob):
    cdef int i
    cdef int N = len(y_true)
    y_true = y_true[np.argsort(y_prob)]
    cdef double nfalse = 0.
    cdef double auc = 0.
    cdef double y_i
    for i in range(N):
        nfalse += (1 - y_true[i])
        auc +=  y_true[i] * nfalse
    auc /= (nfalse * (N - nfalse))
    return auc
