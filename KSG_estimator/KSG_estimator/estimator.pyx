import numpy as np
cimport numpy as np
cimport cython
from cython cimport double, size_t

from libc.math cimport log, fabs
from libc.stdlib cimport rand


def cythonKSG(
    np.ndarray[double, ndim=2, mode='c'] x not None, 
    np.ndarray[double, ndim=2, mode='c'] y not None, k=5, eps=1e-16):
    assert x.shape[0] == y.shape[0]
    cdef np.ndarray[double, ndim=1, mode='c'] x_ij = np.empty(x.shape[0])
    cdef np.ndarray[double, ndim=1, mode='c'] y_ij = np.empty(x.shape[0])
    cdef np.ndarray[double, ndim=1, mode='c'] buff = np.empty(x.shape[0])
    cdef double I
    return _mi(x, y, k, eps, x_ij, y_ij, buff) 
    
#
# estimate I(x;y) where x,y are scalar/vector
#
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double _mi(double[:,:] x,        # input data
                double[:,:] y, 
                int k,              # parameters
                double eps,
                double[:] x_ij,     # temp buffers
                double[:] y_ij, 
                double[:] buff) nogil:
    cdef double I = 0
    cdef double e_i, r, k_i, nx, ny
    cdef int N = <int>x.shape[0]    
    cdef int i
    for i in range(N):
        # compute differences: x_ij, y_ij, z_ij
        _1norm_vector(x, x_ij, i)
        _1norm_vector(y, y_ij, i)
        # compute d_ij
        _hmax2(x_ij, y_ij, buff)
        # compute e_i, k_i, r
        e_i = _quickselect(buff, 0, N-1, k)
        if e_i < eps:
            k_i = _count_compare(buff, eps) 
            nx  = _count_compare(x_ij, eps)
            ny  = _count_compare(y_ij, eps)
        else:
            k_i = k
            nx = _count_compare(x_ij, fabs(e_i + eps))
            ny = _count_compare(y_ij, fabs(e_i + eps))
        # update estimate
        I += _digamma(k_i) - _digamma(nx) - _digamma(ny) + _digamma(N)
    return I / <double>N

#
# digamma function
#
@cython.cdivision(True)
cdef inline double _digamma(double x) nogil:
    cdef double r, f, t
    r = 0
    while (x <= 5):
        r -= 1/x
        x += 1
    f = 1.0/(x*x)
    t = f*(-1.0/12.0 + 
           f*(1.0/120.0 + 
              f*(-1.0/252.0 + 
                 f*(1.0/240.0 + 
                    f*(-1.0/132.0 + 
                       f*(691.0/32760.0 + 
                          f*(-1.0/12.0 + 
                             f*3617.0/8160.0)))))))
    return r + log(x) - 0.5/x + t

#
# return pairwise distances between vectors
#
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _1norm_vector(double[:,:] x, double[:] out, int diff_row) nogil:
    cdef double row
    cdef int i,j
    for i in range(x.shape[0]):
        row = 0
        for j in range(x.shape[1]):
            row += fabs(x[i,j] - x[diff_row, j])
        out[i] = row

#
# count number of entries below a threshold
#
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int _count_compare(double[:] x, double thresh_high) nogil:
    cdef int count = 0
    cdef int i
    for i in range(x.shape[0]):
        if x[i] < thresh_high:
            count += 1
    return count

#
# Take elementwise maximum of three arrays
#
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _hmax3(double[:] xd, double[:] yd, double[:] zd, double[:] out) nogil:
    cdef int i
    cdef double temp_max
    for i in range(out.shape[0]):
        temp_max = xd[i]
        if yd[i] > temp_max:
            temp_max = yd[i]
        if zd[i] > temp_max:
            temp_max = zd[i]
        out[i] = temp_max

#
# Take elementwise maximum of two arrays
#
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _hmax2(double[:] xd, double[:] yd, double[:] out) nogil:
    cdef int i
    for i in range(out.shape[0]):
        out[i] = xd[i] if xd[i] > yd[i] else yd[i]

#
# quickselect: return kth smallest element within x[left:right]
#
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double _quickselect(double[:] x, int left, int right, int k) nogil:
    if left == right:
        return x[left]
    cdef int pivot = left + (rand() % (right - left + 1))
    pivot = _partition(x, left, right, pivot)
    if pivot == k:
        return x[k]
    elif pivot > k:
        return _quickselect(x, left, pivot - 1, k)
    else:
        return _quickselect(x, pivot + 1, right, k)
    
#
# partition x[left:right] about pivot for quickselect
#
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int _partition(double[:] x, int left, int right, int pivot) nogil:
    cdef int i
    cdef int storeIndex = left
    cdef double swap_val
    cdef double pivotVal = x[pivot]
    x[pivot] = x[right]
    x[right] = pivotVal
    for i in range(left, right):
        if x[i] < pivotVal:
            swap_val = x[storeIndex]
            x[storeIndex] = x[i]
            x[i] = swap_val
            storeIndex += 1
    swap_val = x[right]
    x[right] = x[storeIndex]
    x[storeIndex] = swap_val
    return storeIndex
