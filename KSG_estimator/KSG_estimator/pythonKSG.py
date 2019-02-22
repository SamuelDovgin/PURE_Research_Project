import numpy as np
from numpy.linalg import norm
from scipy.special import digamma


def pythonKSG(x, y, k=5, eps=1e-15):
    print(x)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    print(x)
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    N = len(x)

    I = 0
    for i in range(N):
        x_ij = norm(x-x[i], 1, axis=1).reshape(-1,1)
        print(x_ij)
        y_ij = norm(y-y[i], 1, axis=1).reshape(-1,1)
        print(y_ij)

        # creates one matrix where the larger x_ij or y_ij is added
        d_ij = np.max(np.concatenate(( x_ij, y_ij ), axis=1),axis=1)

        # larger elements are placed after index k (5) and smaller before
        # the first index is where it would be in the sorted array
        e_i = np.partition(d_ij, k)[k]

        # assume k_i is always k
        k_i = np.sum(d_ij < eps) if e_i == 0 else k

        r = np.abs(e_i + eps)
        nx = np.sum(np.max(x_ij,axis=1) < r)
        ny = np.sum(np.max(y_ij,axis=1) < r)

        I += digamma(k_i) - digamma(nx) - digamma(ny) + digamma(N)

    return I / N
