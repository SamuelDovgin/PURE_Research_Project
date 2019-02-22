import numpy as np
import matplotlib.pyplot as plt
import time

from KSG_estimator.estimator import cythonKSG
from KSG_estimator.pythonKSG import pythonKSG
from KDTreePythonKSG import kraskov_mi

# identity matrix with d-dimensions
# create a symmetry matrix


# X and Y correlated zero mean, unit variance Gaussians
def gen_samples(N, correlation_coeff):
    cov = [[1,correlation_coeff],
            [correlation_coeff,1]]
    X,Y = np.random.multivariate_normal([0,0], cov, size=N).T

    # reshape to c-type array for cython
    X = np.array(X.reshape(X.shape[0],-1), order="C")
    #print(X)
    Y = np.array(Y.reshape(Y.shape[0],-1), order="C")
    #print(Y)
    return X,Y

# analytic value of mutual information
def analytic_MI(correlation_coeff):
    from math import log
    return -0.5 * log(1-correlation_coeff*correlation_coeff)


if __name__ == "__main__":
    correlation_coeff = .3
    sample_sizes = [80000,90000]

    # estimate MI using a certain sample size
    # replacing the pythonKSG estimator with the cythonKSG estimator
    # speeds things up, even though both are brute force. Cython is fast.
    MI_cython = list()
    for N in sample_sizes:
        X,Y = gen_samples(N, correlation_coeff)

        start_time = time.time()
        #est_MI = cythonKSG(X, Y, k=5)
        est_MI = kraskov_mi(X, Y, k=5)
        end_time = time.time()-start_time
        print ("Estimated MI(KD-Tree method): %f using %d samples. Time taken: %f seconds" % (est_MI, N, end_time))
        MI_cython.append(est_MI)

        start_time2 = time.time()
        est_MI = cythonKSG(X, Y, k=5)
        end_time2 = time.time() - start_time2
        print("Cython brute force: %f using %d samples. Time taken: %f seconds" % (est_MI, N, end_time2))
        #MI_cython.append(est_MI)

        times_different = end_time2/end_time
        print("KD tree method is " + str(times_different) + " times faster")

    # compute ground truth
    true_MI = analytic_MI(correlation_coeff)

    # plot results. You can see that as sample size increases, the estimate approaches the true value
    plt.plot(sample_sizes, MI_cython, 'o-', label='Estimated MI')
    plt.plot(sample_sizes, true_MI*np.ones(len(sample_sizes)), label='Analytic MI')
    plt.title('Estimated Mutual Information')
    plt.xlabel('Number of Samples')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.show()
