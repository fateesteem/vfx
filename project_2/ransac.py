import numpy as np
from random import shuffle


P = 0.99

def RANSAC(v, v_prime, n, p, solver):
    """
    Implementation of RANSAC Algorithm
    Args:
      v:        Location vectors of standard coordinate, numpy array of shape [N, D].
      v_prime:  Location vectors of transformed coordinate, numpy array of shape [N, D].
      n:        The number of sample to draw at each run.
      p:        Probability of real inliers.
      solver:   The solver instance with method "solve" to solve the transform matrix, method "count" to count inliers,
                and method "getMatrix" to retrieve solved transform matrix.
    Returns
      res_matrix: result transform matrix.
    """
    assert v.shape[0] == v_prime.shape[0] and v.shape[0] > n
    N = v.shape[0]
    k = int(np.log(1 - P) / np.log(1 - p**n)) + 1
    max_cnt = -1
    min_dist = np.inf
    res_matrix = None
    for run in range(k):
        # sampling #
        choice = [True,]*n + [False,]*(N - n)
        shuffle(choice)
        not_choice = np.logical_not(choice)
        sample_v = v[choice]
        sample_v_prime = v_prime[choice]
        # solve transform matrix #
        solver.solve(sample_v, sample_v_prime)
        # count inliers #
        cnt, dist = solver.count(v[not_choice], v_prime[not_choice])
        if cnt > max_cnt or (cnt == max_cnt and dist < min_dist):
            max_cnt = cnt
            min_dist = dist
            res_matrix = solver.getMatrix()
    print("Matrix:    %12.6f%12.6f%12.6f" % tuple(res_matrix[0, :]))
    print("           %12.6f%12.6f%12.6f" % tuple(res_matrix[1, :]))
    print("mean_dist: %12.6f" % min_dist)
    print("inliers:   %d/%d" % (max_cnt, N))
    return res_matrix


