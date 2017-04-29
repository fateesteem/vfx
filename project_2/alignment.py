import numpy as np
from random import shuffle
from data_helper import Load_Data

P = 0.99


class SolveTransform:
    def __init__(self, threshold):
        self.matrix = None
        self.threshold = threshold

    def solve(self, v, v_prime):
        assert v.shape[0] == v_prime.shape[0]
        assert v.shape[1] == 2 and v_prime.shape[1] == 2
        num = v.shape[0]
        A = np.zeros((num*2, 6), dtype='float')
        A[np.arange(num)*2, :2] = v
        A[np.arange(num)*2+1 , 3:5] = v
        A[range(num*2), [2, 5]*num] = 1.
        b = v_prime.reshape(-1)
        x = np.matmul(np.linalg.pinv(A), b)
        self.matrix = x.reshape(2, 3)
        return self.matrix

    def count(self, v, v_prime):
        if self.matrix == None:
            raise Exception("Transform matrix has not been solved!")
        assert v.shape[0] == v_prime.shape[0]
        assert v.shape[1] == 2 and v_prime.shape[1] == 2
        threshold = self.threshold
        num = v.shape[0]
        add1_v = np.zeros((num, 3), dtype='float')
        add1_v[:, :2] = v
        add1_v[:, 2] = 1.
        v_prime_calc = (self.matrix @ add1_v.transpose()).transpose()
        dists = (np.sum((v_prime - v_prime_calc)**2, axis=1))**0.5
        return np.sum(dists < threshold), np.mean(dists[dists < threshold])

    def getMatrix(self):
        if self.matrix == None:
            raise Exception("Transform matrix has not been solved!")
        return self.matrix



def RANSAC(v, v_prime, n, p, solver):
    """
    Implementation of RANSAC Algorithm
    Args:
      v:        Location vectors of standard coordinate (x, y), numpy array of shape [N, 2].
      v_prime:  Location vectors of transformed coordinate, numpy array of shape [N, 2].
      n:        The number of sample to draw at each run.
      p:        Probability of real inliers.
      solver:   The solver instance with method "solve" to solve the transform matrix, and method "count" to count inliers.
    Returns
      res_matrix: result transform matrix.
    """
    assert v.shape[0] == v_prime.shape[0] and v.shape[0] > n
    assert v.shape[1] == 2 and v_prime.shape[1] == 2
    N = v.shape[0]
    k = np.log(1 - P) / np.log(1 - p**n)
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
    return res_matrix


if __name__ == "__main__":
    imgs, fs = Load_Data('parrington', 'parrington/f.txt', '.jpg')
    cv2.imshow('old', imgs[0][:, :, :])
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
