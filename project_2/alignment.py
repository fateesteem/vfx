import numpy as np
from random import shuffle
from data_helper import Load_Data
import matplotlib.pyplot as plt

P = 0.99


class SolveLine:
    def __init__(self, threshold):
        self.matrix = None
        self.solved = False
        self.threshold = threshold

    def solve(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == 1 and y.shape[1] == 1
        num = x.shape[0]
        A = np.zeros((num, 2), dtype='float')
        A[:, 0] = x[:, 0]
        A[:, 1] = 1.
        b = y
        x = np.matmul(np.linalg.pinv(A), b)
        self.matrix = x
        self.solved = True

    def count(self, x, y):
        if not self.solved:
            raise Exception("Transform matrix has not been solved!")
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == 1 and y.shape[1] == 1
        threshold = self.threshold
        num = x.shape[0]
        add1_x = np.zeros((num, 2), dtype='float')
        add1_x[:, 0] = x[:, 0]
        add1_x[:, 1] = 1.
        y_calc = add1_x @ self.matrix
        dists = (np.sum((y - y_calc)**2, axis=1))**0.5
        is_inlier = dists < threshold
        if np.any(is_inlier):
            return np.sum(is_inlier), np.mean(dists[is_inlier])
        else:
            return np.sum(is_inlier), np.inf

    def getMatrix(self):
        if not self.solved:
            raise Exception("Transform matrix has not been solved!")
        return self.matrix


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
        v_prime_calc = add1_v @ self.matrix.transpose()
        dists = (np.sum((v_prime - v_prime_calc)**2, axis=1))**0.5
        is_inlier = dists < threshold
        if np.any(is_inlier):
            return np.sum(is_inlier), np.mean(dists[is_inlier])
        else:
            return np.sum(is_inlier), np.inf

    def getMatrix(self):
        if self.matrix == None:
            raise Exception("Transform matrix has not been solved!")
        return self.matrix


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
    return res_matrix


if __name__ == "__main__":
    """
    imgs, fs = Load_Data('./parrington', './parrington/f.txt', '.jpg')
    cv2.imshow('old', imgs[0][:, :, :])
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
    """
    x = (np.arange(40).reshape(-1, 1)+1) / 4 + np.random.normal(loc=0.0, scale=0.25, size=(40, 1))
    y = 2.*x + 1. + np.random.normal(loc=0.0, scale=0.25, size=(40, 1))
    rand_x = np.random.random_sample((100, 1)) * 10
    rand_y = np.random.random_sample((100, 1)) * 20
    line_solver = SolveLine(threshold=0.1)
    line = RANSAC(np.concatenate([x, rand_x], axis=0), np.concatenate([y, rand_y], axis=0), 2, 40/130., line_solver)
    plt.plot(x.flatten(), y.flatten(), 'ro', ms=3)
    plt.plot(rand_x.flatten(), rand_y.flatten(), 'ro', ms=3)
    plt.plot([0, 10], np.array([0, 10])*line[0, 0] + line[1, 0], 'b-')
    plt.axis([0, 15, 0, 25])
    plt.show()
