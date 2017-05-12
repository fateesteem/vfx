import math
import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
from MOPS import Build_pyramid
from feature_matching import genMatchPairs
from inverse_cylindrical_proj import BiInterpn
from ransac import RANSAC


def drift_adjustment(last_img, first_img, matrix, stitch_img, stitch_img_mask, solver):
    H, W = stitch_img.shape[:2]

    feats0 = Build_pyramid(last_img)
    feats1 = Build_pyramid(first_img)
    v0 = np.array([[feat.x, feat.y] for feat in feats0])
    v1 = np.array([[feat.x, feat.y] for feat in feats1])
    des0 = np.array([feat.descriptor for feat in feats0])
    des1 = np.array([feat.descriptor for feat in feats1])

    ## self implemented matching ##
    id_pairs = genMatchPairs(des0, des1, k=2, p=2)
    
    ## calc transform matrix ##
    v = v0[id_pairs[:, 0]]
    v_prime = v1[id_pairs[:, 1]]

    if solver == 'Affine':
        affine_solver = SolveAffine(threshold=0.1)
    elif solver == 'Translation':
        affine_solver = SolveTranslation(threshold=0.1)
    M = RANSAC(v, v_prime, 6, 0.3, affine_solver)
    matrix = matrix @ np.append(M, [[0, 0, 1]], axis=0)

    y_shift = matrix[1, 2]
    a = -1.0 * y_shift / W
    M = np.array([[1, 0, 0], [a, 1, 0], [0, 0, 1]], dtype = 'float')

    y0, x0 = GridAffineTransform(H, W, M[:2, :])
    min_y = np.min(y0)
    min_x = np.min(x0)
    max_y = np.max(y0)
    max_x = np.max(x0)
    ## estimates new image ##
    new_H = math.ceil(max_y - min_y + 1)
    new_W = math.ceil(max_x - min_x + 1)
    new_stitch_img = np.zeros((new_H, new_W, 3))
    y_org, x_org = GridAffineTransform(new_H, new_W, np.linalg.inv(M)[:2, :])
    new_stitch_img, mask = BiInterpn(x_org, y_org, stitch_img, new_H, new_W, 3, stitch_img_mask)
    ## crop ##
    
    y_coor, x_coor = np.mgrid[range(new_H), range(new_W)]
    x_min = np.amin(x_coor[mask])
    x_max = np.amax(x_coor[mask])

    y_min = np.amin(y_coor[mask])
    y_max = np.amax(y_coor[mask])
    stitch_img = new_stitch_img[y_min:y_max+1, x_min:x_max+1, :]
    #stitch_img = cv2.warpAffine(stitch_img, M, (W, H))
    return stitch_img

def AffineTransform(v, M):
    """
    Perform Affine transformation to vectors v by matrix M.
    Args:
      v:    The location vectors to be transformed. A numpy array of shape [N, D]
      M:    Affine transform matrix. A numpy array of shape [2, 3].

    Returns:
      Transformed vectors. A numpy array of shape same as v.
    """
    assert v.shape[1] == 2
    assert M.shape == (2, 3)
    num = v.shape[0]
    add1_v = np.zeros((num, 3), dtype='float')
    add1_v[:, :2] = v
    add1_v[:, 2] = 1.
    return add1_v @ M.transpose()


def AffineCompose(M1, M2):
    assert M1.shape == (2, 3)
    assert M2.shape == (2, 3)
    M1_3x3 = np.append(M1, [[0., 0., 1.]], axis=0)
    M2_3x3 = np.append(M2, [[0., 0., 1.]], axis=0)
    return (M1 @ M2)[:2, :]


class SolveLine:
    def __init__(self, threshold):
        self.matrix = None
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

    def count(self, x, y):
        if self.matrix is None:
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
        if self.matrix is None:
            raise Exception("Transform matrix has not been solved!")
        return self.matrix


class SolveTranslation:
    def __init__(self, threshold):
        self.matrix = None
        self.threshold = threshold

    def solve(self, v, v_prime):
        assert v.shape[0] == v_prime.shape[0]
        assert v.shape[1] == 2 and v_prime.shape[1] == 2
        num = v.shape[0]
        t = np.mean(v_prime - v, axis = 0)
        self.matrix = np.eye(3)
        self.matrix[:2, 2] = t
        self.matrix = self.matrix[:2, :3]

    def count(self, v, v_prime):
        if self.matrix is None:
            raise Exception("Transform matrix has not been solved!")
        assert v.shape[0] == v_prime.shape[0]
        assert v.shape[1] == 2 and v_prime.shape[1] == 2
        threshold = self.threshold
        v_prime_calc = AffineTransform(v, self.matrix)
        dists = (np.sum((v_prime - v_prime_calc)**2, axis=1))**0.5
        is_inlier = dists < threshold
        if np.any(is_inlier):
            return np.sum(is_inlier), np.mean(dists[is_inlier])
        else:
            return np.sum(is_inlier), np.inf

    def getMatrix(self):
        if self.matrix is None:
            raise Exception("Transform matrix has not been solved!")
        return self.matrix

class SolveAffine:
    def __init__(self, threshold):
        self.matrix = None
        self.threshold = threshold

    def solve(self, v, v_prime):
        assert v.shape[0] == v_prime.shape[0]
        assert v.shape[1] == 2 and v_prime.shape[1] == 2
        num = v.shape[0]
        A = np.zeros((num*2, 6), dtype='float')
        A[np.arange(num)*2, :2] = v
        A[np.arange(num)*2+1, 3:5] = v
        A[range(num*2), [2, 5]*num] = 1.
        b = v_prime.reshape(-1)
        x = np.matmul(np.linalg.pinv(A), b)
        self.matrix = x.reshape(2, 3)

    def count(self, v, v_prime):
        if self.matrix is None:
            raise Exception("Transform matrix has not been solved!")
        assert v.shape[0] == v_prime.shape[0]
        assert v.shape[1] == 2 and v_prime.shape[1] == 2
        threshold = self.threshold
        v_prime_calc = AffineTransform(v, self.matrix)
        dists = (np.sum((v_prime - v_prime_calc)**2, axis=1))**0.5
        is_inlier = dists < threshold
        if np.any(is_inlier):
            return np.sum(is_inlier), np.mean(dists[is_inlier])
        else:
            return np.sum(is_inlier), np.inf

    def getMatrix(self):
        if self.matrix is None:
            raise Exception("Transform matrix has not been solved!")
        return self.matrix


def GridAffineTransform(H, W, matrix):
    y, x = np.mgrid[range(H), range(W)]
    coords = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    # coords_prime = AffineTransform(coords, matrix).astype('int')
    coords_prime = AffineTransform(coords, matrix)
    y = coords_prime[:, 1].reshape(H, W)
    x = coords_prime[:, 0].reshape(H, W)
    return y, x

def BlendingWeights(mask0, mask1, H, W):
    assert mask0.shape == mask1.shape
    weight = np.tile(range(W), (H, 1)).astype('float')
    overlap = np.logical_and(mask0, mask1)
    x_min = np.argmax(overlap, axis=1)
    x_max = W - np.argmax(overlap[:, ::-1], axis=1) - 1
    max_eq_min = (x_max == x_min)
    length = x_max - x_min
    weight_left = -1. * (weight - (x_max + 0.5 * max_eq_min)[:, None]) / (length + max_eq_min)[:, None]
    weight_right = 1. - weight_left
    weight_left[np.logical_not(overlap)] = 1.
    weight_right[np.logical_not(overlap)] = 1.
    if np.min(np.where(mask0 == 1)[1]) < np.min(np.where(mask1 == 1)[1]):
        return weight_left, weight_right
    return weight_right, weight_left

if __name__ == "__main__":
    imgs, fs = Load_Data('./parrington', './parrington/f.txt', '.jpg')
    imgs_proj = []
    imgs_proj_mask = []
    for i in range(imgs.shape[0]):
        new_img, new_img_mask = inverse_cylindrical_projection(imgs[i], fs[i])
        imgs_proj.append(new_img)
        imgs_proj_mask.append(new_img_mask)
    """
    H0 = imgs_proj[0].shape[0]
    W0 = imgs_proj[0].shape[1]
    H1 = imgs_proj[1].shape[0]
    W1 = imgs_proj[1].shape[1]
    ## MOPS features ##
    feats0 = Build_pyramid(imgs_proj[0])
    feats1 = Build_pyramid(imgs_proj[1])
    v0 = np.array([[feat.x, feat.y] for feat in feats0])
    v1 = np.array([[feat.x, feat.y] for feat in feats1])
    des0 = np.array([feat.descriptor for feat in feats0])
    des1 = np.array([feat.descriptor for feat in feats1])

    ## self implemented matching ##
    id_pairs = genMatchPairs(des0, des1, k=2, p=2)
    my_img = np.zeros((H0, (W0+W1), 3), dtype='uint8')
    my_img[:, :W0, :] = imgs_proj[0]
    my_img[:, W0:, :] = imgs_proj[1]
    for feat in feats0:
        cv2.circle(my_img, (feat.x, feat.y), 3, (255,0,0), 2)
    for feat in feats1:
        cv2.circle(my_img, (feat.x+W0, feat.y), 3, (255,0,0), 2)
    for pair in id_pairs:
        pt0 = (feats0[pair[0]].x, feats0[pair[0]].y)
        pt1 = (feats1[pair[1]].x+W0, feats1[pair[1]].y)
        cv2.circle(my_img, pt0, 3, (0,255,0), 2)
        cv2.circle(my_img, pt1, 3, (0,255,0), 2)
        cv2.line(my_img,pt0,pt1,(0,255,0),1)

    v = v0[id_pairs[:, 0]]
    v_prime = v1[id_pairs[:, 1]]
    affine_solver = SolveAffine(threshold=0.1)
    M = RANSAC(v, v_prime, 6, 0.3, affine_solver)

    y0, x0 = GridAffineTransform(H0, W0, M)
    y1, x1 = np.mgrid[range(H1), range(W1)]
    shift_x = min(np.min(x0), 0)
    shift_y = min(np.min(y0), 0)
    y0 -= shift_y
    x0 -= shift_x
    y1 -= shift_y
    x1 -= shift_x
    new_H = max(np.max(y0), np.max(y1)) + 1
    new_W = max(np.max(x0), np.max(x1)) + 1
    stitch_img = np.zeros((new_H, new_W, 3), dtype='uint8')
    stitch_img[y0, x0, :] = imgs_proj[0]
    stitch_img[y1, x1, :] = imgs_proj[1]
    """
    stitch_img = ImageStitching(imgs_proj, imgs_proj_mask)

    cv2.imwrite('stitch.jpg', stitch_img)
    """
    cv2.imshow('stitch', stitch_img)
    cv2.imshow('features matching', my_img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
    """ 

    """ 2D Line test
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
    """
