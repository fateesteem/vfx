import numpy as np
from numpy.linalg import inv
import cv2
from data_helper import Load_Data
from ransac import RANSAC
from feature_matching import genMatchPairs
from MOPS import Build_pyramid
from cylindrical_proj import inverse_cylindrical_projection, BiInterpn
from poisson_blending import PoissonBlending
import matplotlib.pyplot as plt


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
        A[np.arange(num)*2+1 , 3:5] = v
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
    coords_prime = AffineTransform(coords, matrix).astype('int')
    y = coords_prime[:, 1].reshape(H, W)
    x = coords_prime[:, 0].reshape(H, W)
    return y, x
"""

def BlendingWeights(y0, x0, mask0, y1, x1, mask1, H):
    if np.min(x0) < np.min(x1):
        y_left = y0
        x_left = x0
        mask_left = mask0
        y_right = y1
        x_right = x1
        mask_right = mask1
        left_is_left = True
    else:
        y_left = y1
        x_left = x1
        mask_left = mask1
        y_right = y0
        x_right = x0
        mask_right = mask0
        left_is_left = False
    left_weights = np.ones_like(y_left, dtype='float')
    right_weights = np.ones_like(y_right, dtype='float')
    for j in range(H):
        #jx_left = (y_left == j)
        jx_left = np.logical_and(y_left == j, mask_left)
        if np.sum(jx_left) == 0:
            continue
        right_border = np.max(x_left[jx_left])
        #jx_right = (y_right == j)
        jx_right = np.logical_and(y_right == j, mask_right)
        if np.sum(jx_right) == 0:
            continue
        left_border = np.min(x_right[jx_right])
        if left_border > right_border:
            continue
        else:
            overlap_left = np.logical_and(jx_left, (x_left >= left_border))
            overlap_right = np.logical_and(jx_right, (x_right <= right_border))
            if left_border < right_border:
                overlap_len = right_border - left_border
                left_weights[overlap_left] = (-1. / overlap_len * (x_left - right_border))[overlap_left]
                right_weights[overlap_right] = (1. / overlap_len * (x_right - left_border))[overlap_right]
            else:
                left_weights[overlap_left] = 0.5
                right_weights[overlap_right] = 0.5
    if left_is_left:
        return left_weights, right_weights
    else:
        return right_weights, left_weights


def ImageStitching(imgs_proj, imgs_proj_mask):
    matrix = np.eye(3)
    coords_y = [None]*len(imgs_proj)
    coords_x = [None]*len(imgs_proj)
    min_y = 0
    min_x = 0
    max_y = 0
    max_x = 0
    for i in range(len(imgs_proj)-1, -1, -1):
        H = imgs_proj[i].shape[0]
        W = imgs_proj[i].shape[1]
        if i == len(imgs_proj) - 1:
            coords_y[i], coords_x[i] = np.mgrid[range(H), range(W)]
            max_y = H - 1
            max_x = W - 1
        else:
            ## MOPS features ##
            feats0 = Build_pyramid(imgs_proj[i])
            feats1 = Build_pyramid(imgs_proj[i+1])
            v0 = np.array([[feat.x, feat.y] for feat in feats0])
            v1 = np.array([[feat.x, feat.y] for feat in feats1])
            des0 = np.array([feat.descriptor for feat in feats0])
            des1 = np.array([feat.descriptor for feat in feats1])

            ## self implemented matching ##
            id_pairs = genMatchPairs(des0, des1, k=2, p=2)
            
            ## calc transform matrix ##
            v = v0[id_pairs[:, 0]]
            v_prime = v1[id_pairs[:, 1]]
            affine_solver = SolveAffine(threshold=0.1)
            M = RANSAC(v, v_prime, 6, 0.3, affine_solver)
            matrix = matrix @ np.append(M, [[0, 0, 1]], axis=0)

            ## coord transform  ##
            y0, x0 = GridAffineTransform(H, W, matrix[:2, :])
            coords_y[i] = y0
            coords_x[i] = x0
            min_y = min(min_y, np.min(y0))
            min_x = min(min_x, np.min(x0))
            max_y = max(max_y, np.max(y0))
            max_x = max(max_x, np.max(x0))
    new_H = max_y - min_y + 1
    new_W = max_x - min_x + 1
    stitch_img = np.zeros((new_H, new_W, 3))
    shift_y = [coords_y[0] - min_y]
    shift_x = [coords_x[0] - min_x]
    i_prev_w = np.ones_like(shift_y[0], dtype='float')
    for i in range(len(imgs_proj) - 1):
        shift_y.append(coords_y[i+1] - min_y)
        shift_x.append(coords_x[i+1] - min_x)
        i_w, ip1_w = BlendingWeights(shift_y[i],
                                     shift_x[i],
                                     imgs_proj_mask[i],
                                     shift_y[i+1],
                                     shift_x[i+1],
                                     imgs_proj_mask[i+1],
                                     new_H)
        stitch_img[shift_y[i], shift_x[i], :] += imgs_proj[i] * (i_prev_w + i_w - 1.)[:, :, None]
        i_prev_w = ip1_w
    stitch_img[shift_y[-1], shift_x[-1], :] += imgs_proj[-1] * (i_prev_w)[:, :, None]
    return stitch_img.astype('uint8')
"""

def ImageStitching(imgs_proj, imgs_proj_mask):
    matrix = np.eye(3)
    coords_y = [None]*len(imgs_proj)
    coords_x = [None]*len(imgs_proj)
    min_y = 0
    min_x = 0
    max_y = 0
    max_x = 0
    matrices = [None] * (len(imgs_proj) - 1)
    images_mask = [None] * (len(imgs_proj))
    tmp_imgs = [None] * (len(imgs_proj))
    for i in range(len(imgs_proj)-1, -1, -1):
        H = imgs_proj[i].shape[0]
        W = imgs_proj[i].shape[1]
        if i == len(imgs_proj) - 1:
            max_y = H - 1
            max_x = W - 1
        else:
            ## MOPS features ##
            feats0 = Build_pyramid(imgs_proj[i])
            feats1 = Build_pyramid(imgs_proj[i+1])
            v0 = np.array([[feat.x, feat.y] for feat in feats0])
            v1 = np.array([[feat.x, feat.y] for feat in feats1])
            des0 = np.array([feat.descriptor for feat in feats0])
            des1 = np.array([feat.descriptor for feat in feats1])

            ## self implemented matching ##
            id_pairs = genMatchPairs(des0, des1, k=2, p=2)
            
            ## calc transform matrix ##
            v = v0[id_pairs[:, 0]]
            v_prime = v1[id_pairs[:, 1]]
            affine_solver = SolveAffine(threshold=0.1)
            M = RANSAC(v, v_prime, 6, 0.3, affine_solver)
            matrix = matrix @ np.append(M, [[0, 0, 1]], axis=0)
            matrices[i] = matrix

            ## coord transform  ##
            y0, x0 = GridAffineTransform(H, W, matrix[:2, :])
            min_y = min(min_y, np.min(y0))
            min_x = min(min_x, np.min(x0))
            max_y = max(max_y, np.max(y0))
            max_x = max(max_x, np.max(x0))
    ## estimates new image ##
    new_H = max_y - min_y + 1
    new_W = max_x - min_x + 1
    stitch_img = np.zeros((new_H, new_W, 3))
    i_prev_w = None
    for i in range(len(imgs_proj)):
        if i != len(imgs_proj) - 1:
            inv_matrix = inv(matrices[i]) # get inverse affine
            y_new, x_new = np.mgrid[range(new_H), range(new_W)] # new grid axis
            y_org, x_org = GridAffineTransform(new_H, new_W, inv_matrix[:2, :])# original coordinate
        else:
            y_org, x_org = np.mgrid[range(new_H), range(new_W)]
        tmp_imgs[i], images_mask[i] = BiInterpn(x_org, y_org, imgs_proj[i], new_H, new_W, 3, imgs_proj_mask[i])        
        
        ## blending with mask ##

        if i > 0:
            i_w, ip1_w = BlendingWeights(tmp_imgs[i - 1], 
                                          images_mask[i - 1],
                                          tmp_imgs[i],
                                          images_mask[i],
                                          new_H)
            stitch_img[images_mask[i-1], :] += (i_prev_w + i_w - 1.)[images_mask[i-1], None] \
                                                * tmp_imgs[i-1][images_mask[i-1], :]
            i_prev_w = ip1_w
        else:
            i_prev_w = np.zeros_like(images_mask[i], dtype = 'float')
            i_prev_w[images_mask[i]] = 1. 
    stitch_img[images_mask[-1], :] += (i_prev_w)[images_mask[-1], None] * tmp_imgs[-1][images_mask[-1], :]
    #stitch_img[images_mask[1], :] += (i_prev_w)[images_mask[1], None] * tmp_imgs[1][images_mask[1], :]
    stitch_img = PoissonBlending(stitch_img, images_mask, tmp_imgs)
    
    return stitch_img.astype('uint8')

def BlendingWeights(img0, mask0, img1, mask1, H):
    if np.min(mask0.nonzero()) < np.min(mask1.nonzero()):
        img_left = img0
        mask_left = mask0
        img_right = img1
        mask_right = mask1
        left_is_left = True
    else:
        img_left = img1
        mask_left = mask1
        img_right = img0
        mask_right = mask0
        left_is_left = False
    left_weights = np.ones_like(mask_left, dtype='float')
    right_weights = np.ones_like(mask_right, dtype='float')
    blending_mask = (mask_left) & (mask_right)
    for j in range(H):
        j_mask = blending_mask[j, :]
        overlap_len = np.sum(j_mask)
        if np.sum(j_mask) == 0:
            continue
        right_border = np.max(j_mask.nonzero()[0])
        left_border = np.min(j_mask.nonzero()[0])
        #assert (right_border - left_border) == np.sum(j_mask) - 1:
        if left_border > right_border:
            continue
        else:
            if left_border < right_border:
                left_weights[j, j_mask] = (1. / (overlap_len - 1) * (np.arange(overlap_len, dtype = 'float'))[::-1])
                right_weights[j, j_mask] = (1. / (overlap_len - 1) * (np.arange(overlap_len, dtype = 'float')))
            else:
                left_weights[j, j_mask] = 0.5
                right_weights[j, j_mask] = 0.5
    if left_is_left:
        return left_weights, right_weights
    else:
        return right_weights, left_weights

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
