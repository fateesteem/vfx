import numpy as np
import math
from alignment_helper import SolveAffine, GridAffineTransform
from data_helper import Load_Data
from ransac import RANSAC
from feature_matching import genMatchPairs
from MOPS import Build_pyramid
from cylindrical_proj import Interp
from poisson_blending import PoissonBlending

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

def ImageStitching(imgs_proj, imgs_proj_mask, btype = 'Linear'):
    matrix = np.eye(3)
    coords_y = [None]*len(imgs_proj)
    coords_x = [None]*len(imgs_proj)
    min_y = 0
    min_x = 0
    max_y = 0
    max_x = 0
    tmp_imgs = [None]*len(imgs_proj)
    images_mask = [None]*len(imgs_proj)
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
    new_H = math.ceil(max_y - min_y + 1)
    new_W = math.ceil(max_x - min_x + 1)
    stitch_img = np.zeros((new_H, new_W, 3))
    prev_shift_y = coords_y[0] - min_y
    prev_shift_x = coords_x[0] - min_x
    prev_img, prev_mask = Interp(prev_shift_y, prev_shift_x, imgs_proj[0], new_H, new_W, Ch=3, mask=imgs_proj_mask[0])
    tmp_imgs[0] = prev_img
    images_mask[0] = prev_mask
    prev_w = np.ones((new_H, new_W), dtype='float')
    for i in range(len(imgs_proj) - 1):
        shift_y = coords_y[i+1] - min_y
        shift_x = coords_x[i+1] - min_x
        img, mask = Interp(shift_y, shift_x, imgs_proj[i+1], new_H, new_W, Ch=3, mask=imgs_proj_mask[i+1])
        w_im1, w = BlendingWeights(prev_mask, mask, new_H, new_W)
        stitch_img += prev_img * (prev_w + w_im1 - 1.)[:, :, None]
        prev_shift_y = shift_y
        prev_shift_x = shift_x
        prev_img = img
        tmp_imgs[i+1] = img
        prev_mask = mask
        images_mask[i+1] = mask
        prev_w = w
    stitch_img += prev_img * (prev_w)[:, :, None]
    if btype == 'Poisson':
        stitch_img = PoissonBlending(stitch_img, images_mask, tmp_imgs)

    return stitch_img.astype('uint8')
