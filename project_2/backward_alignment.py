import numpy as np
import math
from alignment_helper import SolveAffine, SolveTranslation, GridAffineTransform, drift_adjustment
from data_helper import Load_Data
from ransac import RANSAC
from feature_matching import genMatchPairs
from MOPS import Build_pyramid
from inverse_cylindrical_proj import BiInterpn
from numpy.linalg import inv
from poisson_blending import BlendingWeights, PoissonBlending
import cv2

def ImageStitching(imgs_proj, imgs_proj_mask, btype = 'Linear', solver = 'Affine', drift = False):
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
            feats0 = Build_pyramid(imgs_proj_mask[i], imgs_proj[i])
            feats1 = Build_pyramid(imgs_proj_mask[i+1], imgs_proj[i+1])
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
                affine_solver = SolveAffine(threshold=1.0)
            elif solver == 'Translation':
                affine_solver = SolveTranslation(threshold=1.0)
            M = RANSAC(v, v_prime, 5, 0.3, affine_solver)
            matrix = matrix @ np.append(M, [[0, 0, 1]], axis=0)
            matrices[i] = matrix

            ## coord transform  ##
            y0, x0 = GridAffineTransform(H, W, matrix[:2, :])
            min_y = min(min_y, np.min(y0))
            min_x = min(min_x, np.min(x0))
            max_y = max(max_y, np.max(y0))
            max_x = max(max_x, np.max(x0))
    ## estimates new image ##
    new_H = math.ceil(max_y - min_y + 1)
    new_W = math.ceil(max_x - min_x + 1)
    stitch_img = np.zeros((new_H, new_W, 3))
    i_prev_w = None
    weight_acc = np.zeros((new_H, new_W))
    for i in range(len(imgs_proj)):
        print('linear blending on {}'.format(i))
        if i != len(imgs_proj) - 1:
            inv_matrix = inv(matrices[i]) # get inverse affine
            #y_new, x_new = np.mgrid[range(new_H), range(new_W)] # new grid axis
            #y_new += int(min_y)
            #x_new += int(min_x)
            y_org, x_org = GridAffineTransform(new_H, new_W, inv_matrix[:2, :], shift_x=int(min_x), shift_y=int(min_y))# original coordinate
        else:
            y_org, x_org = np.mgrid[range(new_H), range(new_W)]
            y_org += int(min_y)
            x_org += int(min_x)
        tmp_imgs[i], images_mask[i] = BiInterpn(x_org, y_org, imgs_proj[i], new_H, new_W, 3, imgs_proj_mask[i])        
         
        ## blending with mask ##

        if i > 0:
            i_w, ip1_w = BlendingWeights(images_mask[i - 1], images_mask[i], new_H)
            stitch_img[images_mask[i-1], :] += (i_prev_w + i_w )[images_mask[i-1], None] \
                                                * tmp_imgs[i-1][images_mask[i-1], :]
            weight_acc[images_mask[i-1]] += (i_prev_w + i_w )[images_mask[i-1]]
            i_prev_w = ip1_w
        else:
            i_prev_w = np.zeros_like(images_mask[i], dtype = 'float')
            i_prev_w[images_mask[i]] = 1. 
    stitch_img[images_mask[-1], :] += (i_prev_w)[images_mask[-1], None] * tmp_imgs[-1][images_mask[-1], :]
    weight_acc[images_mask[-1]] += (i_prev_w)[images_mask[-1]]
    del i_prev_w
    stitch_img = (stitch_img / (weight_acc[:, :, None] + 1.0e-10)).astype('uint8')
    #stitch_img[images_mask[1], :] += (i_prev_w)[images_mask[1], None] * tmp_imgs[1][images_mask[1], :]
    if btype == 'Poisson':
        stitch_img = PoissonBlending(stitch_img, images_mask, tmp_imgs)
    stitch_img_mask = np.zeros((new_H, new_W), dtype = bool)
    if drift:
        for mask in images_mask:
            stitch_img_mask |= mask
        stitch_img = drift_adjustment(imgs_proj_mask[-1], imgs_proj_mask[0], imgs_proj[-1], imgs_proj[0], matrix, stitch_img, stitch_img_mask, solver)
    return stitch_img.astype('uint8')
