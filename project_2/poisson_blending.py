import numpy as np
import scipy.sparse
import pyamg
import cv2
from feature_matching import genMatchPairs
from inverse_cylindrical_proj import inverse_cylindrical_projection, BiInterpn


def BlendingWeights(mask0, mask1, H):
    if np.min(mask0.nonzero()) < np.min(mask1.nonzero()):
        mask_left = mask0
        mask_right = mask1
        left_is_left = True
    else:
        mask_left = mask1
        mask_right = mask0
        left_is_left = False
    left_weights = np.ones_like(mask_left, dtype='float')
    right_weights = np.ones_like(mask_right, dtype='float')
    blending_mask = (mask_left) & (mask_right)
    for j in range(H):
        j_mask = blending_mask[j, :]
        if np.sum(j_mask) == 0:
            continue
        right_border = np.max(j_mask.nonzero()[0])
        left_border = np.min(j_mask.nonzero()[0])
        if left_border > right_border:
            continue
        else:
            overlap_len = right_border - left_border
            if left_border < right_border:
                left_weights[j, j_mask] = (-1. / (overlap_len ) * (j_mask.nonzero()[0] - right_border))
                right_weights[j, j_mask] = (1. / (overlap_len ) * (j_mask.nonzero()[0] - left_border))
            else:
                left_weights[j, j_mask] = 0.5
                right_weights[j, j_mask] = 0.5
    if left_is_left:
        return left_weights, right_weights
    else:
        return right_weights, left_weights
def PoissonBlending(stitch_img, masks, imgs):
    H, W = stitch_img.shape[:2]
    stitch_img = stitch_img.astype('float')
    for l in range(len(imgs) - 1):
        src_img = imgs[l].astype('float')
        src_mask = masks[l]
        tar_img = imgs[l + 1].astype('float')
        tar_mask = masks[l + 1]
        blending_mask = (src_mask) & (tar_mask)
        fill_mask = (src_mask) | (tar_mask)
        boundary_mask = np.logical_xor(src_mask, tar_mask)
        loc = blending_mask.nonzero()
        loc_map = {} # mapping from coordinate to variable
        for i_loc, (j, i) in enumerate(zip(loc[0], loc[1])):
            loc_map[(j, i)] = i_loc
        w_l, w_r = BlendingWeights(src_mask, tar_mask, src_img.shape[0])
        N = np.count_nonzero(blending_mask)
        y_min = np.min(loc[0])
        y_max = np.max(loc[0])
        x_min = np.min(loc[1])
        x_max = np.max(loc[1])
        res = np.zeros((N, 3))
        size = np.prod((y_max-y_min+1, x_max-x_min+1))
        print('solving...N: {} at level {} and {}'.format(N, l, l+1))
        stride = x_max - x_min + 1
        src = src_img
        tar = tar_img
        A = scipy.sparse.identity(N, format='lil')
        b = np.zeros((N, 3), dtype='float')
        for (j, i) in zip(loc[0], loc[1]):
                alpha = 1.0#w_l[j, i]
                cur_ptr = loc_map[(j, i)]
                if(blending_mask[j, i]):
                    N_p = 0.0
                    v_pq = np.zeros((1,3), dtype='float')
                    f_p = src[j, i, :]
                    g_p = tar[j, i, :]
                    if(j > 0):
                        if(fill_mask[j - 1, i]): #upper neighbor exists
                            f_q = src[j-1, i, :]
                            g_q = tar[j-1, i, :]
                            if(blending_mask[j - 1, i]): # in the omega
                                v_pq += [alpha, 1-alpha]@np.array([(f_p-f_q), (g_p-g_q)])
                                A[cur_ptr, loc_map[(j-1, i)]] = -1.0
                            else: # on the boundary
                                # known function f*_p + v_pq
                                # here we choose gradient image of original image with its
                                # pixel value exists.
                                v_pq += stitch_img[j-1, i, :] + (f_p-f_q, g_p-g_q)[~src_mask[j-1, i]] 
                            N_p += 1.0
                    if(j < H - 1):
                        if(fill_mask[j + 1, i]): #lower neighbor exists
                            f_q = src[j+1, i, :]
                            g_q = tar[j+1, i, :]
                            if(blending_mask[j + 1, i]): # in the omega
                                v_pq +=  [alpha, 1-alpha]@np.array([(f_p-f_q), (g_p-g_q)])
                                A[cur_ptr, loc_map[(j+1, i)]] = -1.0
                            else: # on the boundary
                                v_pq +=stitch_img[j+1, i, :] + (f_p-f_q, g_p-g_q)[~src_mask[j+1, i]]
                            N_p += 1.0
                    if(fill_mask[j, i - 1]): #left neighbor exists
                        f_q = src[j, i-1, :]
                        g_q = tar[j, i-1, :]
                        if(blending_mask[j, i-1]): # in the omega
                            v_pq += [alpha, 1-alpha]@np.array([(f_p-f_q), (g_p-g_q)])
                            A[cur_ptr, loc_map[(j, i-1)]] = -1.0
                        else: # on the boundary
                            v_pq +=stitch_img[j, i-1, :] + (f_p-f_q, g_p-g_q)[~src_mask[j, i-1]]
                        N_p += 1.0
                    if(fill_mask[j, i + 1]): #right neighbor exists
                        f_q = src[j, i+1, :]
                        g_q = tar[j, i+1, :]
                        if(blending_mask[j, i+1]): # in the omega
                            v_pq += [alpha, 1-alpha]@np.array([(f_p-f_q), (g_p-g_q)])
                            A[cur_ptr, loc_map[(j, i+1)]] = -1.0
                        else: # on the boundary
                            v_pq +=stitch_img[j, i+1, :] + (f_p-f_q, g_p-g_q)[~src_mask[j, i+1]]
                        N_p += 1.0
                    A[cur_ptr, cur_ptr] = N_p
                    b[cur_ptr, :] = v_pq.astype('float')
                else: # not in blending region
                    raise Exception('Illegal image!!')
        A = A.tocsr()
        for c in range(3):
            x = pyamg.solve(A, b[:, c], verb=False, tol=1e-5)
            x = np.clip(x, 0, 255)
            res[:, c] = x
        stitch_img[blending_mask, :] = res
    return stitch_img

