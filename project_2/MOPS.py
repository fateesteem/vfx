import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from cylindrical_proj import *
from data_helper import *
from MOPS_feature import FeaturePoint

EPS = 1.0e-10

def Build_pyramid(img, l = 3):
    init_H, init_W = img.shape[:2]
    img_gray = (img[:, :, :3] @ [0.114, 0.587, 0.299])
    print(img_gray.shape)
    imgs_pyramid = [img_gray]
    for i in range(l):
        imgs_pyramid.append(cv2.GaussianBlur(imgs_pyramid[i], None, 1.0)[::2, ::2])
    tot_W = 2*(1 - 2**(-1 * l)) * init_W
    img_tot = np.zeros((init_H, int(tot_W)))
    for i in range(l):
        img_tot[:int(init_H >> i), \
                int(init_W*(2 * (1 - 2 ** (-1 * i)))):int(init_W * (2 * (1 - 2 ** (-1 * (i + 1)))))] = imgs_pyramid[i]
    cv2.imshow('pyramid', img_tot.astype(np.uint8))
    feature_points = []
    for i in range(l):
        print('processing level {} ...'.format(i))
        feature_pts_loc = feature_detect(imgs_pyramid[i])
        feature_points += feature_descriptor(feature_pts_loc, i, imgs_pyramid[i])

    print('{} feature points collected !'.format(len(feature_points)))
    
"""
Adaptive non-maximal supression:

    Given a response map with only local maxima in a certain neighborhood,
    first we need to sort them in descending order, then calcuating the distance
    from each candidate point to its nearest point which is greater than it, this distance
    is the maximal radius for each candidate points, that is, the maximal circular region 
    this candidate point can be considered as local maxima.
    After establishing this table, we can find the top k radius in the table and regard 
    them as final feature points.
    
    Args:
        candidate: response after excluding non-local-maxima
    Returns:
        coordinates of local maximas within radius which satifies supression condition
"""
def ANMS(candidate, max_N=500):
    N = np.count_nonzero(candidate)
    ## extract coordinate of each candidate in tuple form ##
    can_loc_tup = candidate.nonzero()
    ## extract local maxima candidate, then sorting them in descending order ##
    candidate_extract = candidate[can_loc_tup]
    sorted_index = np.argsort(-1 * candidate_extract)
    can_loc = np.transpose(can_loc_tup)
    can_loc = can_loc[sorted_index]
    candidate_extract = candidate_extract[sorted_index]
    if N <= max_N:
        return can_loc
    ## calculate distance to each candidate point ##
    dist = can_loc[:, np.newaxis, :] - can_loc
    assert dist.shape[0] == len(candidate_extract)
    distance = np.sqrt(np.sum(np.square(dist), axis = 2))
    ## build mask to exclude all the point with less and equal response ##
    leq_mask = candidate_extract[np.arange(dist.shape[0])][:, None] \
                >= candidate_extract[np.arange(dist.shape[1])]
    ## find out minimum distance to the candidate strictly greater than it , that is, radius##
    distance[leq_mask] = np.inf
    radius = np.amin(distance, axis = 1)
    #print(radius)
    ## choose out top k radius ##
    sorted_radius_index = np.argsort(-1 * radius)
    return_loc = can_loc[sorted_radius_index][:max_N]
    return return_loc

"""
Subpixel refinement:
    
    Given a array consisting of feature points.
    Do subpixel refinement in 3x3 local region,
    all the approximation is w.r.t its central pt.

    Args:
        feature_pt_lct: coordinates of selected feature points
        Response: corner response over image
    Returns:
        array of refined coordinates
"""
def sub_pixel_refine(feature_pt_lct, Response):
    ## calculating Hessian and gradient in each direction ##
    shape = Response.shape
    fy, fx = np.gradient(Response)
    fxx = np.zeros(shape, dtype = np.float32)
    fyy = np.zeros(shape, dtype = np.float32)
    fxy = np.zeros(shape, dtype = np.float32)
    fxx[:, 1:-1] = Response[:, 2:] - 2 * Response[:, 1:-1] + Response[:, :-2] # f_1,0 - 2f_0,0 + f_-1,0
    fyy[1:-1, :] = Response[2:, :] - 2 * Response[1:-1, :] + Response[:-2, :] # f_0,1 - 2f_0,0 + f_0,-1
    fxy[1:-1, 1:-1] =  (Response[:-2, :-2] - Response[2:, :-2] - Response[:-2, 2:] \
            + Response[2:, 2:]) / 4 # (f_-1,-1 -f_-1,1 - f_1,-1 + f_1,1) / 4
    
    ## expand hessian inverse operator then calculating correction shift##
    det_H = fxx * fyy - fxy ** 2
    xc = -1 * (fyy * fx - fxy * fy)/(det_H + EPS)
    yc = -1 * (-fxy * fx + fxx * fy)/(det_H + EPS)

    ## correct each feature point location with threshold 0.5 ##
    corrected_ft_pts = []
    for i in range(feature_pt_lct.shape[0]):
        y, x = feature_pt_lct[i]
        shift_x = xc[y, x]
        shift_y = yc[y, x]
        if shift_x > 0.5:
            x += 1
        elif shift_x < -0.5:
            x -= 1 
        if shift_y > 0.5:
            y += 1
        elif shift_y < -0.5:
            y -= 1

        feature_pt_lct[i, 0] = y
        feature_pt_lct[i, 1] = x

    return feature_pt_lct


def feature_detect(img):
    H, W = img.shape[:2]
    org_img = np.dstack([img] * 3)
    gradient_img = np.zeros((H, W, 2))
    gradient_img[..., 1], gradient_img[..., 0] = np.gradient(img)
    gradient_img[..., 0] = cv2.GaussianBlur(gradient_img[..., 0], None, 1.0)
    gradient_img[..., 1] = cv2.GaussianBlur(gradient_img[..., 1], None, 1.0)
    Hess = cv2.GaussianBlur(np.repeat(gradient_img, [2, 2], axis = 2) \
            * np.dstack([gradient_img, gradient_img]), None, 1.5) #[xx, xy, yx, yy]

    ## corner response ##
    f_HM = (Hess[..., 0] * Hess[..., 3] - Hess[..., 1] ** 2) / (Hess[..., 0] + Hess[..., 3] + EPS)
    bound_thres = 20 << 0 # at least 20 * 2^(sample scale = 2) pixel needs to generate descriptor
    bound_mask = ((np.arange(H) <= bound_thres) | (np.arange(H) >= H-bound_thres))[:, None]\
                |((np.arange(W) <= bound_thres) | (np.arange(W) >= W-bound_thres))
    f_HM[bound_mask] = 0.0

    ## pick local maximum in 3x3 and >= 10 ##
    selected_max = cv2.dilate(f_HM, np.ones((3,3))) # find local maxima in 3x3 region
    located_mask = (f_HM == selected_max) & (f_HM >= 10.0) 
    print('Current candidates before ANMS: {}'.format(np.count_nonzero(located_mask)))
    
    lcl_max_img = cv2.copyMakeBorder(org_img,0,0,0,0,cv2.BORDER_REPLICATE)
    for x in np.transpose(np.nonzero(f_HM * located_mask)):
        cv2.circle(lcl_max_img, tuple(x[::-1]), 1, (0,0,255))

    ## adaptive non-maxima supression ##
    local_max_candidate = f_HM * located_mask # only local maximum is crucial
    sup_location = ANMS(local_max_candidate)
    print('After ANMS: {}'.format(len(sup_location)))
    
    ANMS_img = cv2.copyMakeBorder(org_img,0,0,0,0,cv2.BORDER_REPLICATE)
    for x in sup_location:
        cv2.circle(ANMS_img, tuple(x[::-1]), 1, (0,0,255))
    print('Subpixel refinement ...')    
    rf_sup_locations = sub_pixel_refine(sup_location, f_HM)
    ref_img = cv2.copyMakeBorder(org_img,0,0,0,0,cv2.BORDER_REPLICATE)
    for x in rf_sup_locations:
        cv2.circle(ANMS_img, tuple(x[::-1]), 1, (0,255,255))
    """
    cv2.imshow('local_max', lcl_max_img.astype(np.uint8))
    cv2.imshow('ANMS', ANMS_img.astype(np.uint8))
    
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    """
    return rf_sup_locations
def feature_descriptor(feature_pts_lct, level, img):
    ## orientation calculation ##
    H, W = img.shape[:2]
    u = np.zeros((H, W, 2))
    u[..., 1], u[..., 0] = np.gradient(img)
    u = cv2.GaussianBlur(u, None, 4.5)
    u /= np.sqrt(np.square(u[..., 0]) + np.square(u[..., 1]) + EPS)[:, :, np.newaxis]
    ## sample at 2sigma_p blurred img with 5x5 avg_pooling ##
    img_blur = cv2.GaussianBlur(img, None, 2 * 1.0)
    kernel = np.ones((5, 5), dtype = np.float32) / 25.0
    sampled_img = cv2.filter2D(img_blur, -1, kernel)
    feature_points = []
    ## write descriptor to each feature points ##
    for pt in feature_pts_lct:
        y, x = pt
        orientation = np.array([u[y, x, 0], u[y, x, 1]]) #[cos, sin]
        rotation = np.array([[orientation[0], -1 * orientation[1]], [orientation[1], orientation[0]]])
        ## first we setting all the sample index on spatial acting like stride 5##
        x, y = np.meshgrid(range(-18, 18, 5), range(-18, 18, 5)) # choose central point as index may be more stable
        ## apply transform on this sample points ##
        transformed_point = rotation @ np.array([x.ravel(), y.ravel()])
        sample_coor = np.int32(np.rint(transformed_point))
        descriptor = sampled_img[sample_coor[1], sample_coor[0]]
        ## normalization ##
        descriptor = (descriptor - descriptor.mean()) / (descriptor.std() + EPS)
        feature_points.append(FeaturePoint(x << level, y << level, orientation, level, descriptor))
    
    return feature_points

if __name__ == '__main__':
    imgs, fs = Load_Data('parrington', 'parrington/f.txt', '.jpg')
    img_proj = []
    Build_pyramid(imgs[1])
    #Harris(imgs[0]) 
    cv2.imshow('old', imgs[1][:, :, :])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
