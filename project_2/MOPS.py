import cv2
import numpy as np
from MOPS_feature import FeaturePoint
from data_helper import Load_Data
from inverse_cylindrical_proj import inverse_cylindrical_projection
EPS = 1.0e-10

"""
Build image pyramid:

    Blur image with std=1.0 and downsample it with factor 2.
    Then apply feature detection and description

    Args:
        img: input image
        l: pyramid level
    Returns:
        list of feature points
"""
def Build_pyramid(img_mask, img, l = 6, verbose = False):
    init_H, init_W = img.shape[:2]
    img_gray = (img[:, :, :3] @ [0.114, 0.587, 0.299])
    imgs_pyramid = [img_gray]
    imgs_mask_pyramid = [img_mask]
    for i in range(l):
        imgs_pyramid.append(cv2.GaussianBlur(imgs_pyramid[i], None, 1.0)[::2, ::2])
        imgs_mask_pyramid.append(imgs_mask_pyramid[i][::2, ::2])
    tot_W = 0
    for i in range(l):
        tot_W += imgs_pyramid[i].shape[1]
    img_tot = np.zeros((init_H, tot_W))
    feature_points = []
    for i in range(l):
        print('processing level {} ...'.format(i))
        feature_pts_loc = feature_detect(imgs_mask_pyramid[i], imgs_pyramid[i])
        feature_points += feature_descriptor(feature_pts_loc, i, imgs_pyramid[i])

    print('{} feature points collected !'.format(len(feature_points)))
    print()
    if verbose:
        W_acc = 0
        for i in range(l):
            H, W = imgs_pyramid[i].shape[:2]
            img_tot[:H, W_acc:W_acc + W] = imgs_pyramid[i]
            W_acc += W
        img_tot = np.dstack([img_tot] * 3)
        ct=0
        for pt in feature_points:
            ct +=1
            level = pt.level
            orientation = pt.orientation
            cv2.circle(img_tot, ((pt.x >> level) + int(init_W*(2 * (1 - 2 ** (-1 * level)))), pt.y >> level), 1, (255 >> (level + 1) ,0 , 255 >> level), -1)
            if level == 0 and ct%10 == 0:
                x, y = np.meshgrid(range(-20, 21, 40), range(-20, 21, 40))
                rotation = np.array([[orientation[0], -1 * orientation[1]], [orientation[1], orientation[0]]])
                x <<= level # [-20, 20, -20, 20]
                y <<= level # [-20, -20, 20, 20]
                transformed_point = rotation @ np.array([x.ravel(), y.ravel()]) + [[pt.x], [pt.y]]
                sample_coor = np.int32(np.rint(transformed_point))
                cv2.line(img_tot, (pt.x, pt.y), ((sample_coor[0][1] + sample_coor[0][3]) >> 1, (sample_coor[1][1] + sample_coor[1][3]) >> 1), 
                        (0, 0, 255), 2)
                cv2.line(img_tot, (sample_coor[0][0], sample_coor[1][0]), (sample_coor[0][1], sample_coor[1][1]), (0, 0, 255), 2)
                cv2.line(img_tot, (sample_coor[0][1], sample_coor[1][1]), (sample_coor[0][3], sample_coor[1][3]), (0, 0, 255), 2)
                cv2.line(img_tot, (sample_coor[0][3], sample_coor[1][3]), (sample_coor[0][2], sample_coor[1][2]), (0, 0, 255), 2)
                cv2.line(img_tot, (sample_coor[0][2], sample_coor[1][2]), (sample_coor[0][0], sample_coor[1][0]), (0, 0, 255), 2)
        cv2.imshow('pyramid', img_tot.astype(np.uint8))
        cv2.imwrite('pyramid.jpg', img_tot.astype(np.uint8))
    return feature_points    
"""
Adaptive non-maximal suppression:

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
def ANMS(candidate, max_N=700):
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

"""
Feature detection:
    
    Use Harris corner detection with harmonic function to evaluate corner Response.
    Choose out local maximum in 3x3 local region, then find out top k feature points,
    under maximal radius criterion with adaptive non-maximal suppression.
    Apply sub-pixel refinement to find out precise maximal points, and correct the coarse
    location but quantized.

    Args:
        img: blurred and downsampled image from image pyramid.

    Returns:
        numpy array of feature points location
"""
def feature_detect(img_mask, img):
    H, W = img.shape[:2]
    org_img = np.dstack([img] * 3)
    img_mask = cv2.dilate(np.logical_not(img_mask).astype(np.uint8), np.ones((41,41)))
    img_mask = np.logical_not((img_mask).astype(bool))
    gradient_img = np.zeros((H, W, 2))
    gradient_img[..., 1], gradient_img[..., 0] = np.gradient(img)
    gradient_img[..., 0] = cv2.GaussianBlur(gradient_img[..., 0], None, 1.0) * img_mask
    gradient_img[..., 1] = cv2.GaussianBlur(gradient_img[..., 1], None, 1.0) * img_mask
    Hess = cv2.GaussianBlur(np.repeat(gradient_img, [2, 2], axis = 2) \
            * np.dstack([gradient_img, gradient_img]), None, 1.5) #[xx, xy, yx, yy]

    ## corner response ##
    f_HM = (Hess[..., 0] * Hess[..., 3] - Hess[..., 1] ** 2) / (Hess[..., 0] + Hess[..., 3] + EPS)
    bound_thres = int(np.ceil(20*np.sqrt(2))) << 0 # at least 20 * 2^(sample scale = 2) pixel needs to generate descriptor
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

"""
MOPS feature descriptor:

    First, calculate orientation of each feature point via smoothed gradient(std = 4.5).
    As recommendation in paper, use current level image but blurred(std = 2*1.0) to sample
    8x8 patches with spacing = 5. 
    The patches need to be sample in the direction consistent with orientation of feature point.
    Hence, apply rotation to each sample coordinate.

    Args:
        feature_pts_lct: numpy array of feature points locations.
        level: current image level of feature points
        img: current image
    Returns:
        list of feature points at curent level.
"""
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
        pt_y, pt_x = pt
        orientation = np.array([u[pt_y, pt_x, 0], u[pt_y, pt_x, 1]]) #[cos, sin]
        rotation = np.array([[orientation[0], -1 * orientation[1]], [orientation[1], orientation[0]]])
        ## first we setting all the sample index on spatial acting like stride 5##
        x, y = np.meshgrid(range(-18, 18, 5), range(-18, 18, 5)) # choose central point as index may be more stable
        ## apply transform on this sample points ##
        transformed_point = rotation @ np.array([x.ravel(), y.ravel()]) + np.array([[pt_x], [pt_y]]) #[[x_coor], [y_coor]]
        sample_coor = np.int32(np.rint(transformed_point))
        descriptor = sampled_img[sample_coor[1], sample_coor[0]]
        ## normalization ##
        descriptor = (descriptor - descriptor.mean()) / (descriptor.std() + EPS)
        feature_points.append(FeaturePoint(pt_x << level, pt_y << level, orientation, level, descriptor))
    
    return feature_points

if __name__ == '__main__':
    #imgs, fs = Load_Data('./photos/bridge', './photos/bridge/f.txt', '.JPG')
    imgs, fs = Load_Data('./parrington', './parrington/f.txt', '.jpg')
    imgs_proj = []
    imgs_proj_mask = []
    fs /= 5
    for i in range(1):
        new_img, new_img_mask = inverse_cylindrical_projection(imgs[i], fs[i])
        imgs_proj.append(new_img)
        imgs_proj_mask.append(new_img_mask)
    feature_points = Build_pyramid(imgs_proj_mask[0], imgs_proj[0], verbose = True)
    draw_img = cv2.copyMakeBorder(imgs_proj[0],0,0,0,0,cv2.BORDER_REPLICATE)
    for pt in feature_points:
        level = pt.level
        orientation = pt.orientation
        cv2.circle(draw_img, (pt.x, pt.y), 1, (255 >> (level + 1) ,0 , 255 >> level), -1)
        if level == -1:
            x, y = np.meshgrid(range(-20, 21, 40), range(-20, 21, 40))
            rotation = np.array([[orientation[0], -1 * orientation[1]], [orientation[1], orientation[0]]])
            x <<= level # [-20, 20, -20, 20]
            y <<= level # [-20, -20, 20, 20]
            transformed_point = rotation @ np.array([x.ravel(), y.ravel()]) + [[pt.x], [pt.y]]
            sample_coor = np.int32(np.rint(transformed_point))
            cv2.line(draw_img, (pt.x, pt.y), ((sample_coor[0][1] + sample_coor[0][3]) >> 1, (sample_coor[1][1] + sample_coor[1][3]) >> 1), 
                    (0, 0, 255))
            cv2.line(draw_img, (sample_coor[0][0], sample_coor[1][0]), (sample_coor[0][1], sample_coor[1][1]), (0, 0, 255), 1)
            cv2.line(draw_img, (sample_coor[0][1], sample_coor[1][1]), (sample_coor[0][3], sample_coor[1][3]), (0, 0, 255), 1)
            cv2.line(draw_img, (sample_coor[0][3], sample_coor[1][3]), (sample_coor[0][2], sample_coor[1][2]), (0, 0, 255), 1)
            cv2.line(draw_img, (sample_coor[0][2], sample_coor[1][2]), (sample_coor[0][0], sample_coor[1][0]), (0, 0, 255), 1)
    cv2.imshow('input', imgs[1][:, :, :])
    cv2.imshow('feature', draw_img.astype(np.uint8))
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
