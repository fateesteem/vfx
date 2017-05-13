import numpy as np
from scipy.interpolate import interp2d
import cv2
import os
from data_helper import Load_Data

def BiInterpn(x, y, img, H, W, C, img_mask = None):
    tmp_img = np.zeros((H, W, C), dtype = np.int)
    ## ramove all the illegal points
    img_h, img_w = img.shape[:2]
    mask = (x < 0) | (x >= img_w - 1) | (y < 0) | (y >= img_h - 1)
    
    x = x[~mask]
    y = y[~mask]
    
    x_coor, y_coor = np.meshgrid(range(W), range(H))

    x_coor = x_coor[~mask]
    y_coor = y_coor[~mask]


    ## find out all the interpolation component ##
    x_1 = np.floor(x + 1).astype(int) # floor(x + 1) to avoid ceil(x) == floor(x)
    x_0 = np.floor(x).astype(int)
    y_1 = np.floor(y + 1).astype(int)
    y_0 = np.floor(y).astype(int)
    
    if img_mask is not None: # need to exclude coordinate in empty region
        assert img.shape[:2] == img_mask.shape[:2]
        mask = (~img_mask[y_0, x_0]) | (~img_mask[y_0, x_1]) | (~img_mask[y_1, x_0]) | (~img_mask[y_1, x_1])
        x = x[~mask]
        y = y[~mask]
        x_coor = x_coor[~mask]
        y_coor = y_coor[~mask]
        x_0 = x_0[~mask]
        x_1 = x_1[~mask]
        y_0 = y_0[~mask]
        y_1 = y_1[~mask]

    ## weighting on four region ##
    a = (x - x_0) * (y - y_0)
    b = (x_1 - x) * (y - y_0)
    c = (x - x_0) * (y_1 - y)
    d = (x_1 - x) * (y_1 - y)
    tmp_img[y_coor, x_coor, :] = (a[..., None] * img[y_1, x_1, :]) \
            + (b[..., None] * img[y_1, x_0, :]) + (c[..., None] * img[y_0, x_1, :]) + (d[..., None] * img[y_0, x_0, :])
    
    new_mask = np.zeros((H, W), dtype = bool)
    new_mask[y_coor, x_coor] = True
    
    return tmp_img, new_mask
        

"""
    cylindrical projection by inverse mapping:
        
        First, find out inverse transform to map new coordinate to original image.
        This guarantees that each point after transformed can find out its correspoding point on original
        image. And it's easy to interpolate using original data on grid.

    Args: 
        img: image to be applied cylindrical projection
        focal: focal length of image
    Returns:
        new_img: transformed image
        mask: black region mask on new image
"""

def inverse_cylindrical_projection(img, focal, Interpolate = True):
    H, W, Ch = img.shape

    ## establish inverse coordinate w.r.t original img ##
    x_center = float(W-1)/2
    y_center = float(H-1)/2
    x = np.arange(W, dtype = np.float32) - x_center
    y = np.arange(H, dtype = np.float32) - y_center
    x = focal * np.tan(x / focal)
    r = np.sqrt(x ** 2 + focal ** 2)
    y = (y / focal)[:, np.newaxis] @ r[np.newaxis, :]

    x += x_center
    y += y_center
    
    
    if not Interpolate :
        x = np.round(np.tile(x, H).ravel()).astype(int)
        y = np.round(y.ravel()).astype(int)
        mask = ((x >= W) | (x < 0)) | ((y >= H) | (y < 0))
        tmp_img = np.zeros((H, W, Ch), dtype = np.int)
        x_coor, y_coor = np.meshgrid(range(W), range(H))
        x_min = np.amin(x_coor.ravel()[~mask])
        x_max = np.amax(x_coor.ravel()[~mask])
        new_W = x_max - x_min + 1
        tmp_img[y_coor.ravel()[~mask], x_coor.ravel()[~mask], :] = img[y[~mask], x[~mask], :]
        new_img = np.zeros((H, new_W, Ch), dtype = np.int) 
        new_img = tmp_img[:, x_min:x_max+1, :] 
        mask = mask.reshape(H, W)[:, x_min:x_max + 1]
    else: 
        img, mask = BiInterpn(np.tile(x, [H, 1]), y, img, H, W, Ch)
        y_coor, x_coor = np.mgrid[range(H), range(W)]
        x_min = np.amin(x_coor[mask])
        x_max = np.amax(x_coor[mask])
        new_W = x_max - x_min + 1
        
        y_min = np.amin(y_coor[mask])
        y_max = np.amax(y_coor[mask])
        new_H = y_max - y_min + 1
        
        new_img = np.zeros((new_H, new_W, Ch), dtype = np.int) 
        new_img = img[y_min:y_max + 1, x_min:x_max+1, :]
        new_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    return new_img, new_mask

    

### establish projection coordinate###
def cylindrical_projection(img, focal):
    
    H, W, Ch = img.shape
    x_center = float(W - 1) / 2
    y_center = float(H - 1) / 2

    ### first we establish coordinate   ###
    x = np.arange(W, dtype = np.float32) - x_center
    y = np.arange(H, dtype = np.float32) - y_center

    r = 1 / np.sqrt(x ** 2 + focal ** 2)
    h = y[:, np.newaxis] @ r[np.newaxis, :]
    x = focal * np.arctan(x / focal) 
    y = focal * h

    #x += x_center
    x -= np.amin(x)
    y += y_center
    #new_img = interpolate(img, np.tile(x, H), y.ravel()).reshape(H, W, Ch).astype(np.uint8)
    new_W = (np.amax(np.ceil(x)) - np.amin(np.floor(x)) + 1).astype(int)
    new_img = np.zeros((H, new_W, Ch), dtype=int)
    new_img_mask = np.zeros((H, new_W), dtype='bool')
    interp_mask = np.zeros((H, new_W), dtype=np.float32)
    
    gray_code = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for i in range(4):
        new_img[np.floor(y + gray_code[i, 0]).astype(int), np.floor(np.tile(x, (H, 1)) + gray_code[i, 1]).astype(int), :] += img
        new_img_mask[np.floor(y + gray_code[i, 0]).astype(int), np.floor(np.tile(x, (H, 1))+gray_code[i, 1]).astype(int)] = True
        interp_mask[np.floor(y + gray_code[i, 0]).astype(int), np.floor(np.tile(x, (H, 1)) + gray_code[i, 1]).astype(int)] += 1.0
    """
    new_img[np.floor(y).astype(int), np.ceil(np.tile(x, (H, 1))).astype(int), :] += img
    new_img_mask[np.floor(y).astype(int), np.ceil(np.tile(x, (H, 1))).astype(int)] = True
    interp_mask[np.floor(y).astype(int), np.ceil(np.tile(x, (H, 1))).astype(int)] += 1.0
    new_img[np.ceil(y).astype(int), np.floor(np.tile(x, (H, 1))).astype(int), :] += img
    new_img_mask[np.ceil(y).astype(int), np.floor(np.tile(x, (H, 1))).astype(int)] = True
    interp_mask[np.ceil(y).astype(int), np.floor(np.tile(x, (H, 1))).astype(int)] += 1.0
    new_img[np.ceil(y).astype(int), np.ceil(np.tile(x, (H, 1))).astype(int), :] += img
    new_img_mask[np.ceil(y).astype(int), np.ceil(np.tile(x, (H, 1))).astype(int)] = True
    interp_mask[np.ceil(y).astype(int), np.ceil(np.tile(x, (H, 1))).astype(int)] += 1.0
    """
    interp_mask[interp_mask < 1.0] = 1.0 

    new_img = new_img / interp_mask[..., None]


    #interpolate(new_img, interp_mask)
    return new_img, new_img_mask


if __name__ == '__main__':
    imgs, fs = Load_Data('./photos/riverside', './photos/riverside/f.txt', '.JPG')
    img_proj = []
    img_proj_mask = []
    fs *= 8
    """
    for i in range(imgs.shape[0]):
        new_img, new_img_mask = cylindrical_projection(imgs[i], fs[i])
        img_proj.append(new_img)
        img_proj_mask.append(new_img_mask)
    """
    img, mask= inverse_cylindrical_projection(imgs[1], fs[1])
    cv2.imshow('old', imgs[1])
    
    #cv2.imwrite('cy.jpg', (img_proj[1]*img_proj_mask[1][:,:,None]).astype('uint8'))
    cv2.imwrite('cy.jpg', (img * mask[:,:,None]).astype('uint8'))
    cv2.imshow('new', (img[:,:,:]).astype('uint8'))
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()


