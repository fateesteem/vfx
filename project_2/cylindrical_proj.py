import os
from itertools import product
import numpy as np
from scipy.interpolate import interp2d
import cv2
from data_helper import Load_Data


def Interp(y, x, values, H, W, Ch, mask=None):
    if mask is None:
        mask = np.ones_like(y, dtype='bool')
    new_img = np.zeros((H, W, Ch), dtype='float')
    new_img_weight = np.zeros((H, W), dtype='float')
    for dy, dx in product(range(2), range(2)):
        if dy == 0 and dx == 0:
            weight = 0.97
        else:
            weight = 0.01
        gause_y = np.floor(y).astype(int)+dy
        gause_x = np.floor(x).astype(int)+dx
        new_img[gause_y, gause_x, :] += values * weight
        new_img_weight[gause_y, gause_x] += weight * mask
    new_img_mask = new_img_weight > 0.
    new_img_weight[np.logical_not(new_img_mask)] = 1.
    new_img =  (new_img / new_img_weight[:, :, None]).astype('uint8')
    return new_img, new_img_mask

# establish projection coordinate #
def cylindrical_projection(img, focal):
    H, W, Ch = img.shape
    x_center = float(W - 1) / 2
    y_center = float(H - 1) / 2

    # first we establish coordinate #
    x = np.arange(W, dtype=np.float32) - x_center
    y = np.arange(H, dtype=np.float32) - y_center

    r = 1 / np.sqrt(x ** 2 + focal ** 2)
    h = y[:, np.newaxis] @ r[np.newaxis, :]
    x = focal * np.arctan(x / focal)
    y = focal * h

    # x += x_center
    x -= np.amin(x)
    y += y_center
    new_W = (np.amax(np.ceil(x)) - np.amin(np.floor(x)) + 1).astype(int)
    new_img, new_img_mask = Interp(y, np.tile(x, (H, 1)), img, H, new_W, Ch)
    """
    #new_img = interpolate(img, np.tile(x, H), y.ravel()).reshape(H, W, Ch).astype(np.uint8)
    new_W = (np.amax(np.floor(x)) - np.amin(np.floor(x)) + 1).astype(int)
    new_img = np.zeros((H, new_W, Ch), dtype=np.uint8)
    new_img_mask = np.zeros((H, new_W), dtype='bool')
    new_img[np.floor(y).astype(int), np.floor(np.tile(x, (H, 1))).astype(int), :] = img
    new_img_mask[np.floor(y).astype(int), np.floor(np.tile(x, (H, 1))).astype(int)] = True
    """
    return new_img, new_img_mask


if __name__ == '__main__':
    imgs, fs = Load_Data('parrington', 'parrington/f.txt', '.jpg')
    img_proj = []
    img_proj_mask = []
    for i in range(imgs.shape[0]):
        new_img, new_img_mask = cylindrical_projection(imgs[i], fs[i])
        img_proj.append(new_img)
        img_proj_mask.append(new_img_mask)
    cv2.imshow('old', imgs[1])
    cv2.imshow('new', (img_proj[1]*img_proj_mask[1][:, :, None]).astype('uint8'))
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
