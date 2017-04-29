import numpy as np
import cv2
import os
from data_helper import Load_Data


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

    x += x_center
    x -= np.amin(x)
    y += y_center
    #new_img = interpolate(img, np.tile(x, H), y.ravel()).reshape(H, W, Ch).astype(np.uint8)
    new_W = (np.amax(np.floor(x)) - np.amin(np.floor(x)) + 1).astype(int)
    new_img = np.zeros((H, new_W, Ch), dtype=np.uint8)
    new_img[np.floor(y).astype(int), np.floor(np.tile(x, (H, 1))).astype(int), :] = img 
    return new_img


if __name__ == '__main__':
    imgs, fs = Load_Data('parrington', 'parrington/f.txt', '.jpg')
    img_proj = []
    for i in range(imgs.shape[0]):
        img_proj.append(cylindrical_projection(imgs[i], fs[i]))
    cv2.imshow('old', imgs[1])
    cv2.imshow('new', img_proj[1])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()


