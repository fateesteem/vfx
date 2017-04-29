import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from cylindrical_proj import *

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
        img_tot[:int(init_H * 2 ** (-1 * i)), int(init_W*(2 * (1 - 2 ** (-1 * i)))):int(init_W * (2 * (1 - 2 ** (-1 * (i + 1)))))] = imgs_pyramid[i]
    #cv2.imshow('pyramid', img_tot.astype(np.uint8))
    Harris(imgs_pyramid[0])
def Harris(img):
    H, W = img.shape[:2]
    img = cv2.GaussianBlur(img, None, 1.5)
    gradient_img = np.zeros((H, W, 2))
    gradient_img[..., 1], gradient_img[..., 0] = np.gradient(img)
    H = cv2.GaussianBlur(np.repeat(gradient_img, [2, 2], axis = 2) * np.dstack([gradient_img, gradient_img]), None, 1.5) #[xx, xy, yx, yy]
    print(H.shape)
    #fx = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
    
    cv2.imshow('gray', img.astype(np.uint8))
    cv2.imshow('Ix', gradient_img[..., 0].astype(np.uint8))
    cv2.imshow('Iy', gradient_img[..., 1].astype(np.uint8))


if __name__ == '__main__':
    imgs, fs = Load_Data('parrington', 'parrington/f.txt', '.jpg')
    img_proj = []
    Build_pyramid(imgs[0])
    #Harris(imgs[0]) 
    cv2.imshow('old', imgs[0][:, :, :])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()

