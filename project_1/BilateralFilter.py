import os
import cv2
import numpy as np
from scipy.interpolate import interpn
from scipy.signal import fftconvolve



def BilateralFilter(img_gray, sigma_S=None, sigma_R=None, sampling_S=None, sampling_R=None):
    assert len(img_gray.shape) == 2, "input image must in gray-scale!"

    img_gray = img_gray.astype('float')
    img_H = img_gray.shape[0]
    img_W = img_gray.shape[1]
    R_min = np.min(img_gray)
    R_delta = np.max(img_gray) - R_min

    if sigma_S == None:
        sigma_S = min(img_W, img_H) / 16
    if sigma_R == None:
        sigma_R = 0.1 * R_delta
    if sampling_S == None:
        sampling_S = sigma_S
    if sampling_R == None:
        sampling_R = sigma_R

    derivedSigma_S = sigma_S / sampling_S
    derivedSigma_R = sigma_R / sampling_R
    padding_img = int(np.floor(2*derivedSigma_S) + 1)
    padding_R = int(np.floor(2*derivedSigma_R) + 1)

    ## build subsample grid data with diff intensity range ##
    downsample_H = int(np.floor((img_H - 1) / sampling_S) + 1 + 2*padding_img)
    downsample_W = int(np.floor((img_W - 1) / sampling_S) + 1 + 2*padding_img)
    downsample_R = int(np.floor(R_delta / sampling_R) + 1 + 2*padding_R)
    grid_data = np.zeros((downsample_H, downsample_W, downsample_R), dtype='float')
    grid_weights = np.zeros((downsample_H, downsample_W, downsample_R), dtype='float')

    xx, yy = np.meshgrid(range(img_W), range(img_H))
    dx = (np.round(xx / sampling_S) + padding_img).reshape(-1).astype('int')
    dy = (np.round(yy / sampling_S) + padding_img).reshape(-1).astype('int')
    dr = (np.round((img_gray - R_min) / sampling_R) + padding_R).reshape(-1).astype('int')
    raw_data = img_gray.reshape(-1)
    for p_dx, p_dy, p_dr, pix in zip(dx, dy, dr, raw_data):
        grid_data[p_dy, p_dx, p_dr] += pix
        grid_weights[p_dy, p_dx, p_dr] += 1.

    ## create gaussian kernel ##
    kernal_W = 2*derivedSigma_S + 1
    kernal_H = kernal_W
    kernal_R = 2*derivedSigma_R + 1

    half_kernal_W = int(np.floor(kernal_W / 2))
    half_kernel_H = int(np.floor(kernal_H / 2))
    half_kernal_R = int(np.floor(kernal_R / 2))

    grid_x, grid_y, grid_r = np.meshgrid(range(int(kernal_W)), range(int(kernal_H)), range(int(kernal_R)))
    grid_x -= half_kernal_W
    grid_y -= half_kernel_H
    grid_r -= half_kernal_R
    gaussian_xy = np.exp( -1 * (grid_x**2 + grid_y**2) / (2*derivedSigma_S**2))
    gaussian_r = np.exp( -1 * grid_r**2 / (2*derivedSigma_R**2))
    kernel = gaussian_xy * gaussian_r

    ## fft convolution ##
    blur_grid_data  = fftconvolve(grid_data, kernel, mode='same')
    blur_grid_weights  = fftconvolve(grid_weights, kernel, mode='same')

    ## normalize by weight ##
    blur_grid_weights = np.where(blur_grid_weights == 0, -2, blur_grid_weights)
    blur_grid_data /= blur_grid_weights

    ## upsample and interpolation ##
    xx, yy = np.meshgrid(range(img_W), range(img_H))
    dx = xx / sampling_S + padding_img
    dy = yy / sampling_S + padding_img
    dr = (img_gray - R_min) / sampling_R + padding_R

    grid_coordinate = (range(blur_grid_data.shape[0]), range(blur_grid_data.shape[1]), range(blur_grid_data.shape[2]))
    interp_points = (dy, dx, dr)
    img_blur = interpn(grid_coordinate, blur_grid_data, interp_points)
    return img_blur



if __name__ == "__main__":
    img = cv2.imread('./test/memorial0065.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = img_gray.shape[0]
    cols = img_gray.shape[1]
    img_blur = BilateralFilter(img_gray)
    print(img_blur.shape)
    cv2.imshow('img_gray', img_gray)
    cv2.imshow('img_blur', img_blur.astype('uint8'))
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
