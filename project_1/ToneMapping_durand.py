import os
import numpy as np
import cv2
from BilateralFilter import BilateralFilter
from HDRAssemble import *
from MTBAlignment import MTBAlignment


def ToneMapping_durand(rad_maps, target_contrast=np.log10(5), gamma=0.75):
    rad_b = rad_maps[:, :, 0]
    rad_g = rad_maps[:, :, 1]
    rad_r = rad_maps[:, :, 2]
    I = 0.0722 * rad_b + 0.7152 * rad_g + 0.2126 * rad_r
    #I = 0.114 * rad_b + 0.587 * rad_g + 0.299 * rad_r
    logI = np.log10(I)
    blur_logI = BilateralFilter(logI)
    detail = logI - blur_logI

    blur_logI = (blur_logI - np.max(blur_logI)) * target_contrast / (np.max(blur_logI) - np.min(blur_logI))
    I_new = 10**(blur_logI + detail)
    rad_b_new = I_new * rad_b / I
    rad_g_new = I_new * rad_g / I
    rad_r_new = I_new * rad_r / I

    img_hdr = np.array([rad_b_new, rad_g_new, rad_r_new]).transpose(1, 2, 0)
    img_hdr = img_hdr**gamma
    img_hdr = np.where(img_hdr > 1.0, 1.0, img_hdr)
    return img_hdr



if __name__ == "__main__":
    #images, d_ts = Load_Data_test('./Memorial_SourceImages', ".png")
    images, d_ts = Load_Data('./test/images/7', './test/images/speed.txt', ".JPG")
    num_img = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    images_align = MTBAlignment(images, shift_range=20)
    imgs_b_align = images_align[:, :, :, 0]
    imgs_g_align = images_align[:, :, :, 1]
    imgs_r_align = images_align[:, :, :, 2]
    rad = np.zeros((H, W, 3), dtype='float')
    rad[:, :, 0], _ = Radiance_Map(imgs_b_align, d_ts, l=500.)
    rad[:, :, 1], _ = Radiance_Map(imgs_g_align, d_ts, l=500.)
    rad[:, :, 2], _ = Radiance_Map(imgs_r_align, d_ts, l=500.)
    SaveRad(rad, "radiance.hdr")
    img_hdr = ToneMapping_durand(rad)
    cv2.imshow('img_hdr', img_hdr)#cv2.resize(img_hdr, (840, int(H*840/W))))
    cv2.imwrite('memorial_hdr_durand.jpg', (img_hdr*255).astype('uint8'))
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
