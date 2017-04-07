import numpy as np
import cv2
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
plt.style.use('classic')
from HDRAssemble import *

def Construct_Rad_map(Images, d_ts, g, shape):
    w = Weight_func()
    rad = np.zeros((shape), dtype = np.float32)
    W = np.zeros((shape), dtype = np.float32)
    W += 1e-10
    for i, (image, d_t) in enumerate(zip(Images, d_ts)):
        rad += w[image]*(g[image] - d_t)
        W += w[image]

    return rad / W

### radiance: HDR value ###
def Tone_mapping(radiance, key = 0.18, delta = 1.0e-6, Lwhite = 1.0):
    ### Reinhard mathod ###

    ### transform BGR to Luminance domain   ###
    Lw = 0.0722 * radiance[0] + 0.7152 * radiance[1] + 0.2126 * radiance[2]
    #Lw = 0.2126 * radiance[0] + 0.7152 * radiance[1] + 0.0722 * radiance[2]
    Lw_ = np.exp(np.mean(np.log(delta + Lw)))
    Lm = key * Lw / Lw_
    Ld = Lm * (1 + Lm / (Lwhite ** 2)) / (1 + Lm)

    img = radiance * Ld / Lw

    return np.transpose(img, (1, 2, 0))

if __name__ == '__main__':
    images, d_ts = Load_Data_test('./Memorial_SourceImages', '.png')
    #images, d_ts = Load_Data('./test/images/7', './test/images/speed.txt', '.JPG')
    num_img = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    images_align = MTBAlignment(images, shift_range=20)
    imgs_b_align = images_align[:, :, :, 0]
    imgs_g_align = images_align[:, :, :, 1]
    imgs_r_align = images_align[:, :, :, 2]
    g = np.zeros((3, bin), dtype = np.float32)
    rad = np.zeros((3, H, W), dtype = np.float32)
    rad[0], g[0] = Radiance_Map(imgs_b_align, d_ts, l=500.)
    rad[1], g[1] = Radiance_Map(imgs_g_align, d_ts, l=500.)
    rad[2], g[2] = Radiance_Map(imgs_r_align, d_ts, l=500.)
    img = Tone_mapping(rad, key=0.18)
    """
    g = np.zeros((3, bin), dtype = np.float32)
    g_tmp, _ = Radiance_Map(images_b_align, d_ts, images_b_align[0].shape, l=20.)
    g[0] = g_tmp.reshape((bin, ))
    g_tmp, _ = Radiance_Map(images_g_align, d_ts, images_g_align[0].shape, l=20.)
    g[1] = g_tmp.reshape((bin, ))
    g_tmp, _ = Radiance_Map(images_r_align, d_ts, images_r_align[0].shape, l=20.)
    g[2] = g_tmp.reshape((bin, ))
    rad = np.zeros((3, shape[0], shape[1]), dtype = np.float32)
    rad[2] = Construct_Rad_map(images_b_align, d_ts, g[0], shape)
    rad[1] = Construct_Rad_map(images_g, g[0], shape)
    rad[0] = Construct_Rad_map(images_r, g[0], shape)
    img = Tone_mapping(np.exp(rad))
    """

    cv2.imshow('img',img)#cv2.resize(img, (840, int(H*840/W))))
    cv2.imwrite("memorial_hdr_photo.jpg", img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

    plt.ion()
    plt.figure()
    plt.plot(g[2], color = 'blue')
    plt.plot(g[1], color = 'green')
    plt.plot(g[0], color = 'red')
    plt.title('Reconstruct camera response curve')

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 8))#, sharex=False, sharey=True)
    im = ax1.imshow(rad[0], aspect = 'auto')
    ax1.set_title('Blue')
    im = ax2.imshow(rad[1], aspect = 'auto')
    ax2.set_title('Green')
    im = ax3.imshow(rad[2], aspect = 'auto')
    ax3.set_title('Red')
    #plt.colorbar(im, ax = ax3)
    plt.suptitle('Reconstruct Radiance Map', fontsize=20)
    f.tight_layout()
    plt.subplots_adjust(top=0.85)
    """
    plt.figure(figsize =(6, 8))
    plt.imshow(img)
    plt.title("HDR image")
    """
    plt.draw()
    plt.pause(10)

