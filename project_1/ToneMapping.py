import numpy as np
import cv2
import matplotlib
#matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from HDRAssemble import *
import math
from scipy.signal import fftconvolve

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
def Tone_mapping(radiance, mode = 'local', key = 1, delta = 1.0e-6, Lwhite = 1.0, phi = 10.0, eps = 0.3):
    ### Reinhard mathod ###

    ### transform BGR to Luminance domain   ###
    Lw = 0.2126 * radiance[:, :, 2] + 0.7152 * radiance[:, :, 1] + 0.0722 * radiance[:, :, 0] + delta
    #Lw = np.sqrt(0.114 * (radiance[0] ** 2) + 0.587 * (radiance[1] ** 2) + 0.299 * (radiance[2] ** 2))
    Lw_ = np.exp(np.mean(np.log(delta + Lw)))
    Lm = key * Lw / Lw_
    if mode == 'global':
        Ld = Lm * (1 + Lm / (Lwhite ** 2)) / (1 + Lm)
    else:
        s_max = None
        L_blur = np.zeros((9, Lw.shape[0], Lw.shape[1]), dtype = np.float32)
        kernel = gen_gaussian(1)
        L_blur[0] = fftconvolve(Lm, kernel, mode = 'same')
        for s_level in range(1, 9):
            s = 1.6 ** (s_level - 1)
            s_1 = 1.6 ** s_level    #next level
            kernel_s_1 = gen_gaussian(s_1)
            L_blur[s_level] = fftconvolve(Lm, kernel_s_1, mode = 'same')
            V_s = (L_blur[s_level - 1] - L_blur[s_level]) / ((2 ** phi)*key/(s ** 2) +  L_blur[s_level - 1])
            print('s: ' + str(s_level - 1) + '= ' + str(np.amax(np.abs(V_s))))
            if np.amax(np.abs(V_s)) < eps:
                s_max = s_level - 1
                #break
        if s_max == None:
            print('None of DOG satisfies epsilon criterion!!!\n' + 'Use global operator instead')
            Ld = Lm * (1 + Lm / (Lwhite ** 2)) / (1 + Lm)
        else:
            print('in local operator, s_max = ' + str(s_max) + ', w.r.t epsilon = ' + str(eps))
            Ld = Lm/(1 + L_blur[s_max])
    ### generate different scale gaussian filter ###
    img = radiance * Ld[:, :, np.newaxis] / Lw[:, :, np.newaxis]
    img = np.where(img > 1.0, 1.0, img)
    return img

def gen_gaussian(s, alpha = 1.0 / 2 / np.sqrt(2)):
    
    s = math.ceil(s)
    sigma = alpha * s

    miu = math.floor(s / 2)
    x, y = np.meshgrid(range(s), range(s))
    x -= miu
    y -= miu

    return np.exp(-(x ** 2 + y ** 2) / ((sigma) ** 2) / 2) / np.pi / (sigma ** 2) / 2

if __name__ == '__main__':
    #images, d_ts = Load_Data('./image3/1','./image3/1/speed.txt', '.JPG')
    images, d_ts = Load_Data_test('./Memorial_SourceImages', '.png')
    shape = images.shape[1:3]
    g = np.zeros((3, bin), dtype = np.float32)
    images_align = MTBAlignment(images, shift_range=20)
    rad = np.zeros((shape[0], shape[1], 3), dtype = np.float32)
    rad[:, :, 0], g_tmp = Radiance_Map(images_align[:, :, :, 0], d_ts, l=500.)
    g[0] = g_tmp.reshape((bin, ))
    rad[:, :, 1], g_tmp = Radiance_Map(images_align[:, :, :, 1], d_ts, l=500.)
    g[1] = g_tmp.reshape((bin, ))
    rad[:, :, 2], g_tmp = Radiance_Map(images_align[:, :, :, 2], d_ts, l=500.)
    g[2] = g_tmp.reshape((bin, ))
    img = Tone_mapping(rad, key = 0.25, Lwhite = 1.0, eps = 0.1,mode = 'local')
    Save_Rad(img, 'img.hdr')


    plt.figure()
    plt.plot(g[2], color = 'blue')
    plt.plot(g[1], color = 'green')
    plt.plot(g[0], color = 'red')
    plt.title('Reconstruct camera response curve')
    plt.show()
    """
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 8))#, sharex=False, sharey=True)
    im = ax1.imshow(np.log(rad[0]), aspect = 'auto')
    ax1.set_title('Blue')
    im = ax2.imshow(np.log(rad[1]), aspect = 'auto')
    ax2.set_title('Green')
    im = ax3.imshow(np.log(rad[2]), aspect = 'auto')
    ax3.set_title('Red')
    #plt.colorbar(im, ax = ax3)
    plt.suptitle('Reconstruct Radiance Map', fontsize=20)
    f.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    """
    cv2.imwrite('_hdr_reinhard.jpg', np.clip(img*255, 0, 255).astype(np.int32))
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()

