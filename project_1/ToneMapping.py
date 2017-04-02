import numpy as np
import cv2
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
plt.style.use('classic')
from HDRAssemble import *

def Construct_Rad_map(Images, g, shape):
    w = np.array(Weight_func())
    rad = np.zeros((shape), dtype = np.float32)
    W = np.zeros((shape), dtype = np.float32)
    W += 1e-10
    for i, (image, d_t) in enumerate(Images):
        rad += w[image]*(g[image] - d_t)
        W += w[image]

    return rad / W
    
### radiance: HDR value ###
def Tone_mapping(radiance, key = 0.18, delta = 1.0e-6, Lwhite = 1.0):
    ### Reinhard mathod ###
    
    ### transform BGR to Luminance domain   ###
    Lw = 0.2126 * radiance[0] + 0.7152 * radiance[1] + 0.0722 * radiance[2]
    Lw_ = np.exp(np.mean(np.log(delta + Lw)))
    Lm = key * Lw / Lw_
    Ld = Lm * (1 + Lm / (Lwhite ** 2)) / (1 + Lm)

    img = radiance * Ld / Lw

    return np.transpose(img, (1, 2, 0))

if __name__ == '__main__':
    images_b, images_g, images_r, shape = Load_Data('./Image1')
    g = np.zeros((3, bin), dtype = np.float32)
    g_tmp, _ = Radiance_Map(images_b, shape, 20.)
    g[0] = g_tmp.reshape((bin, ))
    g_tmp, _ = Radiance_Map(images_g, shape, 20.)
    g[1] = g_tmp.reshape((bin, ))
    g_tmp, _ = Radiance_Map(images_r, shape, 20.)
    g[2] = g_tmp.reshape((bin, ))
    rad = np.zeros((3, shape[0], shape[1]), dtype = np.float32)
    rad[2] = Construct_Rad_map(images_b, g[0], shape)
    rad[1] = Construct_Rad_map(images_g, g[0], shape)
    rad[0] = Construct_Rad_map(images_r, g[0], shape)
    img = Tone_mapping(np.exp(rad))
    """
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
    """
    
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
    
    plt.figure(figsize =(6, 8))
    plt.imshow(img)
    plt.title("HDR image")
    
    plt.show()
    
