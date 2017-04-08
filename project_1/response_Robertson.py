import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('classic')
from HDRAssemble import *
from MTBAlignment import *
from ToneMapping import *
def RobertsonSolver(Z, d_ts, w, maxIter = 5):
    g = np.ones((bin, ), dtype = np.float32)
    E = np.ones((Z.shape[-1], 1), dtype = np.float32)
    eps = 1.0e-10
    for it in range(maxIter):
        print("At iteration: " + str(it))
        ### Assume g(Z_ij) is known, opt E_i    ###
        E = np.sum(w[Z] * g[Z] * d_ts[:, np.newaxis], axis = 0) / (np.sum(w[Z] * (d_ts[:, np.newaxis] ** 2), axis = 0) + eps)
        ### Assume E_i is known, opt g(Z_ij)    ###
        for i in range(bin):
            time_idx, pixel_idx = np.where(Z == i)
            E_m = len(time_idx)
            g[i] = np.sum(E[pixel_idx] * d_ts[time_idx]) / (E_m + eps)

    g /= g[127] if g[127] != 0 else eps

    return g, E
def getResponseCurveRobertson(images, d_ts, w):
    num_img = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    Z = images.reshape(num_img, -1)
    
    g, E = RobertsonSolver(Z, d_ts, w)

    E = E.reshape((H, W))

    return g, E

def RadianceMapRobertson(images, d_ts):
    w = Weight_func()
    g, rad = getResponseCurveRobertson(images, d_ts, w)

    return rad, np.log(g + 1e-10)

if __name__ == '__main__':
    images, d_ts = Load_Data('./image1','./image1/speed.txt', '.png')
    #images, d_ts = Load_Data_test('./Memorial_SourceImages', '.png')
    shape = images.shape[1:3]
    g = np.zeros((3, bin), dtype = np.float32)
    images_align = MTBAlignment(images, shift_range=20)
    rad = np.zeros((shape[0], shape[1], 3), dtype = np.float32)
    rad[:, :, 0], g_tmp = RadianceMapRobertson(images_align[:, :, :, 0], d_ts)
    g[0] = g_tmp.reshape((bin, ))
    rad[:, :, 1], g_tmp = RadianceMapRobertson(images_align[:, :, :, 1], d_ts)
    g[1] = g_tmp.reshape((bin, ))
    rad[:, :, 2], g_tmp = RadianceMapRobertson(images_align[:, :, :, 2], d_ts)
    g[2] = g_tmp.reshape((bin, ))
    img = Tone_mapping(rad, key = 0.18, Lwhite = 2.0, eps = 0.4,mode = 'local')
    Save_Rad(img, 'img.hdr')


    plt.figure()
    plt.plot(g[2], color = 'blue')
    plt.plot(g[1], color = 'green')
    plt.plot(g[0], color = 'red')
    plt.title('Reconstruct camera response curve')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 8))#, sharex=False, sharey=True)
    im = ax1.imshow(np.log(rad[:, :, 0]), aspect = 'auto')
    ax1.set_title('Blue')
    im = ax2.imshow(np.log(rad[:, :, 1]), aspect = 'auto')
    ax2.set_title('Green')
    im = ax3.imshow(np.log(rad[:, :, 2]), aspect = 'auto')
    ax3.set_title('Red')
    #plt.colorbar(im, ax = ax3)
    plt.suptitle('Reconstruct Radiance Map', fontsize=20)
    f.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    cv2.imwrite('_hdr_reinhard.jpg', np.clip(img*255, 0, 255).astype(np.int32))
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
