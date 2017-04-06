import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from MTBAlignment import MTBAlignment
from Rad2RGBE import Rad2RGBE

Z_min = 0
Z_max = 255
bin = 256


def Weight_func():
    weights = []
    for i in range(Z_max+1):
        if i <= 0.5*(Z_min + Z_max):
            weights.append(i - Z_min)
        else:
            weights.append(Z_max - i)
    return np.array(weights, dtype='float')

def gSolve(Z, B, l):
    assert Z.shape[1] == B.shape[0]
    w = Weight_func()
    A = np.zeros((Z.shape[0]*Z.shape[1]+(bin-2)+1, bin+Z.shape[0]), dtype='float')
    b = np.zeros((A.shape[0], 1), dtype='float')
    row = 0
    for n in range(Z.shape[0]):
        for p in range(Z.shape[1]):
            w_np = w[Z[n, p]]
            A[row, Z[n, p]] = w_np
            A[row, bin+n] = -1.*w_np
            b[row, 0] = w_np*B[p]
            row += 1

    #A[row, 127] = 1.
    A[row, 127] = w[127]
    row += 1

    for i in range(bin-2):
        A[row, i] = l*w[i+1]
        A[row, i+1] = -2.*l*w[i+1]
        A[row, i+2] = l*w[i+1]
        row += 1
    X = np.matmul(np.linalg.pinv(A), b).reshape(-1)
    g = X[:bin]
    logE = X[bin:]
    return g, logE

def getResponseCurve(images, d_ts, l, sample_skip=50):
    num_img = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    num_samples = 2 * (bin // (num_img-1) + 1)
    cand_list = [row*W+col for row in range(sample_skip, H-sample_skip) for col in range(sample_skip, W-sample_skip)]
    sample_idxs = random.sample(cand_list, num_samples)
    Z = (images.reshape(num_img, -1)[:, sample_idxs]).transpose()
    B = np.log(d_ts)
    g, logE = gSolve(Z, B, l)
    return g, logE

def Radiance_Map(images, d_ts, l):
    assert len(images.shape) == 3
    plt.ion()
    w = Weight_func() + 1e-10
    g_func, _ = getResponseCurve(images, d_ts, l=l, sample_skip=100)
    plt.plot(g_func, range(g_func.shape[0]), 'r-')
    plt.draw()
    plt.pause(5)
    rad_tot = np.sum(w[images]*(g_func[images] - np.log(d_ts[:, np.newaxis, np.newaxis])), axis=0)
    weight_tot = np.sum(w[images], axis=0)

    return np.exp(rad_tot / weight_tot), g_func

def SaveRad(img_rad, filename):
    assert img_rad.shape[2] == 3
    H = img_rad.shape[0]
    W = img_rad.shape[1]
    file = open(filename, "wb")
    file.write("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n".encode('utf-8'))
    file.write("-Y {0} +X {1}\n".format(H, W).encode('utf-8'))
    img_rgbe = Rad2RGBE(img_rad)
    img_rgbe.flatten().tofile(file)
    file.close()

def Load_Data_test(dir, img_type):
    speed_file = open(os.path.join(dir, "shutter_speed.txt"))
    d_t_dict = {}
    for line in speed_file:
        tokens = line.split('\t')
        d_t_dict[tokens[0]] = 1/float(tokens[1])
    images = []
    d_ts = []
    for file in os.listdir(dir):
        if file.endswith(img_type):
            d_t = d_t_dict[file]
            img = cv2.imread(os.path.join(dir, file))
            print("load image "+file+' - d_t:', d_t)
            images.append(img)
            d_ts.append(d_t)
    return np.array(images), np.array(d_ts)

def Load_Data(imgs_dir, speed_file, img_type):
    speed = open(speed_file)
    d_t_map = {}
    for line in speed:
        tokens = line.split('\t')
        fractions = tokens[1][:-1].split('/')
        assert len(fractions) <= 2
        d_t_map[tokens[0]] = float(fractions[0]) / float(fractions[1]) if len(fractions) == 2 else float(fractions[0])
    images = []
    d_ts = []
    for file in os.listdir(imgs_dir):
        if file.endswith(img_type):
            d_t = d_t_map[file[:-4]]
            img = cv2.imread(os.path.join(imgs_dir, file))
            print("load image "+file+' - d_t:', d_t)
            images.append(img)
            d_ts.append(d_t)
    return np.array(images), np.array(d_ts)

if __name__ == "__main__":
    images, d_ts = Load_Data_test("./Memorial_SourceImages", ".png")
    images_align = MTBAlignment(images, shift_range=20)
    imgs_b_align = images_align[:, :, :, 0]
    imgs_g_align = images_align[:, :, :, 1]
    imgs_r_align = images_align[:, :, :, 2]
    img_rad, _ = Radiance_Map(imgs_b_align, d_ts, l=1000.)
    print(img_rad.shape)
    #g, logE = Radiance_Map(images_b, shape, 0.5))
    #g, logE = Radiance_Map(images_b, shape, 0.5))
