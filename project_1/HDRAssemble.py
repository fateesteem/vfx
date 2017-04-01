import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    return weights

def gSolve(Z, B, l):
    assert Z.shape[1] == B.shape[0]
    w = Weight_func()
    A = np.zeros((Z.shape[0]*Z.shape[1]+(bin)+1, bin+Z.shape[0]), dtype='float')
    b = np.zeros((A.shape[0], 1), dtype='float')
    row = 0
    for n in range(Z.shape[0]):
        for p in range(Z.shape[1]):
            w_np = w[Z[n, p]]
            A[row, Z[n, p]] = w_np
            A[row, bin+n] = -1*w_np
            b[row, 0] = w_np*B[p]
            row += 1

    A[row, 129] = 1.
    row += 1

    for i in range(bin-2):
        A[row, i] = l*w[i+1]
        A[row, i+1] = -2*l*w[i+1]
        A[row, i+2] = l*w[i+1]
        row += 1
    X = np.matmul(np.linalg.pinv(A), b)
    g = X[:bin]
    logE = X[bin:]
    return g, logE

def Radiance_Map(images, shape, l):
    assert len(shape) == 2
    num_img = len(images)
    cand_list = [row*shape[1]+col for row in range(100, shape[0]-100) for col in range(50, shape[1]-50)]
    sample_idxs = random.sample(cand_list, 2*(bin//(num_img-1)+1))
    Z = np.empty((len(sample_idxs), 0), dtype='uint8')
    B = np.zeros(num_img, dtype='float')
    for i, (image, d_t) in enumerate(images):
        sample_pixs = image.reshape((-1, 1))[sample_idxs]
        Z = np.concatenate((Z, sample_pixs), axis=1)
        B[i] = d_t
    g, logE = gSolve(Z, B, l)
    return g, logE

def Load_Data(dir):
    speed_file = open(os.path.join(dir, "shutter_speed.txt"))
    d_time = {}
    for line in speed_file:
        tokens = line.split('\t')
        d_time[tokens[0]] = 1/float(tokens[1])
    images_r = []
    images_g = []
    images_b = []
    shape = None
    for file in os.listdir(dir):
        if file.endswith(".png"):
            d_t = d_time[file]
            img = cv2.imread(os.path.join(dir, file))
            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            print("load image "+file+' - d_t:', d_t)
            images_b.append((b, np.log(d_t)))
            images_g.append((g, np.log(d_t)))
            images_r.append((r, np.log(d_t)))
            assert b.shape == g.shape and g.shape == r.shape
            if not shape == None:
                assert b.shape == shape
            else:
                shape = b.shape
    return images_b, images_g, images_r, shape



if __name__ == "__main__":
    plt.ion()
    images_b, images_g, images_r, shape = Load_Data("../Memorial_SourceImages")
    g, logE = Radiance_Map(images_b, shape, 20.)
    print(g, logE)
    plt.plot(g.reshape(-1), range(g.shape[0]), 'r-')
    #plt.show()
    plt.draw()
    plt.pause(0.001)
    #g, logE = Radiance_Map(images_b, shape, 0.5))
    #g, logE = Radiance_Map(images_b, shape, 0.5))
