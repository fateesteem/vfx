import numpy as np
import cv2
import os
from gsolver import gsolver
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
class Image:
    def __init__(self, dir):
        self.dir = dir
        self.files = sorted(os.listdir(self.dir))
        self.files = [os.path.join(self.dir, f) for i, f in enumerate(self.files) if f[:9] == 'memorial0']

        print (self.files)
        self.shutter_speed = 1.0 / np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4
                                       , 8, 16, 32, 64, 128, 256, 512, 1024])

        self.img = []
        for i, f in enumerate(self.files):
            self.img.append(cv2.imread(f))
        self.img = np.array(self.img)
        self.num_img = len(self.img)
        self.height, self.width, self.channel= self.img.shape[1:]
        self.lamb = 10         
        self.sample_img = None
        self.g = np.zeros((256, 3), dtype = np.float32)
        
        #print('H: ' + str(self.height) + ' W: ' + str(self.width) + 
        #      ' #: ' + str(self.num_img))

    def sample(self, num_sample = None):
        if num_sample is None:
            num_sample = int(np.ceil(255.0 / self.num_img * 2))
        idx = list(range(self.width * self.height))
        np.random.shuffle(idx)
        sample_idx = idx[:num_sample]
        tmp = self.img.reshape([self.num_img, self.width * self.height, 3])
        self.sample_img = tmp[:, sample_idx, :]
        print(np.transpose(self.sample_img, (1, 0, 2))[:, :, 1])
    
    def solve_response(self):
        ### generate weighting function ###
        w = np.arange(256)
        w = np.minimum(w, 255 - w)
        
        ### transpose smaple image to be consistent with gsolver    ###
        Z = np.transpose(self.sample_img, (1, 0, 2))
        
        ### compute argument for solver ###
        ln_t = np.log(self.shutter_speed)
        
        for i in range(3):
            self.g[:, i], ln_E = gsolver(Z[:, :, i], ln_t, self.lamb, w)
        ln_E = ln_E.reshape([self.sample_img.shape[0], self.sample_img.shape[1]])
        """
        plt.plot(self.g[:, 0], 'blue', label = 'channel B')
        plt.plot(self.g[:, 1], 'green', label = 'channel G')
        plt.plot(self.g[:, 2], 'red', label = 'channel R')
        """
        plt.plot(ln_E)
        plt.show()
        

if __name__ == '__main__': 
    img = Image('Images_1/')
    img.sample()
    img.solve_response()
    cv2.imshow('image',img.img[10])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
