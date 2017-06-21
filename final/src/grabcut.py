import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import time
import maxflow 

class GCManager:
    def __init__(self, src_img):
        self.src_img = src_img
        self.img = src_img.copy()
        self.row, self.col = src_img.shape[:2]
        print('read img... {} x {}'.format(self.row, self.col))
        ## coeff of energy ##
        self.gamma = 50.0
        self.lamb = self.gamma * 9.0
        self.beta = None

        ## drawing color ##
        self._DRAW_BG = {'color': [0,0,0], 'val':0}
        self._DRAW_FG = {'color': [255,255,255], 'val':1}
        self._DRAW_PR_BG = {'color': [0,255,255], 'val':2}
        self._DRAW_PR_FG = {'color': [0,0,255], 'val':3}

        ## flags & state ##
        self._rect = [0, 0, 1, 1]
        self._draw_rect = False
        self._rect_over = False
        self._drawing = False
        self._thickness = 3
        
        self.GC_BG = 0
        self.GC_FG = 1
        self.GC_PR_BG = 2
        self.GC_PR_FG = 3

        self.calcBeta()
        self.calcSmooth()

        self._mask = np.zeros_like(src_img, dtype=np.uint8)[:, :, 0]
        self._mask[:, :] = self.GC_BG

    def calcBeta(self):
        self._h_diff = self.src_img[:, 1:] - self.src_img[:, :-1]
        self._v_diff = self.src_img[1:, :] - self.src_img[:-1, :]
        self._n_diff = self.src_img[1:, 1:] - self.src_img[:-1, :-1]
        self._p_diff = self.src_img[1:, :-1] - self.src_img[:-1, 1:]

        beta = np.sum(self._h_diff ** 2) + np.sum(self._v_diff ** 2) + np.sum(self._p_diff ** 2) + np.sum(self._n_diff ** 2)
        self.beta = 1 / (2*beta / (self.row * self.col * 4 - 3 * (self.row + self.col) + 2)) # avg over total connection

    def calcSmooth(self):
        assert self.beta is not None
        self.h_weight = np.zeros([self.row, self.col], dtype='float32')
        self.v_weight = np.zeros([self.row, self.col], dtype='float32')
        self.p_weight = np.zeros([self.row, self.col], dtype='float32')
        self.n_weight = np.zeros([self.row, self.col], dtype='float32')

        self.h_weight[:, 1:] = self.gamma*np.exp(-self.beta*np.sum(self._h_diff ** 2, axis=2)) 
        self.v_weight[1:, :] = self.gamma*np.exp(-self.beta*np.sum(self._v_diff ** 2, axis=2)) 
        self.n_weight[1:, 1:] = self.gamma*np.exp(-self.beta*np.sum(self._n_diff ** 2, axis=2))
        self.p_weight[1:, :-1] = self.gamma*np.exp(-self.beta*np.sum(self._p_diff ** 2, axis=2))

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self._draw_rect = True
            self._ix, self._iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._draw_rect == True:
                self.img = self.src_img.copy()
                cv2.rectangle(self.img, (self._ix, self._iy), (x, y), [255, 0, 0], 2)
                self._rect = [min(self._ix, x), min(self._iy, y), abs(self._ix-x), abs(self._iy-y)]
                self.rect_mask = 0

        elif event == cv2.EVENT_RBUTTONUP:
            self._draw_rect = False
            self._rect_over = True
            self.img = self.src_img.copy()
            cv2.rectangle(self.img, (self._ix, self._iy), (x, y), [255, 0, 0], 2)
            self._mask[self._rect[1]:self._rect[1]+self._rect[3], self._rect[0]:self._rect[0]+self._rect[2]] = self.GC_PR_FG 

    def initGMM(self):
        self._bg = np.where(np.logical_or(self._mask == self.GC_BG, self._mask == self.GC_PR_BG))
        self._fg = np.where(np.logical_or(self._mask == self.GC_FG, self._mask == self.GC_PR_FG))
        self._bg_data = self.src_img[self._bg].reshape(-1, 3)
        self._fg_data = self.src_img[self._fg].reshape(-1, 3)
        self.BG_GMM = GaussianMixture(n_components=5, covariance_type='full', max_iter=50)
        self.FG_GMM = GaussianMixture(n_components=5, covariance_type='full', max_iter=50)
        cur_t = time.clock()
        self.BG_GMM.fit(self._bg_data)
        self.FG_GMM.fit(self._fg_data)
        print('finish initial train on GMM... {} s'.format(time.clock()-cur_t))

    def assignGMM(self):
        self._flatten_data = self.src_img.reshape(-1, 3)
        self.BG_label = self.BG_GMM.predict(self._bg_data)
        self.FG_label = self.FG_GMM.predict(self._fg_data)
        print('predict ...')

    def learnGMM(self):
        ## reinitialize GMM model ##
        self.BG_GMM = GaussianMixture(n_components=5, covariance_type='full', max_iter=50)
        self.FG_GMM = GaussianMixture(n_components=5, covariance_type='full', max_iter=50)

        self.BG_GMM.means_init = np.array([self._bg_data[self.BG_label == i].mean(axis=0) for i in range(5)])
        self.FG_GMM.means_init = np.array([self._fg_data[self.FG_label == i].mean(axis=0) for i in range(5)])

        self.BG_GMM.fit(self._bg_data)
        self.FG_GMM.fit(self._fg_data)
        print('finish train on GMM...')

    def constructGraph(self):
        self.BG_prob = self.BG_GMM.score_samples(self._flatten_data).reshape(self.row, self.col)
        self.FG_prob = self.FG_GMM.score_samples(self._flatten_data).reshape(self.row, self.col)

        self.graph = maxflow.GraphFloat()
        nodeids = self.graph.add_grid_nodes((self.row, self.col))
        for y in range(self.row):
            for x in range(self.col):
                ## assign data term ##
                if self._mask[y, x] == self.GC_PR_BG or self._mask[y, x] == self.GC_PR_FG:
                    fromSource = -self.BG_prob[y, x]
                    toSink = -self.FG_prob[y, x]
                elif self._mask[y, x] == self.GC_BG:
                    fromSource = 0
                    toSink = self.lamb
                else: #FG
                    fromSource = self.lamb
                    toSink = 0
                self.graph.add_tedge(nodeids[y, x], fromSource, toSink)
                
                ## assign smooth term ##
                if x > 0: # left term exists
                    w = self.h_weight[y, x]
                    self.graph.add_edge(nodeids[y, x], nodeids[y, x-1], w, w)
                if y > 0: # upper term exists
                    w = self.v_weight[y, x]
                    self.graph.add_edge(nodeids[y, x], nodeids[y-1, x], w, w)
                if x > 0 and y > 0: # upper left term exists
                    w = self.n_weight[y, x]
                    self.graph.add_edge(nodeids[y-1, x-1], nodeids[y, x], w, w)
                if x < self.col - 1 and y > 0: # upper right term exists
                    w = self.p_weight[y, x]
                    self.graph.add_edge(nodeids[y-1, x+1], nodeids[y, x], w, w)
        print('graph construction end...')
        print('maxflow: {}'.format(self.graph.maxflow()))
        self.nodeids = nodeids

    def EstSegmentation(self):
        est_seg = self.graph.get_grid_segments(self.nodeids)
        for y in range(self.row):
            for x in range(self.col):
                if self._mask[y, x] == self.GC_PR_BG or self._mask[y, x] == self.GC_PR_FG:
                    if est_seg[y, x]: # belongs to PR_BG
                        self._mask[y, x] = self.GC_PR_BG
                    else:
                        self._mask[y, x] = self.GC_PR_FG
        print('finish segmentation...')


    def run(self):
        self.initGMM()
        self.assignGMM()
        self.learnGMM()
        self.constructGraph()
        self.EstSegmentation()


if __name__ == '__main__':
    img = cv2.imread('../img/Dog.jpg')
    GrabCut = GCManager(img.copy())

    cv2.namedWindow('input')
    cv2.namedWindow('output')
    cv2.moveWindow('input',img.shape[1]+10,90)
    cv2.setMouseCallback('input',GrabCut.onmouse)
    output = np.zeros_like(img)
    while True:
        cv2.imshow('input', GrabCut.img)
        cv2.imshow('output', output)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('n'):
            if GrabCut.rect_mask == 0:
                print('initialize ...')
                GrabCut.run()
                GrabCut.rect_mask = 1
        
        mask = np.where((GrabCut._mask == GrabCut.GC_FG) + (GrabCut._mask == GrabCut.GC_PR_FG)
                        , 255, 0).astype('uint8')
        output = cv2.bitwise_and(GrabCut.src_img, GrabCut.src_img, mask=mask)
    cv2.destroyAllWindows()


