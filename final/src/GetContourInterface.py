import cv2
import numpy as np


class GetContourInterface:
    def __init__(self, src_img=None):
        if src_img is None:
            self.src_img = np.zeros((512, 512, 3), np.uint8)
        else:
            self.src_img = src_img
        self.drawing = False # true if mouse is pressed
        self.add = False
        self.contour = np.empty([0, 2], dtype='int32')
        self.track = np.empty([0, 2], dtype='int32')
        self.first_idx = 0

    # mouse callback function
    def draw_contour(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.add = True
            self.contour = np.empty([0, 2], dtype='int32')
            self.track = np.empty([0, 2], dtype='int32')
            self.contour = np.append(self.contour, [[x, y]], axis=0)
            self.track = np.append(self.track, [[x, y]], axis=0)

        elif event == cv2.EVENT_MOUSEMOVE:
            pnt = np.array([x, y], dtype='int32')
            if self.drawing == True:
                self.track = np.append(self.track, [pnt], axis=0)
            if self.add:
                dist2origs = np.linalg.norm(pnt - self.contour[:20], axis=1)
                if self.contour.shape[0] > 40 and np.any(dist2origs < 5):
                    self.add = False
                    self.first_idx = np.argmin(dist2origs)
                elif not np.all(pnt == self.contour[-1]):
                    line_pnts = self.fix_contour(self.contour[-1], pnt)
                    self.contour = np.append(self.contour, line_pnts, axis=0)

        elif event == cv2.EVENT_LBUTTONUP:
            pnt = np.array([x, y], dtype='int32')
            dist2origs = np.linalg.norm(pnt - self.contour[:20], axis=1)
            self.track = np.append(self.track, [pnt], axis=0)
            if self.add:
                self.dist += np.linalg.norm(pnt - self.contour[-1])
                if not(self.contour.shape[0] > 40 and np.any(dist2origs < 5)):
                    if not np.all(pnt == self.contour[-1]):
                        line_pnts = self.fix_contour(self.contour[-1], pnt)
                        self.contour = np.append(self.contour, line_pnts, axis=0)
            self.first_idx = np.argmin(dist2origs)
            self.contour = self.contour[self.first_idx:]
            line_pnts = self.fix_contour(self.contour[-1], self.contour[0])
            self.contour = np.append(self.contour, line_pnts[:-1], axis=0)
            self.drawing = False
            self.add = False
            self.dist = 0.
            self.first_idx = 0

    def fix_contour(self, start_pnt, end_pnt):
        step_x = np.abs(end_pnt[0] - start_pnt[0])
        step_y = np.abs(end_pnt[1] - start_pnt[1])
        step = step_x if step_x >= step_y else step_y
        ratio = np.linspace(1, step, step)/step
        line_pnts = np.round((end_pnt*ratio[..., None] + start_pnt*(1-ratio[..., None]))).astype('int32')
        return line_pnts # return with end point

    def reset(self):
        self.drawing = False # true if mouse is pressed
        self.add = False
        self.contour = np.empty([0, 2], dtype='int32')
        self.track = np.empty([0, 2], dtype='int32')
        self.dist = 0.
        self.first_idx = 0

    def run(self):
        self.reset()
        cv2.namedWindow('GetContour')
        cv2.setMouseCallback('GetContour', self.draw_contour)
        while True:
            img = self.src_img.copy()
            if not len(self.contour) == 0:
                if self.drawing:
                    cv2.polylines(img, [self.track], False, (0, 255, 0), 3)
                else:
                    cv2.drawContours(img, [self.contour], 0, (0,255,0), 3)
            cv2.imshow('GetContour', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                self.reset()
            elif k == 13:
                break
        cv2.destroyAllWindows()

    def GetContour(self):
        #approx_contour = cv2.approxPolyDP(self.contour, 0.001, True)
        #approx_contour = approx_contour.reshape(-1, 2)
        approx_contour = self.contour
        return approx_contour


if __name__ == "__main__":
    src_img = cv2.imread('./moon.jpg')
    GetContourUI = GetContourInterface(src_img)
    GetContourUI.run()
    contour = GetContourUI.GetContour()
    img = np.zeros_like(src_img, dtype='uint8')
    cv2.drawContours(img, [contour], 0, (0, 255, 0), -1)
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 3)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    print("Contour:\n", contour)

