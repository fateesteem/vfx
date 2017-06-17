import os
import time
import numpy as np
import cv2
from GetPatchInterface import GetPatchInterface 
from  preprocess import MVCSolver, GetAdaptiveMesh, CalcBCCoordinates


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception("Failed to load the image from "+path)
    return img


class MVCCloner:
    def __init__(self, src_img_path, target_img_path, output_path, mvc_config):
        self.src_img = load_img(src_img_path)
        self.target_img = load_img(target_img_path)
        self.output_path = output_path
        self.GetPatchUI = GetPatchInterface(self.src_img)
        self.mvc_solver = MVCSolver(mvc_config)
        # source patch attributes #
        self.lefttop = None
        self.rightbottom = None
        self.boundary = None
        self.boundary_values = None
        self.patch_pnts = None
        self.patch_values = None
        # UI attributes #
        self.moving = False
        self.anchor = None
        self.win_X = self.target_img.shape[1]
        self.win_Y = self.target_img.shape[0]
        # Cloning attributes #
        self.MVCoords = None
        self.BCCoords = None
        self.triangles_vertices = None
        self.mesh_diffs = None
        self.num_boundary = None

    def GetPatch(self):
        # get source patch from UI #
        self.GetPatchUI.run()
        start_t = time.time()
        self.boundary, self.boundary_values, self.patch_pnts, self.patch_values = self.GetPatchUI.GetPatch(sample_step=2)
        print("GetPatch:", time.time() - start_t)
        start_t = time.time()
        # get adaptive triangular mesh #
        mesh, scipy_mesh = GetAdaptiveMesh(self.boundary, show=False)
        print("GetAdaptiveMesh:", time.time() - start_t)
        start_t = time.time()
        # vertices except boundary #
        self.num_boundary = self.boundary.shape[0]
        mesh_inner_vertices = scipy_mesh.points[self.num_boundary:]
        # Calc MV Coords #
        self.MVCoords = self.mvc_solver.CalcMVCoordinates(mesh_inner_vertices, self.boundary)
        print("MVCCoords:", time.time() - start_t)
        start_t = time.time()
        simplex_idxs, self.BCCoords = CalcBCCoordinates(scipy_mesh, self.patch_pnts)
        inliners_idxs = ~np.any(self.BCCoords < 0.-1e-8, axis=1)
        # filter outliers #
        self.patch_pnts = self.patch_pnts[inliners_idxs]
        self.patch_values = self.patch_values[inliners_idxs]
        simplex_idxs = simplex_idxs[inliners_idxs]
        self.BCCoords = self.BCCoords[inliners_idxs]
        # find simplex vertices of mesh points #
        self.triangles_vertices = scipy_mesh.simplices[simplex_idxs]
        # create space for storing mesh diffs #
        self.mesh_diffs = np.zeros((len(scipy_mesh.points), 3), dtype='float32')
        print("BCCoords:", time.time() - start_t)

    # mouse callback function
    def move_patch(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if np.all([x, y] > self.lefttop) and np.all([x, y] < self.rightbottom):
                self.moving = True
                self.anchor = np.array([x, y], dtype='int32')

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.moving:
                max_corner = [self.win_X, self.win_Y]
                displacement = [x, y] - self.anchor
                self.anchor = np.array([x, y], dtype='int32')
                if np.any((self.lefttop + displacement) < 0) or np.any((self.rightbottom + displacement) >= max_corner):
                    return
                self.set_patch_pos(displacement)

        elif event == cv2.EVENT_LBUTTONUP:
            self.moving = False
            self.anchor = None

    def set_patch_pos(self, displacement):
        self.boundary += displacement
        self.patch_pnts += displacement
        self.lefttop += displacement
        self.rightbottom += displacement

    def reset(self):
        screen_center = np.array([self.win_X >> 1 , self.win_Y >> 1], dtype='int32')
        self.lefttop = np.min(self.boundary, axis=0)
        self.rightbottom = np.max(self.boundary, axis=0)
        patch_center = ((self.rightbottom + self.lefttop) / 2).astype('int32')
        self.set_patch_pos(screen_center - patch_center)
        self.moving = False
        self.anchor = None

    def run(self):
        assert not self.boundary is None, "Source Patch is not selected yet!"
        self.reset()
        cv2.namedWindow('MVCCloner')
        cv2.setMouseCallback('MVCCloner', self.move_patch)
        while True:
            img = self.target_img.copy()
            clone_values = self.CalcCloningValues()
            self.patch_img(img, clone_values)
            cv2.imshow('MVCCloner', img)
            k = cv2.waitKey(5) & 0xFF
            if k == 32:     # space
                self.reset()
            elif k == 13:   # enter
                cv2.imwrite(self.output_path, img)
                break
        cv2.destroyAllWindows()

    def CalcCloningValues(self):
        target_boundary_values = self.target_img[self.boundary[:, 1], self.boundary[:, 0], :]
        diffs = target_boundary_values - self.boundary_values
        interpolants = self.MVCoords @ diffs
        self.mesh_diffs[:self.num_boundary, :] = diffs
        self.mesh_diffs[self.num_boundary:, :] = interpolants
        BCinterps = self.mesh_diffs[self.triangles_vertices]
        clone_values = np.einsum('ijk,ij->ik', BCinterps, self.BCCoords) + self.patch_values
        return np.clip(clone_values, 0., 255.).astype('uint8')

    def patch_img(self, img, set_values):
        img[self.patch_pnts[:, 1], self.patch_pnts[:, 0], :] = set_values



if __name__ == "__main__":
    mvc_config = {'hierarchic': True,
                  'base_angle_Th': 0.75,
                  'base_angle_exp': 0.8,
                  'base_length_Th': 2.5,
                  'adaptiveMeshShapeCriteria': 0.125,
                  'adaptiveMeshSizeCriteria': 0.,
                  'min_h_res': 16.}
    src_img = cv2.imread('./source.jpg')
    target_img = cv2.imread('./target.jpg')

    mvc_cloner = MVCCloner(src_img, target_img, mvc_config)
    mvc_cloner.GetPatch()
    mvc_cloner.run()

