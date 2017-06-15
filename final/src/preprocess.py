import triangle
import triangle.plot
import numpy as np
import matplotlib.pyplot as plt
from GetContourInterface import GetContourInterface 


def GetAdaptiveMesh(contour, show=True):
    num_pnt = contour.shape[0]
    segments = np.stack([np.arange(num_pnt), (np.arange(num_pnt)+1)%num_pnt], axis=1)
    boundary = dict(vertices=contour, segments=segments)
    triangular_mesh = triangle.triangulate(boundary, 'pq')
    if show:
        triangle.plot.compare(plt, boundary, triangular_mesh)
        plt.show()
    return triangular_mesh

def CalcMVCoordinates(mesh_vertices, boundary):
    """
    Calculate Mean-Value Coordinates for each mesh vertex
    Args:
      mesh_vertices:    vertices of the triangular mesh without the boundary, numpy array of shape [N1, 2].
      boundary:         vertices of the boundary, numpy array of shape [N2, 2].
    Returns
      MVCoords:         Mean-Value Coordinates for each vertex, numpy array of shape [N1, N2]
    """
    pass


if __name__ == "__main__":
    GetContourUI = GetContourInterface()
    GetContourUI.run()
    contour = GetContourUI.GetContour()
    contour = contour[::2]
    num_pnt = contour.shape[0]
    print("num_pnt:", num_pnt)

    triangular_mesh = GetAdaptiveMesh(contour)
    print("# of mesh vertices:", triangular_mesh['vertices'].shape[0])

