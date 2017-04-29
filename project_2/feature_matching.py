import numpy as np
import cv2
from data_helper import Load_Data
from scipy.spatial import KDTree


def genMatchPairs(feats_1, feats_2, k, p):
    """
    Generates feature matching pairs by searching for K nearest neighbors using KDTree.
    K nearest neighbors are searched mutually.
    Args:
      feats1: numpy array of shape [N, D]
      feats2: numpy array of shape [N, D]
      k:      The number of nearest neighbors to search.
      p:      Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance. 2 is the usual Euclidean di              stance. Infinity is the maximum-coordinate-difference distance.
    Returns:
      idxs:   indexs of features of matching pairs (first column: indexs of feats_1, second column: indexs of feats_2)
    """
    num_1 = feats_1.shape[0]
    num_2 = feats_2.shape[0]
    assert num_1 > k and num_2 > k, "The number of features should be greater than k!"

    ## build KDTree ##
    tree_1 = KDTree(feats_1)
    tree_2 = KDTree(feats_2)

    ## K nearest neighbors ##
    _, nb_of_1 = tree_2.query(feats_1, k=k, p=p)
    _, nb_of_2 = tree_1.query(feats_2, k=k, p=p)
    
    ## creates return values ##
    idxs = np.zeros(shape=((num_1 + num_2)*k, 2), dtype='int')
    idxs[:num_1*k, 0] = np.repeat(np.arange(num_1), k)
    idxs[:num_1*k, 1] = nb_of_1.reshape(-1)
    idxs[num_1*k:, 1] = np.repeat(np.arange(num_2), k)
    idxs[num_1*k:, 0] = nb_of_2.reshape(-1)

    ## remove duplicates ##
    pairs = [tuple(row) for row in idxs]
    unique_pairs = set(pairs)
    idxs = np.array(list(unique_pairs))

    return idxs


if __name__ == "__main__":
    imgs, fs = Load_Data('parrington', 'parrington/f.txt', '.jpg')
    cv2.imshow('old', imgs[0][:, :, :])
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
