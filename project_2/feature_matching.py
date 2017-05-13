import time
import numpy as np
from scipy.spatial import KDTree
import cv2
from matplotlib import pyplot as plt
from data_helper import Load_Data
from MOPS import Build_pyramid
from cylindrical_proj import cylindrical_projection


def genMatchPairs(feats_1, feats_2, k, p):
    """
    Generates feature matching pairs by searching for K nearest neighbors using KDTree.
    K nearest neighbors are searched mutually.

    Args:
      feats1: numpy array of shape [N, D]
      feats2: numpy array of shape [N, D]
      k:      The number of nearest neighbors to search.
      p:      Which Minkowski p-norm to use. 1 is the sum-of-absolute-values "Manhattan" distance. 2 is the usual Euclidean di              stance. Infinity is the maximum-coordinate-difference distance.

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
    """
    batch = num_2 // 10
    x = []
    y = []
    for l in range(10):
        random_idx = np.arange(num_2)
        np.random.shuffle(random_idx)
        test_points = feats_2[random_idx[:(l+1)*batch], :]
        test_tree = KDTree(test_points)
        time_elapsed = []
        for f in test_points:
            startTime = time.time()
            test_tree.query(f, k=1, p=p)
            elapsedTime = time.time() - startTime
            time_elapsed.append(elapsedTime)
        x.append((l+1)*batch)
        y.append(np.mean(time_elapsed)*1000)
        print('[{}] finished in avg {:.2f} ms'.format("search in "+str((l+1)*batch)+" points", np.mean(time_elapsed) * 1000))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1000, 100))
    ax.set_yticks(np.arange(0, 6, 1))
    plt.scatter(x, y)
    plt.grid()
    plt.show()
    batch = num_1 // 10
    x = []
    y = []
    for l in range(10):
        random_idx = np.arange(num_1)
        np.random.shuffle(random_idx)
        test_points = feats_1[random_idx[:(l+1)*batch], :]
        test_tree = KDTree(test_points)
        time_elapsed = []
        for f in test_points:
            startTime = time.time()
            test_tree.query(f, k=1, p=p)
            elapsedTime = time.time() - startTime
            time_elapsed.append(elapsedTime)
        x.append((l+1)*batch)
        y.append(np.mean(time_elapsed)*1000)
        print('[{}] finished in avg {:.2f} ms'.format("search in "+str((l+1)*batch)+" points", np.mean(time_elapsed) * 1000))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1000, 100))
    ax.set_yticks(np.arange(0, 6, 1))
    plt.scatter(x, y)
    plt.grid()
    plt.show()
    """

    dist_1, nb_of_1 = tree_2.query(feats_1, k=k, p=p)
    dist_2, nb_of_2 = tree_1.query(feats_2, k=k, p=p)
    
    ## cross check ##
    mutual = nb_of_2[:, 0][nb_of_1[:, 0]] == np.arange(num_1)
    ## threshold ##
    is_pass = dist_1[:, 0] < 10.0*np.min(dist_1)
    is_good = np.logical_and(mutual, is_pass)

    ## creates return values ##
    pair_num = np.sum(is_good)
    idxs = np.zeros(shape=(pair_num, 2), dtype='int')
    idxs[:, 0] = np.arange(num_1)[is_good]
    idxs[:, 1] = nb_of_1[:, 0][is_good]

    ## ratio test ##
    if p > 1:
        comply_1 = dist_1[idxs[:, 0], 0] < 0.7*dist_1[idxs[:, 0], 1]
        comply_2 = dist_2[idxs[:, 1], 0] < 0.7*dist_2[idxs[:, 1], 1]
        comply = np.logical_and(comply_1, comply_2)
    idxs = idxs[comply]
    """
    idxs = np.zeros(shape=((num_1 + num_2)*k, 2), dtype='int')
    idxs[:num_1*k, 0] = np.repeat(np.arange(num_1), k)
    idxs[:num_1*k, 1] = nb_of_1.reshape(-1)
    idxs[num_1*k:, 1] = np.repeat(np.arange(num_2), k)
    idxs[num_1*k:, 0] = nb_of_2.reshape(-1)

    ## remove duplicates ##
    pairs = [tuple(row) for row in idxs]
    unique_pairs = set(pairs)
    idxs = np.array(list(unique_pairs))
    """
    return idxs


if __name__ == "__main__":
    imgs, fs = Load_Data('./photos/MHa', './photos/MHa/f.txt', '.JPG')
    imgs_proj = []
    imgs_proj_mask = []
    for i in range(imgs.shape[0]):
        H = imgs[i].shape[0]
        W = imgs[i].shape[1]
        resize_img = cv2.resize(imgs[i], (int(W*1400/H), 1400))
        new_img, new_img_mask = cylindrical_projection(resize_img, fs[i]*2)
        imgs_proj.append(new_img)
        imgs_proj_mask.append(new_img_mask)
    id = 1
    gray0 = cv2.cvtColor(imgs_proj[id], cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(imgs_proj[id+1], cv2.COLOR_BGR2GRAY)
    H = gray0.shape[0]
    W0 = gray0.shape[1]
    W1 = gray1.shape[1]
    ## MOPS features ##
    feats0 = Build_pyramid(imgs_proj[id])
    feats1 = Build_pyramid(imgs_proj[id+1])
    des0 = np.array([feat.descriptor for feat in feats0])
    des1 = np.array([feat.descriptor for feat in feats1])

    ## self implemented matching ##
    id_pairs = genMatchPairs(des0, des1, k=2, p=2)
    my_img = np.zeros((H, W0+W1, 3), dtype='uint8')
    my_img[:, :W0, :] = imgs_proj[id]
    my_img[:, W0:, :] = imgs_proj[id+1]
    for feat in feats0:
        cv2.circle(my_img, (feat.x, feat.y), 3, (255,0,0), 2)
    for feat in feats1:
        cv2.circle(my_img, (feat.x+W0, feat.y), 3, (255,0,0), 2)
    for pair in id_pairs:
        pt0 = (feats0[pair[0]].x, feats0[pair[0]].y)
        pt1 = (feats1[pair[1]].x+W0, feats1[pair[1]].y)
        cv2.circle(my_img, pt0, 3, (0,255,0), 2)
        cv2.circle(my_img, pt1, 3, (0,255,0), 2)
        cv2.line(my_img,pt0,pt1,(0,255,0),1)

    ## feature detection by opencv sift ##
    sift = cv2.xfeatures2d.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0,None)
    kp1, des1 = sift.detectAndCompute(gray1,None)

    ## cv version matching ##
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm =FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des0,des1,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    cv_img = cv2.drawMatchesKnn(gray0,kp0,gray1,kp1,matches,None,**draw_params)


    cv2.imshow('my_img', my_img)
    cv2.imshow('cv_img', cv_img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
