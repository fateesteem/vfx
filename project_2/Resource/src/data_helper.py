import numpy as np
import cv2
import os


def Load_Data(imgs_dir, focal_file, img_type):
    focal = open(focal_file)
    f_map = {}
    for line in focal:
        tokens = line.split('\t')
        fractions = tokens[1][:-1].split('/')
        assert len(fractions) <= 2
        f_map[tokens[0]] = float(fractions[0]) / float(fractions[1]) if len(fractions) == 2 else float(fractions[0])
    images = []
    focal_lengths = []
    for file in sorted(os.listdir(imgs_dir)):
        if file.endswith(img_type) and not file.startswith('._') :#and len(file) == 10:
            f = f_map[file[-6:-4]]
            img = cv2.imread(os.path.join(imgs_dir, file))
            print("load image "+file+' - f:', f)
            images.append(img)
            focal_lengths.append(f)
    if len(images) == 0:
        raise Exception("No images loaded!")
    return np.array(images), np.array(focal_lengths)

