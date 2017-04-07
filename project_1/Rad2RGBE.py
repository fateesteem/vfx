import numpy as np


def Rad2RGBE(img_rad):
    assert len(img_rad.shape) == 3
    H = img_rad.shape[0]
    W = img_rad.shape[1]
    max_pixvals = np.max(img_rad, axis=2)
    mantissa, exponent = np.frexp(max_pixvals)
    scaled_mantissa = mantissa*256.0/max_pixvals
    img_rgbe = np.zeros((H, W, 4), dtype='uint8')
    img_rgbe[:, :, 0:3] = np.around(img_rad[:, :, ::-1]*scaled_mantissa[:, :, np.newaxis])
    img_rgbe[:, :, 3] = np.around(exponent +128)

    return img_rgbe
