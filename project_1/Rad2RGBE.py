import numpy as np


def Rad2RGBE(img_rad):
    img_rad = img_rad[:, :, ::-1] ## BGR2RGB
    assert len(img_rad.shape) == 3
    H = img_rad.shape[0]
    W = img_rad.shape[1]
    max_pixvals = np.max(img_rad, axis=2, keepdims=True)
    max_mantissa, exponent = np.frexp(max_pixvals)
    mantissa = max_mantissa * img_rad[:, :, 0:3] / max_pixvals * 256.0
    img_rgbe = np.zeros((H, W, 4), dtype='uint8')
    img_rgbe[:, :, 0:3] = np.around(mantissa)
    img_rgbe[:, :, 3] = np.around(exponent[:,:,0]+128)

    return img_rgbe
