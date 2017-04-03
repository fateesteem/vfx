import cv2
import numpy as np
import os



M = np.array([[1,0,0],
              [0,1,0]], dtype='float32')


def computBitmap(img):
    median = np.median(img)
    tb = img > median
    eb = (img > median + 4) | (img < median - 4)
    return tb, eb

def getExpShift(img_base, img_align, shift_bits):
    assert len(img_base.shape) == 2
    assert img_base.shape == img_align.shape
    rows = img_base.shape[0]
    cols = img_base.shape[1]
    ## recursive call to get shift dx, dy ##
    if shift_bits > 0:
        img_base_reduce = cv2.resize(img_base, None, fx=0.5, fy=0.5)
        img_align_reduce = cv2.resize(img_align, None, fx=0.5, fy=0.5)
        cur_shift = getExpShift(img_base_reduce, img_align_reduce, shift_bits-1)
        cur_shift = cur_shift*2
    else:
        cur_shift = np.array([0, 0], dtype='float32')

    ## comput mtb and mask ##
    tb_base, eb_base = computBitmap(img_base)
    tb_align, eb_align = computBitmap(img_align)

    ## get optimize dx, dy ##
    min_err = rows*cols
    shift_ret = None
    """
    tb_align_res = None
    eb_align_res = None
    diff_res = None
    diff_ori = None
    """
    for i in [0, 1, -1]:
        for j in [0, 1, -1]:
            transform_m = M.copy()
            transform_m[:, 2] = [i, j] + cur_shift
            tb_align_shift = cv2.warpAffine(tb_align.astype('uint8'), transform_m, (cols, rows)).astype('bool')
            eb_align_shift = cv2.warpAffine(eb_align.astype('uint8'), transform_m, (cols, rows)).astype('bool')
            diff = ((tb_base ^ tb_align_shift) & eb_base) & eb_align_shift
            err = np.sum(diff)
            if err < min_err:
                shift_ret = transform_m[:, 2]
                min_err = err
                """
                tb_align_res = tb_align_shift
                eb_align_res = eb_align_shift
                diff_res = diff
                diff_ori = tb_base ^ tb_align_shift
                """
    """
    cv2.imshow("tb_base", tb_base.astype('uint8')*255)
    cv2.imshow("tb_align", tb_align_res.astype('uint8')*255)
    cv2.imshow("eb_base", eb_base.astype('uint8')*255)
    cv2.imshow("eb_align", eb_align_res.astype('uint8')*255)
    cv2.imshow("diff_ori", diff_ori.astype('uint8')*255)
    cv2.imshow("diff", diff_res.astype('uint8')*255)
    key  = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
    """
    return shift_ret

def MTBAlignment(imgs, shift_range):
    levels = np.floor(np.log2(shift_range))
    num_img = imgs.shape[0]
    H = imgs.shape[1]
    W = imgs.shape[2]
    imgs_align = [imgs[0]]
    imgs_align_gray = [cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)]
    pre_shift_ret = np.array([0, 0], dtype='float32')
    for i in range(1, num_img):
        cur_gray = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        shift_ret = getExpShift(imgs_align_gray[i-1], cur_gray, levels)
        transform_m = M.copy()
        transform_m[:, 2] = shift_ret + pre_shift_ret
        img_align = cv2.warpAffine(imgs[i], transform_m, (W, H))
        cur_gray_align = cv2.warpAffine(cur_gray, transform_m, (W, H))
        print("img"+str(i)+" shift_ret:", transform_m[:, 2])
        imgs_align.append(img_align)
        imgs_align_gray.append(cur_gray)
        pre_shift_ret = shift_ret
    return np.array(imgs_align)

def RandomShift(img):
    rows = img.shape[0]
    cols = img.shape[1]
    transform_m = M.copy()
    transform_m[:, 2] = np.random.choice(range(-50, 51), size=2)
    img_shift = cv2.warpAffine(img, transform_m, (cols, rows))
    return img_shift, transform_m[:, 2]

if __name__ == "__main__":
    """
    img1 = cv2.imread("./test/images/1/10.JPG")
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img1_shift, shift_true = RandomShift(img1_gray)
    img2 = cv2.imread("./test/images/1/11.JPG")
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    shift_ret = getExpShift(img1_gray, img2_gray, np.floor(np.log2(50)))
    print("exp shift:", shift_ret)

    transform_m = M.copy()
    transform_m[:, 2] = shift_ret
    img2_align = cv2.warpAffine(img2, transform_m, img1_gray.shape[::-1])
    cv2.imshow('base img', img1)
    #cv2.imshow('shift img', img_shift)
    cv2.imshow('align img', img2_align)
    key  = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
    """
    imgs = []
    for file in os.listdir("./Memorial_SourceImages/"):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join("./Memorial_SourceImages", file))
            print("load", os.path.join("./Memorial_SourceImages", file))
            imgs.append(img)
    imgs_align = MTBAlignment(np.array(imgs), 10)
    cv2.imshow('base img', imgs_align[0])
    cv2.imshow('align img', imgs_align[1])
    key  = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
