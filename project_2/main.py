import argparse
import numpy as np
import cv2
from data_helper import Load_Data


if __name__ == '__main__':
    ## argument parse ##
    parser = argparse.ArgumentParser(description="Create Panorama!")
    parser.add_argument("dir", help="The directory that contains all image files")
    parser.add_argument("focal", help="The path to focal length txt file")
    parser.add_argument("format", help="Image format")
    parser.add_argument("output", help="The name of output results")
    parser.add_argument("-m", "--method", help="Algorithm for coordinate transform", default='forward')
    parser.add_argument("-b", "--blending", help="Algorithm for image blending", default='Linear')
    parser.add_argument("-v", "--verbose", help="plot feature points", default=False)
    args = parser.parse_args()

    ## set arguments ##
    imgs_dir = args.dir
    f_filename = args.focal
    img_type = args.format
    output = args.output
    method = args.method
    blending_type = args.blending
    verbose = args.verbose
    is_show = True 
    
    imgs, fs = Load_Data(imgs_dir, f_filename, img_type)
    imgs_proj = []
    imgs_proj_mask = []

    if method == 'backward':
        from inverse_cylindrical_proj import inverse_cylindrical_projection as cylindrical_projection
        from backward_alignment import ImageStitching
    
    for i in range(imgs.shape[0]):
        new_img, new_img_mask = cylindrical_projection(imgs[i], fs[i])
        imgs_proj.append(new_img)
        imgs_proj_mask.append(new_img_mask)

    stitch_img = ImageStitching(imgs_proj, imgs_proj_mask, btype = blending_type)
    cv2.imwrite(output, stitch_img)

    if is_show:
        cv2.imshow('stitch', stitch_img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()






