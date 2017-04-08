from MTBAlignment import MTBAlignment
from HDRAssemble import Load_Data, Load_Data_test, Radiance_Map, Save_Rad
from response_Robertson import RadianceMapRobertson
from ToneMapping_durand import ToneMapping_durand
from ToneMapping import Tone_mapping
import matplotlib
#matplotlib.style.use('classic')
from matplotlib import pyplot as plt
import argparse
import numpy as np
import cv2

def drawer(rad, g):
    plt.figure()
    plt.plot(g[2], color = 'blue')
    plt.plot(g[1], color = 'green')
    plt.plot(g[0], color = 'red')
    plt.title('Reconstruct camera response curve')
    plt.savefig('response_curve.png')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 8))#, sharex=False, sharey=True)
    im = ax1.imshow(np.log(rad[:, :, 0]), aspect = 'auto')
    ax1.set_title('Blue')
    im = ax2.imshow(np.log(rad[:, :, 1]), aspect = 'auto')
    ax2.set_title('Green')
    im = ax3.imshow(np.log(rad[:, :, 2]), aspect = 'auto')
    ax3.set_title('Red')
    #plt.colorbar(im, ax = ax3)
    plt.suptitle('Reconstruct Radiance Map', fontsize=20)
    f.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('Rad.png')

def Create_HDR(imgs, d_ts, shift_range, ld, rad_save_path=None, method='durand', gsolver='Debevec', verbose=False):
    assert len(imgs.shape) == 4
    assert imgs.shape[3] == 3
    num_img = imgs.shape[0]
    H = imgs.shape[1]
    W = imgs.shape[2]

    ## images alignment ##
    imgs_align = MTBAlignment(imgs, shift_range=shift_range)
    imgs_b_align = imgs_align[:, :, :, 0]
    imgs_g_align = imgs_align[:, :, :, 1]
    imgs_r_align = imgs_align[:, :, :, 2]

    ## radiance map ##
    if gsolver == 'Debevec':
        RadianceMap = Radiance_Map
    elif gsolver == 'Robertson':
        RadianceMap = RadianceMapRobertson
    else:
        raise Exception('Wrong gsolver !!!')
    rad = np.zeros((H, W, 3), dtype='float')
    g = np.zeros((3, 256), dtype = np.float32)
    rad[:, :, 0], g[0, :] = RadianceMap(imgs_b_align, d_ts, l=ld)
    rad[:, :, 1], g[1, :] = RadianceMap(imgs_g_align, d_ts, l=ld)
    rad[:, :, 2], g[2, :] = RadianceMap(imgs_r_align, d_ts, l=ld)
    if verbose:
        drawer(rad, g)
    if not rad_save_path == None:
        Save_Rad(rad, rad_save_path)

    ## tone mapping ##
    if method == 'durand':
        img_hdr = ToneMapping_durand(rad)
    elif method == 'photo':
        img_hdr = Tone_mapping(rad)
    else:
        raise Exception("Unsupported Tone Mapping method!")

    return img_hdr


if __name__ == "__main__":
    ## argument parse ##
    parser = argparse.ArgumentParser(description="Create HDR Image!")
    parser.add_argument("dir", help="The directory that contains all image files")
    parser.add_argument("speed", help="The path to shutter speed txt file")
    parser.add_argument("format", help="Image format")
    parser.add_argument("output", help="The name of output results")
    parser.add_argument("-r", "--range", help="Alignment maximum shift range", type=int, default=20)
    parser.add_argument("-l", "--ld", help="Lambda factor of calculating response curve", default=250)
    parser.add_argument("-m", "--method", help="Algorithm for tone mapping", default='durand')
    parser.add_argument("-g", "--gsolver", help="Algorithm for reconstructing response curve", default='Debevec')
    parser.add_argument("-v", "--verbose", help="plot response curve & radiance map", default=False)
    args = parser.parse_args()

    ## set arguments ##
    imgs_dir = args.dir
    speed_filename = args.speed
    img_type = args.format
    output = args.output
    shift_range = args.range
    ld = args.ld
    method = args.method
    gsolver = args.gsolver
    verbose = args.verbose
    is_show = True

    ## laod images from dir ##
    images, d_ts = Load_Data(imgs_dir, speed_filename, img_type)
    #images, d_ts = Load_Data_test(imgs_dir, img_type)

    ## create HDR image ##
    img_hdr = Create_HDR(images, d_ts, shift_range, ld, rad_save_path=output+'.hdr'
            , method=method, gsolver=gsolver, verbose=verbose)
    cv2.imwrite(output+'.jpg', (img_hdr*255).astype('uint8'))

    ## show result ##
    if is_show:
        cv2.imshow('img_hdr', img_hdr) #cv2.resize(img_hdr, (840, int(H*840/W))))
        print("press q to quit...")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
