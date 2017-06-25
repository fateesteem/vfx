import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import argparse
from MVCCloner import MVCCloner

def select_image():
    mvc_config = {'hierarchic': True,
                'base_angle_Th': 0.75,
                'base_angle_exp': 0.8,
                'base_length_Th': 2.5,
                'adaptiveMeshShapeCriteria': 0.125,
                'adaptiveMeshSizeCriteria': 0.,
                'min_h_res': 16.}

    src_img_path = tkFileDialog.askopenfilename(title='Please select a src to analyze')
    target_img_path = tkFileDialog.askopenfilename(title='Please select a target to analyze')
    if len(src_img_path) > 0 and len(target_img_path) > 0:
        mvc_cloner = MVCCloner(src_img_path, target_img_path, './out.jpg', mvc_config)
        mvc_cloner.GetPatch()
        mvc_cloner.run()

if __name__ == '__main__':
    ## argument parse ##
    #parser = argparse.ArgumentParser(description="Mean-Value Seamless Cloning")
    #parser.add_argument("src", help="The path to source image")
    #parser.add_argument("target", help="The path to target image")
    #parser.add_argument("-o", "--output", help="The path to save the cloning image", default="./out.jpg")
    #args = parser.parse_args()

    # set arguments #
    #src_img_path = args.src
    #target_img_path = args.target
    #output_path = args.output


    root = Tk()
    panelA = None
    panelB = None

    btn = Button(root, text="Select an image", command=select_image)
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    root.mainloop()

    #mvc_cloner = MVCCloner(src_img_path, target_img_path, output_path, mvc_config)
    #mvc_cloner.GetPatch()
    #mvc_cloner.run()
