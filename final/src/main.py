import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

from Tkinter import *

import tkFileDialog
import cv2

from MVCCloner import MVCCloner

def select_source_image(A):

    global src_img_path
    src_img_path = tkFileDialog.askopenfilename(title='Please select a src to analyze',filetypes=[('Jpg files','*.jpg'), ('Jpeg file','*.jpeg'), ('Png file', '*.png')])
    if len(src_img_path)>0:
      A.select()
   

def select_target_image(AA):
   
    global target_img_path
    target_img_path = tkFileDialog.askopenfilename(title='Please select a target to analyze',filetypes=[('Jpg files','*.jpg'), ('Jpeg file','*.jpeg'), ('Png file', '*.png')])
    if len(target_img_path)>0:
      AA.select()
    

def compute_image_cloning():
    mvc_config = {'hierarchic': True,
                'base_angle_Th': 0.75,
                'base_angle_exp': 0.8,
                'base_length_Th': 2.5,
                'adaptiveMeshShapeCriteria': 0.125,
                'adaptiveMeshSizeCriteria': 0.,
                'min_h_res': 16.}

   
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
    src_img_path = None
    target_img_path = None

    root = Tk()
    root.geometry("550x300+300+150")
    root.resizable(width=True, height=True)
    panelA = None
    panelB = None

    var = IntVar()
    A = Checkbutton(root, text="Done", variable=var)

    var2 = IntVar()
    AA = Checkbutton(root, text="Done", variable=var2)

    btn = Button(root, text="Select source image", command= lambda: select_source_image(A),height =2, width = 30)

    btn.pack()
    A.pack()


    btn2 = Button(root, text="Select target image",command=lambda: select_target_image(AA),height =2, width = 30)
    btn2.pack()
    AA.pack()

    btn3 = Button(root, text="Compute Image Cloning",command=compute_image_cloning, height = 2, width = 30)
    btn3.pack()

    root.mainloop()

    #mvc_cloner = MVCCloner(src_img_path, target_img_path, output_path, mvc_config)
    #mvc_cloner.GetPatch()
    #mvc_cloner.run()
