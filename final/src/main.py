import argparse
from MVCCloner import MVCCloner


if __name__ == '__main__':
    ## argument parse ##
    parser = argparse.ArgumentParser(description="Mean-Value Seamless Cloning")
    parser.add_argument("src", help="The path to source image")
    parser.add_argument("target", help="The path to target image")
    parser.add_argument("-o", "--output", help="The path to save the cloning image", default="./out.jpg")
    args = parser.parse_args()

    # set arguments #
    src_img_path = args.src
    target_img_path = args.target
    output_path = args.output

    mvc_config = {'hierarchic': True,
                  'base_angle_Th': 0.75,
                  'base_angle_exp': 0.8,
                  'base_length_Th': 2.5,
                  'adaptiveMeshShapeCriteria': 0.125,
                  'adaptiveMeshSizeCriteria': 0.,
                  'min_h_res': 16.}

    mvc_cloner = MVCCloner(src_img_path, target_img_path, output_path, mvc_config)
    mvc_cloner.GetPatch()
    mvc_cloner.run()
