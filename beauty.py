# -*- coding:gb18030 -*-

import argparse
import cv2
import os
import sys
from age_gender import age_gender_estimate
from remove import remove_moles,buffing
from thin_face import thinFace
from upperbody import inference

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1', type=int,
                        help='磨皮程度', default=3)
    parser.add_argument('--v2', type=int,
                        help='细节程度', default=5)
    parser.add_argument('--p', type=float,
                        help='图像融合参数', default=0.2)
    parser.add_argument('--ratio', type=float,
                        help='The degree of thin face', default=0.3)
    parser.add_argument('--image_file', type=str,
                        help='image file', default="./img_data")
    parser.add_argument('--output_file', type=str,
                        help='output file', default="./output")
    parser.add_argument('--estimate_output_file', type=str,
                        help='estimate output file', default="./estimate_output")
    parser.add_argument('--thin_face', type=bool,
                        help='image file', default=True)
    parser.add_argument('--remove_moles', type=bool,
                        help='image file', default=True)
    parser.add_argument('--buffing', type=bool,
                        help='buffing', default=False)
    parser.add_argument('--inference', type=bool,
                        help="clipping half of one's body ", default=True)
    parser.add_argument('--age_gender_estimate', type=bool,
                        help='estimate gender ande age', default=True)
    parser.add_argument('--gpu_fraction', type=float,
                        help='estimate gender ande age', default=1.0)
    parser.add_argument("--model_path", "--M", type=str,
                        help="Model Path",  default="./models", )
    return parser.parse_args(argv)

def main(args):
    if args.image_file:
        image_files = os.listdir(args.image_file)
        for image_path in image_files:
            print(image_path)
            img_result = cv2.imread(os.path.join(args.image_file, image_path))
            if args.age_gender_estimate:
                sess, age, gender, train_mode, images_pl = age_gender_estimate.load_network(args.model_path)
                img_result = age_gender_estimate.estimate(sess, age, gender, train_mode, images_pl, img_result)
                cv2.imwrite(os.path.join(args.output_file, image_path), img_result)
            if args.inference:
                img_result = inference.clipping(args,img_result)
            if args.thin_face:
                img_result = thinFace.face_thin_auto(img_result,args)
            if args.remove_moles:
                mask_3 = remove_moles.blob(img_result)
                mask_1 = cv2.cvtColor(mask_3, cv2.COLOR_BGR2GRAY)
                img_result = remove_moles.inpaint(img_result, mask_1, mask_3)
            if args.buffing:
                img_result = buffing.beauty_face1(img_result, args)
            cv2.imwrite(os.path.join(args.output_file, image_path), img_result)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))