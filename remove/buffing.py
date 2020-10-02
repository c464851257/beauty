import numpy as np
import cv2
import argparse
import sys
import os

output_folder = "./test_output"
def beauty_face1(src,args):
    '''
    Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
    '''
    v1 = args.v1
    v2 = args.v2
    dx = v1 * 5  # 双边滤波参数之一
    fc = v1 * 12.5  # 双边滤波参数之一
    p = args.p #图像融合参数
    # HighPass = cv2.imread('2.jpg')
    HighPass = cv2.bilateralFilter(src, dx, fc, fc)#双边滤波
    if v2!=0:
        HighPass = beauty_face2(HighPass,src,v2,p)
    return HighPass

def beauty_face2(HighPass,src,v2,p):#增加图片细节
    temp2 = cv2.subtract(HighPass, src)
    temp2 = cv2.add(temp2, (10, 10, 10, 128))
    temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 5)  # 高斯模糊
    # cv2.imwrite(os.path.join(output_folder, '1.jpg'), temp3)
    temp4 = cv2.subtract(cv2.add(cv2.add(temp3, temp3), src), (10, 10, 10, 255))
    dst = cv2.addWeighted(src, p, temp4, 1 - p, 0.0)  # 图像融合
    dst = cv2.add(dst, (10, 10, 10, 255))
    return dst

def init(args):
    if args.image:
        image_pths = os.listdir(args.image)
        for image_pth in image_pths:
            print(image_pth)
            img = cv2.imread(os.path.join(args.image, image_pth))
            blur4 = beauty_face1(img,args)
            cv2.imwrite(os.path.join(output_folder, image_pth), blur4)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--v1', type=int,
                        help='磨皮程度', default=4)
    parser.add_argument('--v2', type=int,
                        help='细节程度', default=0)
    parser.add_argument('--p', type=float,
                        help='图像融合参数', default=0.2)
    parser.add_argument('--image',type=str,
                        help='图像路径',default='./test')
    return parser.parse_args(argv)

if __name__ == "__main__":
    init(parse_arguments(sys.argv[1:]))