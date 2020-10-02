# -*- coding:gb18030 -*-

import dlib
import cv2
import numpy as np
import math
from numba import jit
import argparse
import sys
import os

# 使用dlib自带的frontal_face_detector作为特征提取器
detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
path = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(path,predictor_path)
predictor = dlib.shape_predictor(predictor_path)


# 提取人脸68个特征点
def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)
    print(len(rects))
    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        for idx,point in enumerate(land_marks_node):
            # 68点坐标
            pos = (point[0,0],point[0,1])
            # 利用cv2.circle给每个特征点画一个圈，共68个
            print(idx)
            cv2.circle(img_src, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_src, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        land_marks.append(land_marks_node)
        # print(len(land_marks))
    return land_marks


'''
方法： Interactive Image Warping 局部平移算法
'''


@jit(nopython=True)
def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copy_img = srcImg.copy()

    # 计算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)
            if (distance < ddradius):
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio

                # 映射原位置
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)

                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg, UX, UY)
                # 改变当前 i ，j的值
                copy_img[j, i] = value

    return copy_img

# 双线性插值法
@jit(nopython=True)
def BilinearInsert(src, ux, uy):
    h, w, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1] * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2] * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1] * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2] * (ux - float(x1)) * (uy - float(y1))

        insert_value = part1 + part2 + part3 + part4
        return insert_value.astype(np.int8)

def face_thin_auto(src,args):
    landmarks = landmark_dec_dlib_fun(src)
    # print(len(landmarks))
    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        print(0)
        return
    thin_image=src
    for landmarks_node in landmarks:
        p = landmarks_node
        end_pt = landmarks_node[33]
        dis = []
        cv2.rectangle(src, (p[0][0,0],p[0][0,1]), (p[16][0,0], p[16][0,1]), (255, 0, 0), 5)

        # 计算两点的距离作为瘦脸距离
        for i in range(17):
            if i ==8:temp = 0
            else:temp = math.sqrt((p[i][0, 0] - p[i+1][0, 0]) * (p[i][0, 0] - p[i+1][0, 0]) +
                (p[i][0, 1] - p[i+1][0, 1]) * (p[i][0, 1] - p[i+1][0, 1]))
            dis.append(temp)
        dis[16] = dis[15]
        ratio = args.ratio

        # 瘦左边脸
        thin_image = localTranslationWarp(thin_image, p[0][0, 0], p[0][0, 1],
                                          end_pt[0, 0], end_pt[0, 1],(1+ratio)*dis[0])
        for i in range(1,11):
            # 不瘦下巴
            if i==6 or i==8 or i==9 or i==10:continue
            # 瘦左边脸
            if i < 7:
                thin_image = localTranslationWarp(thin_image, p[i][0, 0], p[i][0, 1],
                                                  end_pt[0, 0], end_pt[0, 1],(1+ratio)*dis[i])
            # 瘦右边脸
            if i > 7:
                thin_image = localTranslationWarp(thin_image, p[i][0, 0], p[i][0, 1],
                                                  end_pt[0, 0], end_pt[0, 1],(1+ratio)*dis[i])
        # thin_image = localTranslationWarp(thin_image, p[4][0, 0], p[4][0, 1],
        #                                   end_pt[0, 0], end_pt[0, 1], (1 + ratio) * dis[4])
    return thin_image

def main(args):
    if args.image_file:
        image_files = os.listdir(args.image_file)
        for image_path in image_files:
            print(image_path)
            src = cv2.imread(os.path.join(args.image_file, image_path))
            thin_image = face_thin_auto(src,args)
            cv2.imwrite(os.path.join(args.output_file, image_path), thin_image)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float,
                        help='The degree of thin face', default=0.5)
    parser.add_argument('--image_file', type=str,
                        help='image file', default="./test")
    parser.add_argument('--output_file', type=str,
                        help='output file', default="./test_output")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))