#!/usr/bin/python

# Standard imports
import cv2
import numpy as np
import dlib
import math
import os

# 提取人脸81个特征点
def landmark_dec_dlib_fun(img_src):
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_81_face_landmarks.dat'
    path = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(path, predictor_path)
    predictor = dlib.shape_predictor(predictor_path)
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    land_marks = []
    rects = detector(img_gray, 0)
    print(len(rects))
    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        land_marks.append(land_marks_node)
    return land_marks

# 获取人脸痣的范围
def nevus_range(land_mark):
    p = land_mark
    x_max, y_max = 0, 0
    x_min, y_min = p[0][0, 0], p[0][0, 1]
    # TODO
    # 鼻子范围
    by_up, by_down, bx_left, bx_right = p[30][0, 1], p[33][0, 1], p[31][0, 0], p[35][0, 0]

    left_eye_minx, left_eye_miny = p[17][0,0], p[17][0,1]
    left_eye_maxx, left_eye_maxy = 0,0
    right_eye_minx, right_eye_miny = p[22][0,0], p[22][0,1]
    right_eye_maxx,right_eye_maxy = 0,0
    mouth_minx, mouth_miny = p[48][0,0], p[48][0,1]
    mouth_maxx, mouth_maxy = 0,0
    # 左眼范围
    for i in [17,18,19,20,21,36,37,38,39,40,41]:
        if p[i][0, 0] > left_eye_maxx: left_eye_maxx = p[i][0, 0]
        if p[i][0, 1] > left_eye_maxy: left_eye_maxy = p[i][0, 1]
        if p[i][0, 0] < left_eye_minx: left_eye_minx = p[i][0, 0]
        if p[i][0, 1] < left_eye_miny: left_eye_miny = p[i][0, 1]
    # 右眼范围
    for i in [22,23,24,25,26,42,43,44,45,46,47]:
        if p[i][0, 0] > right_eye_maxx: right_eye_maxx = p[i][0, 0]
        if p[i][0, 1] > right_eye_maxy: right_eye_maxy = p[i][0, 1]
        if p[i][0, 0] < right_eye_minx: right_eye_minx = p[i][0, 0]
        if p[i][0, 1] < right_eye_miny: right_eye_miny = p[i][0, 1]
    # 嘴范围
    for i in range(48,60):
        if p[i][0, 0] > mouth_maxx: mouth_maxx = p[i][0, 0]
        if p[i][0, 1] > mouth_maxy: mouth_maxy = p[i][0, 1]
        if p[i][0, 0] < mouth_minx: mouth_minx = p[i][0, 0]
        if p[i][0, 1] < mouth_miny: mouth_miny = p[i][0, 1]
    # 脸范围
    for i in range(len(p)):
        if p[i][0, 0] > x_max: x_max = p[i][0, 0]
        if p[i][0, 1] > y_max: y_max = p[i][0, 1]
        if p[i][0, 0] < x_min: x_min = p[i][0, 0]
        if p[i][0, 1] < y_min: y_min = p[i][0, 1]
    remove = {'x_max': x_max, 'x_min'  : x_min,    'y_max' : y_max,   'y_min'   :y_min,
              'by_up': by_up, 'by_down': by_down, 'bx_left': bx_left, 'bx_right': bx_right,
              'left_eye_minx':left_eye_minx,'left_eye_miny':left_eye_miny,'left_eye_maxx':left_eye_maxx,
              'left_eye_maxy':left_eye_maxy,'right_eye_minx':right_eye_minx,'right_eye_miny':right_eye_miny,
              'right_eye_maxx':right_eye_maxx,'right_eye_maxy':right_eye_maxy,'mouth_minx':mouth_minx,
              'mouth_miny':mouth_miny,'mouth_maxx':mouth_maxx,'mouth_maxy':mouth_maxy}
    return remove

# 痣检测
def blob(image):
    img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    img[:, :, :] = image[:, :, :]
    im = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    im[:, :, :] = 0
    im = cv2.GaussianBlur(img, (5, 5), 1)
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.thresholdStep = 5
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20
    # params.maxArea = 550

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.4

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    # # Filter by Color
    params.filterByColor = True
    params.blobColor = 0

    # 创建一张单通道的图片
    img_circle = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_circle[:, :, :] = 0  # 设置为全透明
    land_marks = landmark_dec_dlib_fun(img)
    for land_mark in land_marks:
        remove = nevus_range(land_mark)
        #TODO
        length = remove['x_max'] - remove['x_min']
        height = remove['y_max'] - remove['y_min']
        face_area = length*height
        if face_area/(image.shape[0]*image.shape[1]) < 0.02:
            continue
        print("length, height")
        params.maxArea = int((length+height)/6)
        print(params.maxArea)
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(im)
        for i in range(len(keypoints)):
            #TODO
            x, y = int(keypoints[i].pt[0]), int(keypoints[i].pt[1])  # keypoints的中心坐标
            is_face = x >= remove['x_min']   and x <= remove['x_max']    and y >= remove['y_min'] and y <= remove['y_max']
            is_nose = x >= remove['bx_left'] and x <= remove['bx_right'] and y >= remove['by_up'] and y <= remove['by_down']
            is_left_eye = x >= remove['left_eye_minx'] and x <= remove['left_eye_maxx'] and y >= remove['left_eye_miny'] \
                          and y <= remove['left_eye_maxy']
            is_rigth_eye = x >= remove['right_eye_minx'] and x <= remove['right_eye_maxx'] and y >= remove['right_eye_miny'] \
                           and y <=remove['right_eye_maxy']
            is_moth = x >= remove['mouth_minx'] and x <= remove['mouth_maxx'] and y >= remove['mouth_miny'] \
                      and y <= remove['mouth_maxy']
            if is_face and not(is_nose) and not(is_left_eye) and not(is_rigth_eye) and not(is_moth):
                img_circle = cv2.circle(img_circle, (x, y), math.ceil(keypoints[i].size), 255, -1)
    img[:,:,0][img_circle[:,:,0]==0] = 0
    img[:,:,1][img_circle[:,:,0]==0] = 0
    img[:,:,2][img_circle[:,:,0]==0] = 0
    return img

# 祛痣
def inpaint(img, mask_1,mask_3):
    # mask_1是一个通道，mask_3是3个通道，用来算高频的
    dst_TELEA = cv2.inpaint(img, mask_1, 7, cv2.INPAINT_TELEA)
    img_low = cv2.GaussianBlur(img, (15, 15), 0)
    mask_high = np.int16(mask_3) - np.int16(img_low)

    mask_high[mask_high < 0] = 0

    dst_TELEA = dst_TELEA + mask_high
    return dst_TELEA

input_path="test/"
output_path="test_output/"
paths = os.listdir(input_path)
print(paths)
for path in paths:
    image = cv2.imread(input_path + path)
    mask_3 = blob(image)
    mask_1 = cv2.cvtColor(mask_3,cv2.COLOR_BGR2GRAY)
    dst_TELEA = inpaint(image,mask_1,mask_3)
    cv2.imwrite(output_path + path, dst_TELEA)