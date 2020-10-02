import tensorflow as tf
import numpy as np
import os
from scipy import misc
import argparse
import cv2
import sys

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
output_folder = "./test_output"
path = os.path.dirname(os.path.abspath(__file__))
checkpoint = os.path.join(path,'salience_model')
meta_path = os.path.join(path,'meta_graph/my-model.meta')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--rgb', type=str,
                        help='input rgb', default=None)
    parser.add_argument('--rgb_folder', type=str,
                        help='input rgb', default="./test")
    parser.add_argument('--gpu_fraction', type=float,
                        help='how much gpu is needed, usually 4G is enough', default=1.0)
    return parser.parse_args(argv)

def rgba2rgb(img):
    return img[:, :, :3] * np.expand_dims(img[:, :, 3], 2)

def clipping(args):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        image_batch = tf.get_collection('image_batch')[0]
        pred_mattes = tf.get_collection('mask')[0]

        if args.rgb_folder:
            rgb_pths = os.listdir(args.rgb_folder)
            for rgb_pth in rgb_pths:
                print(rgb_pth)
                rgb = misc.imread(os.path.join(args.rgb_folder, rgb_pth))
                if rgb.shape[2] == 4:
                    rgb = rgba2rgb(rgb)
                origin_shape = rgb.shape
                img = np.expand_dims(
                    misc.imresize(rgb.astype(np.uint8), [320, 320, 3], interp="nearest").astype(np.float32) - g_mean, 0)

                feed_dict = {image_batch: img}
                pred_alpha = sess.run(pred_mattes, feed_dict=feed_dict)
                final_alpha = misc.imresize(np.squeeze(pred_alpha), origin_shape)
                final_alpha[final_alpha >= 25] = 255
                final_alpha[final_alpha < 25] = 0

                _, y, _, _ = cv2.boundingRect(final_alpha)# 获取头部y值
                x, _, x2, y2 = upperBody(rgb)
                #TODO
                if x == 0:
                    y=0
                    dst = rgb[y:y2, x:x2]
                else:
                    print(x,y,x2,y2,rgb.shape[1], rgb.shape[0])
                    x, y, x2, y2 = keepRatio(x,y,x2,y2,rgb.shape[1],rgb.shape[0])
                    print(rgb.shape[1],rgb.shape[0])
                    dst = rgb[y:y2,x:x2]
                # cv2.rectangle(im, (x, y), (x2, y2), (255, 0, 0), 5)
                # return dst
                misc.imsave(os.path.join(output_folder, rgb_pth), dst)
        else:
            rgb = misc.imread(args.rgb)
            if rgb.shape[2] == 4:
                rgb = rgba2rgb(rgb)
            origin_shape = rgb.shape[:2]
            rgb = np.expand_dims(
                misc.imresize(rgb.astype(np.uint8), [320, 320, 3], interp="nearest").astype(np.float32) - g_mean, 0)

            feed_dict = {image_batch: rgb}
            pred_alpha = sess.run(pred_mattes, feed_dict=feed_dict)
            final_alpha = misc.imresize(np.squeeze(pred_alpha), origin_shape)
            final_alpha[final_alpha >= 25] = 255
            final_alpha[final_alpha < 25] = 0
            misc.imsave(os.path.join(output_folder, 'alpha.png'), final_alpha)
def keepRatio(x, y, w, y2, shape_1, shape_0):
    x_center = x + w / 2
    scale = shape_1 / shape_0  # 原图宽高比
    h_new = y2 - y
    w_new = h_new * scale
    x = int(x_center - w_new / 2)
    x2 = int(x_center + w_new / 2)
    #TODO
    if x2 > shape_1 or x < 0:
        x2, y2 = shape_1,shape_0
        x, y = 0,0

    return x, y, x2, y2

def upperBody(im):
    if not os.path.isdir('model'):
        os.mkdir("model")
    # Load a Caffe Model
    protoFile = os.path.join(path,"model/pose_deploy_linevec_faster_4_stages.prototxt")
    weightsFile = os.path.join(path,"model/pose_iter_440000.caffemodel")

    # Specify number of points in the model
    nPoints = 18
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Read Image
    # global im
    # im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    inWidth = im.shape[1]
    inHeight = im.shape[0]

    # Convert image to blob
    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)

    # Run Inference (forward pass)
    output = net.forward()

    # X and Y Scale
    scaleX = float(inWidth) / output.shape[3]
    scaleY = float(inHeight) / output.shape[2]

    # Empty list to store the detected keypoints
    points = []

    # Confidence treshold
    threshold = 0.1

    for i in range(nPoints):
        # Obtain probability map
        probMap = output[0, i, :, :]
        # print(probMap)

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold:
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    maxX, maxY = 0, 0
    minX, minY = list(points[0])[0], list(points[0])[1]
    # if points[8] or points[11]:# 检测到上半身关节
    #TODO
    if points[8] or points[11]:
        for i, p in enumerate(points):
            if p:
                p1 = list(p)
                if i == 9 or i == 10 or i == 12 or i == 13:# 剔除下半身
                    continue
                # 遍历所有关节获取最大值最小值
                if p1[0] > maxX:
                    maxX = p1[0]
                if p1[1] > maxY:
                    maxY = p1[1]
                if p1[0] < minX:
                    minX = p1[0]
        w = maxX - minX
    else:# 未检测到上半身关节
        h = inHeight
        w = inWidth
        return 0,0,w,h
    x,y,w,maxY = minX,minY,w,maxY
    return x,y,w,maxY

if __name__ == '__main__':
    print(parse_arguments(sys.argv[1:]))
    clipping(parse_arguments(sys.argv[1:]))
