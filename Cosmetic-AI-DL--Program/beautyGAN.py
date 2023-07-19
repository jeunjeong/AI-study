import tensorflow.compat.v1 as tf
import numpy as np
import os
import glob
from imageio import imread, imsave

import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default =
os.path.join('C:/Users/SSAFY/Desktop/hi/AI-study/Cosmetic-AI-DL--Program/imgs', 'no_makeup', 'xfsy_0068.png'), help = 'path to the no_makeup image')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256
no_makeup = cv2.resize(cv2.imread(args.no_makeup), (img_size, img_size))
X_img = np.expand_dims(preprocess(no_makeup), 0)
makeups = glob.glob(os.path.join('C:/Users/SSAFY/Desktop/hi/AI-study/Cosmetic-AI-DL--Program/imgs', 'makeup', '*.*'))

result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 * img_size, :img_size] = no_makeup / 255.

with tf.compat.v1.Session() as ses:
    ses.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(os.path.join('C:/Users/SSAFY/Desktop/hi/AI-study/Cosmetic-AI-DL--Program/model', 'model.meta'))
    saver.restore(ses, tf.train.latest_checkpoint('C:/Users/SSAFY/Desktop/hi/AI-study/Cosmetic-AI-DL--Program/model'))

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    Y = graph.get_tensor_by_name('Y:0')
    Xs = graph.get_tensor_by_name('generator/xs:0')
    for i in range(len(makeups)):
        makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
        Y_img = np.expand_dims(preprocess(makeup), 0)

        # 입력과 결과를 주고 예측합니다.
        Xs_ = ses.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = deprocess(Xs_)
        result[:img_size, (i + 1) * img_size: (i + 2)
               * img_size] = makeup / 255.
        result[img_size: 2 * img_size,
               (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]
    imsave('C:/Users/SSAFY/Desktop/hi/AI-study/Cosmetic-AI-DL--Program/result.jpg', result)