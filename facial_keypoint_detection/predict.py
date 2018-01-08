# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
sys.path.insert(0,'/media/HDD/cvteam14/Mask_RCNN/facial_keypoint_detection')

import face
from PIL import Image
import numpy as np
from PIL import ImageDraw


def print_img(X):
    array2 = np.array(X)
    for i in array2:
        print (i)


def read_img(img):
    '''
    path = 'PATH'
    im = Image.open(path)
    '''
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    L = img.convert('L')
    out = L.resize((96, 96))
    #draw = ImageDraw.Draw(out)
    im_array = np.array(out)
    im_array = im_array / 255.0
    im_array = im_array.reshape(-1, 96, 96, 1)
    print (im_array, im_array.shape)
    return im_array

def predict(img):
    sess = tf.InteractiveSession()
    y_conv, rmse = face.model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)
    ckpt = tf.train.get_checkpoint_state('kaggle/')
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)

    X = read_img(img)
    y_pred = []

    y_batch = y_conv.eval(feed_dict={face.x: X, face.keep_prob: 1.0})
    print (y_batch)
    y_pred.extend(y_batch)
    print ('predict test image done!')
    return y_pred
