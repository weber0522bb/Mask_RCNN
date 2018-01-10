# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
sys.path.insert(0,'/media/HDD/cvteam14/Mask_RCNN/facial_keypoint_detection')

import face
from PIL import Image
import numpy as np
from PIL import ImageDraw
import scipy.misc
import skimage
import sys
#sys.path.insert(0,'/media/HDD/cvteam14/Mask_RCNN/proj4_2/code')
sys.path.append("..")
from  proj4_2.code.proj4 import face_recog
def print_img(X):
    array2 = np.array(X)
    for i in array2:
        print (i)


def read_img():
    path = '/data/VSLab/cvteam14/test1_point.png'
    img = Image.open(path)
    print ('img:',img)
    '''
    img = scipy.misc.toimage(img)
    print ('img1:',img, np.size(img))
    img = Image.fromarray(img, 'RGB')
    print ('img2:',img)'''
    L = img.convert('L')
    print ('L:',L)
    out = L.resize((96, 96))
    print ('out:',out)
    #draw = ImageDraw.Draw(out)
    im_array = np.array(out)
    print ('im_array:',im_array)
    '''im_array = skimage.color.rgb2grey(im_array)'''
    im_array = im_array / 255.0
    im_array = im_array.reshape(-1, 96, 96, 1)
    print (im_array, im_array.shape)
    return im_array

sess = tf.InteractiveSession()
y_conv, rmse = face.model()
train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)
sess.run(tf.initialize_all_variables())

ckpt = tf.train.get_checkpoint_state('kaggle/')
if ckpt and ckpt.model_checkpoint_path:
    saver = tf.train.Saver()
    print(ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
 
else: 
    print('NO CHECKPONIT')
    '''
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "/media/HDD/cvteaam14/facial_point_detection/kaggle/checkpoint")
    
    '''
    '''
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('/media/HDD/cvteaam14/facial_point_detection/kaggle/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('/media/HDD/cvteaam14/facial_point_detection/kaggle/'))
        sess.run(tf.global_variables_initializer())
    '''
X = read_img()
y_pred = []

y_batch = y_conv.eval(feed_dict={face.x: X, face.keep_prob: 1.0})
print (y_batch)
y_pred.extend(y_batch)
print ('predict test image done!')
y_pred = y_pred[0][:]*96

